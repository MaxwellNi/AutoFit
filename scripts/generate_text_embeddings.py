#!/usr/bin/env python3
"""Generate LLM-based text embeddings for the offers_text corpus.

Uses GTE-Qwen2-1.5B-instruct (Alibaba NLP) — an LLM-native decoder embedding
model based on Qwen2 with bidirectional attention and instruction tuning.

Architecture: Qwen2-1.5B → last-token pooling → L2 normalization
MTEB score: 67.16 (English), competitive with 7B models
License: Apache-2.0 (fully open, commercial OK)
VRAM: ~3.3 GB at FP16 (fits any GPU easily)
Embedding dim: 1536 (Matryoshka support → PCA to 64/128 downstream)

Pipeline:
    1. Load offers_text.parquet (5.77M rows, 15 text columns)
    2. Concatenate key text fields per row → combined_text
    3. Hash-based deduplication → embed only unique texts
    4. Batch inference via transformers (no sentence-transformers needed)
    5. Map embeddings back to full corpus
    6. Save as parquet: (entity_id, crawled_date_day, text_emb_0..text_emb_N)

Usage:
    # On GPU node (V100/H100/L40S):
    python scripts/generate_text_embeddings.py \\
        --model Alibaba-NLP/gte-Qwen2-1.5B-instruct \\
        --batch-size 64 --max-length 512 --pca-dim 64 \\
        --output runs/text_embeddings/

    # With 7B model for higher quality:
    python scripts/generate_text_embeddings.py \\
        --model Alibaba-NLP/gte-Qwen2-7B-instruct \\
        --batch-size 16 --max-length 512 --pca-dim 64 \\
        --output runs/text_embeddings/
"""
import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

# Monkey-patch DynamicCache for transformers >=4.45 compatibility
# GTE-Qwen2's custom modeling_qwen.py calls get_usable_length(seq_len, layer_idx)
# which was removed in newer transformers. Replicate the old behavior.
from transformers import DynamicCache
if not hasattr(DynamicCache, "get_usable_length"):
    def _compat_get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = _compat_get_usable_length

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text columns to embed (ordered by information density)
# ---------------------------------------------------------------------------
PRIMARY_TEXT_COLS = [
    "headline",
    "title",
    "description_text",
    "company_description",
    "product_description",
]
SECONDARY_TEXT_COLS = [
    "financial_condition",
    "financial_forecasts",
    "financial_risks",
    "offering_purpose",
    "use_of_funds_text",
    "reasons_to_invest_text",
]
# Excluded: term_sheet (legal boilerplate), front_video_transcript (noisy),
#           updates_text/questions_text (temporal UGC, low signal)
ALL_TEXT_COLS = PRIMARY_TEXT_COLS + SECONDARY_TEXT_COLS


def load_text_data(pointer_path: str = "docs/audits/FULL_SCALE_POINTER.yaml") -> pd.DataFrame:
    """Load offers_text.parquet using freeze pointer."""
    ptr = yaml.safe_load(open(pointer_path))
    text_dir = ptr["offers_text"]["dir"]
    text_path = Path(text_dir) / "offers_text.parquet"
    logger.info(f"Loading text data from {text_path}")

    # Load only keys + text columns we need
    cols_to_load = ["entity_id", "snapshot_ts"] + [
        c for c in ALL_TEXT_COLS
    ]
    df = pd.read_parquet(text_path, columns=cols_to_load)
    logger.info(f"Loaded {len(df):,} rows, {df['entity_id'].nunique():,} entities")

    # Extract date from snapshot_ts
    df["crawled_date_day"] = pd.to_datetime(df["snapshot_ts"]).dt.strftime("%Y-%m-%d")
    return df


def combine_text_fields(df: pd.DataFrame, max_chars: int = 2048) -> pd.Series:
    """Combine text columns into a single string per row.

    Strategy: Concatenate primary fields first (most informative),
    then secondary fields, with section headers for structure.
    Truncate to max_chars to respect model context limits.

    NOTE: Uses object dtype (Python str) to avoid pandas Arrow backend
    OOM on .loc[] assignment with long strings (155K+ chars in raw data
    causes 2.55 TiB allocation via ArrowStringArray → numpy).
    """
    per_field_limit = max(200, max_chars // max(1, len(ALL_TEXT_COLS)))
    parts = []
    for col in ALL_TEXT_COLS:
        if col in df.columns:
            # Force object dtype to avoid Arrow string backend OOM
            clean = df[col].fillna("").astype("object").str[:per_field_limit]
            mask = clean != ""
            label = col.replace("_", " ").title()
            labeled = clean.copy()
            labeled.loc[mask] = label + ": " + clean.loc[mask]
            parts.append(labeled)

    combined = parts[0].copy()
    for p in parts[1:]:
        non_empty = p != ""
        combined.loc[non_empty] = combined.loc[non_empty] + " | " + p.loc[non_empty]

    # Remove excessive whitespace and truncate
    combined = combined.str.replace(r"\s+", " ", regex=True).str.strip()
    combined = combined.str[:max_chars]
    return combined


def deduplicate_texts(combined: pd.Series) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Hash-based deduplication. Returns (hash_per_row, unique_indices, unique_texts)."""
    logger.info("Computing text hashes for deduplication...")

    # Fast vectorized hashing
    hashes = combined.apply(lambda x: hashlib.md5(x.encode("utf-8", errors="replace")).hexdigest()[:16])

    # Find unique texts
    seen = {}
    unique_indices = []
    hash_to_idx = {}
    for i, h in enumerate(hashes):
        if h not in seen:
            seen[h] = len(unique_indices)
            unique_indices.append(i)
            hash_to_idx[h] = seen[h]

    unique_texts = [combined.iloc[i] for i in unique_indices]
    row_to_unique = np.array([seen[h] for h in hashes], dtype=np.int32)

    logger.info(
        f"Dedup: {len(combined):,} rows → {len(unique_texts):,} unique texts "
        f"({len(unique_texts)/len(combined)*100:.1f}%)"
    )
    return row_to_unique, np.array(unique_indices), unique_texts


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool embeddings using the last non-padding token (GTE-Qwen2 method)."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


@torch.no_grad()
def encode_texts(
    texts: List[str],
    model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    batch_size: int = 64,
    max_length: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """Encode texts using GTE-Qwen2 model via transformers API."""
    from transformers import AutoModel, AutoTokenizer

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Use eager attention for V100 compatibility (no FlashAttention-2)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    logger.info(f"Model loaded on {device} (dtype=float16, attn=eager)")

    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_idx = i // batch_size + 1

        # Skip completely empty texts
        batch_texts = [t if t.strip() else "empty" for t in batch_texts]

        batch_dict = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().float().numpy())

        if batch_idx % 100 == 0 or batch_idx == n_batches:
            elapsed = time.time() - t0
            rate = (i + len(batch_texts)) / elapsed
            eta = (len(texts) - i - len(batch_texts)) / max(rate, 1)
            logger.info(
                f"  Batch {batch_idx}/{n_batches} | "
                f"{rate:.1f} texts/sec | ETA: {eta/60:.1f} min"
            )

    model.cpu()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    result = np.vstack(all_embeddings)
    logger.info(f"Encoded {result.shape[0]:,} texts → shape {result.shape}")
    return result


def apply_pca(embeddings: np.ndarray, n_components: int = 64) -> Tuple[np.ndarray, object]:
    """Apply PCA dimensionality reduction to embeddings."""
    from sklearn.decomposition import PCA

    # Replace NaN/Inf with 0 (from empty text embeddings)
    nan_mask = ~np.isfinite(embeddings)
    n_nan = nan_mask.any(axis=1).sum()
    if n_nan > 0:
        logger.warning(f"PCA input: {n_nan} rows contain NaN/Inf, replacing with 0")
        embeddings = np.where(nan_mask, 0.0, embeddings)

    logger.info(f"Applying PCA: {embeddings.shape[1]} → {n_components} dims")
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA explained variance: {explained:.3f}")
    return reduced, pca


def main():
    parser = argparse.ArgumentParser(description="Generate LLM text embeddings")
    parser.add_argument("--model", default="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                        help="HuggingFace model name for embeddings")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max tokens per text (512 = ~2KB text)")
    parser.add_argument("--max-chars", type=int, default=2048,
                        help="Max chars for combined text before tokenization")
    parser.add_argument("--pca-dim", type=int, default=64,
                        help="PCA output dimensions (0 = no PCA, keep full 1536)")
    parser.add_argument("--output", default="runs/text_embeddings/",
                        help="Output directory for embedding parquet")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pointer", default="docs/audits/FULL_SCALE_POINTER.yaml")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Shard index for parallel processing (0 = all)")
    parser.add_argument("--n-shards", type=int, default=1,
                        help="Total number of shards")
    parser.add_argument("--combine-shards", action="store_true",
                        help="Combine embeddings_shard_*.npy files, apply PCA, and write final parquet")
    parser.add_argument("--skip-existing-shard", action="store_true",
                        help="In shard mode, exit successfully if this shard already exists")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load text data
    df = load_text_data(args.pointer)

    # 2. Combine text fields
    logger.info("Combining text fields...")
    combined = combine_text_fields(df, max_chars=args.max_chars)

    # 3. Deduplicate
    row_to_unique, unique_indices, unique_texts = deduplicate_texts(combined)
    del combined
    gc.collect()

    if args.combine_shards:
        if args.n_shards <= 1:
            raise ValueError("--combine-shards requires --n-shards > 1")
        logger.info(f"Combining {args.n_shards} embedding shards from {output_dir}")
        shard_arrays = []
        missing = []
        for shard_id in range(args.n_shards):
            shard_size = (len(unique_texts) + args.n_shards - 1) // args.n_shards
            start_i = shard_id * shard_size
            end_i = min(start_i + shard_size, len(unique_texts))
            expected = max(0, end_i - start_i)
            shard_path = output_dir / f"embeddings_shard_{shard_id}.npy"
            if not shard_path.exists():
                missing.append(str(shard_path))
                continue
            arr = np.load(shard_path)
            if arr.shape[0] != expected:
                raise ValueError(
                    f"Shard {shard_id} row mismatch: got {arr.shape[0]}, expected {expected} ({shard_path})"
                )
            shard_arrays.append(arr)
        if missing:
            raise FileNotFoundError("Missing embedding shard files: " + ", ".join(missing[:10]))
        raw_embeddings = np.vstack(shard_arrays)
        if raw_embeddings.shape[0] != len(unique_texts):
            raise ValueError(f"Combined rows {raw_embeddings.shape[0]} != unique texts {len(unique_texts)}")
        logger.info(f"Combined raw embeddings: {raw_embeddings.shape}")

        if args.pca_dim > 0 and args.pca_dim < raw_embeddings.shape[1]:
            embeddings, pca_model = apply_pca(raw_embeddings, n_components=args.pca_dim)
            import pickle
            with open(output_dir / "pca_model.pkl", "wb") as f:
                pickle.dump(pca_model, f)
        else:
            embeddings = raw_embeddings
            args.pca_dim = raw_embeddings.shape[1]
            pca_model = None

        logger.info("Mapping embeddings back to full corpus...")
        full_embeddings = embeddings[row_to_unique]
        emb_cols = [f"text_emb_{i}" for i in range(args.pca_dim)]
        emb_df = pd.DataFrame(full_embeddings, columns=emb_cols)
        emb_df["entity_id"] = df["entity_id"].values
        emb_df["crawled_date_day"] = df["crawled_date_day"].values
        emb_df = emb_df[["entity_id", "crawled_date_day"] + emb_cols]
        out_path = output_dir / "text_embeddings.parquet"
        emb_df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(f"Saved {len(emb_df):,} rows × {len(emb_cols)} dims to {out_path}")

        meta = {
            "model": args.model,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "max_chars": args.max_chars,
            "pca_dim": args.pca_dim,
            "raw_dim": raw_embeddings.shape[1],
            "n_unique_texts": len(unique_texts),
            "n_total_rows": len(df),
            "n_entities": df["entity_id"].nunique(),
            "text_columns_used": ALL_TEXT_COLS,
            "n_shards": args.n_shards,
            "pca_explained_variance": float(pca_model.explained_variance_ratio_.sum())
            if pca_model is not None else 1.0,
        }
        with open(output_dir / "embedding_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Metadata saved to {output_dir / 'embedding_metadata.json'}")
        logger.info("Done!")
        return

    # 4. Handle sharding for parallel processing
    if args.n_shards > 1:
        shard_size = (len(unique_texts) + args.n_shards - 1) // args.n_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, len(unique_texts))
        unique_texts_shard = unique_texts[start:end]
        logger.info(f"Shard {args.shard_id}/{args.n_shards}: texts [{start}:{end}] ({len(unique_texts_shard):,})")
        shard_path = output_dir / f"embeddings_shard_{args.shard_id}.npy"
        if args.skip_existing_shard and shard_path.exists():
            logger.info(f"Shard {args.shard_id} already exists at {shard_path}; skipping")
            return
    else:
        unique_texts_shard = unique_texts
        start = 0

    # 5. Encode unique texts
    raw_embeddings = encode_texts(
        unique_texts_shard,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    # If sharded, save shard and exit
    if args.n_shards > 1:
        np.save(shard_path, raw_embeddings)
        with open(output_dir / f"embeddings_shard_{args.shard_id}.json", "w") as f:
            json.dump({
                "model": args.model,
                "shard_id": args.shard_id,
                "n_shards": args.n_shards,
                "start": int(start),
                "end": int(end),
                "n_rows": int(raw_embeddings.shape[0]),
                "raw_dim": int(raw_embeddings.shape[1]),
            }, f, indent=2)
        logger.info(f"Saved shard {args.shard_id} to {shard_path}")
        return

    # 6. Apply PCA (only for single-shard mode)
    if args.pca_dim > 0 and args.pca_dim < raw_embeddings.shape[1]:
        embeddings, pca_model = apply_pca(raw_embeddings, n_components=args.pca_dim)
        # Save PCA model for reproducibility
        import pickle
        with open(output_dir / "pca_model.pkl", "wb") as f:
            pickle.dump(pca_model, f)
    else:
        embeddings = raw_embeddings
        args.pca_dim = raw_embeddings.shape[1]

    # 7. Map back to full corpus
    logger.info("Mapping embeddings back to full corpus...")
    full_embeddings = embeddings[row_to_unique]

    # 8. Build output DataFrame
    emb_cols = [f"text_emb_{i}" for i in range(args.pca_dim)]
    emb_df = pd.DataFrame(full_embeddings, columns=emb_cols)
    emb_df["entity_id"] = df["entity_id"].values
    emb_df["crawled_date_day"] = df["crawled_date_day"].values

    # Reorder: keys first
    emb_df = emb_df[["entity_id", "crawled_date_day"] + emb_cols]

    # 9. Save as parquet
    out_path = output_dir / "text_embeddings.parquet"
    emb_df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"Saved {len(emb_df):,} rows × {len(emb_cols)} dims to {out_path}")

    # Save metadata
    meta = {
        "model": args.model,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_chars": args.max_chars,
        "pca_dim": args.pca_dim,
        "raw_dim": raw_embeddings.shape[1],
        "n_unique_texts": len(unique_texts),
        "n_total_rows": len(df),
        "n_entities": df["entity_id"].nunique(),
        "text_columns_used": ALL_TEXT_COLS,
        "pca_explained_variance": float(pca_model.explained_variance_ratio_.sum())
        if args.pca_dim > 0 and args.pca_dim < raw_embeddings.shape[1]
        else 1.0,
    }
    with open(output_dir / "embedding_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved to {output_dir / 'embedding_metadata.json'}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
