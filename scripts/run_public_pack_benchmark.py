#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrative.public_pack.registry import (
    expand_public_pack_cells,
    filter_public_pack_cells,
    load_public_pack_registry,
)
from narrative.public_pack.runtime import (
    FIRST_WAVE_MODEL_PRESETS,
    prepare_public_pack_data,
    resolve_public_pack_models,
    run_public_pack_models,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run public-pack family cells through the unified model registry."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--pack", default="")
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--prediction-length", type=int, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--preset", default="first_wave_entrants")
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument("--max-train-samples-per-entity", type=int, default=512)
    parser.add_argument("--max-covariates", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--list-presets", action="store_true")
    return parser.parse_args()


def _cell_slug(family: str, variant: str, context_length: int, prediction_length: int) -> str:
    base = f"{family}__{variant}__ctx{context_length}__pred{prediction_length}"
    return base.replace("/", "_")


def main() -> int:
    args = _parse_args()

    if args.list_presets:
        print(json.dumps(FIRST_WAVE_MODEL_PRESETS, indent=2))
        return 0

    registry = load_public_pack_registry(args.config)
    cells = expand_public_pack_cells(
        registry,
        pack=args.pack,
        requested_families=args.family,
    )
    cells = filter_public_pack_cells(
        cells,
        requested_variants=args.variant,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )
    if not cells:
        raise ValueError("no public-pack cells matched the requested filters")

    all_results = []
    prediction_frames = []
    cell_summaries = []

    for cell in cells:
        prepared = prepare_public_pack_data(
            cell,
            dataset_path=args.dataset_path,
            target=args.target,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            max_entities=args.max_entities,
            max_train_samples_per_entity=args.max_train_samples_per_entity,
            max_covariates=args.max_covariates,
        )
        model_names = resolve_public_pack_models(
            args.model,
            preset=args.preset,
            is_binary_target=prepared.is_binary_target,
        )
        results, predictions = run_public_pack_models(
            prepared,
            model_names,
            n_bootstrap=args.n_bootstrap,
        )
        all_results.extend(results)
        if not predictions.empty:
            prediction_frames.append(predictions)
        cell_summaries.append(
            {
                "pack": cell.pack,
                "family": cell.family,
                "variant": cell.variant,
                "dataset_path": str(prepared.dataset_path),
                "context_length": prepared.context_length,
                "prediction_length": prepared.prediction_length,
                "n_entities": prepared.metadata.get("n_entities", 0),
                "n_train_samples": prepared.metadata.get("n_train_samples", 0),
                "n_test_samples": prepared.metadata.get("n_test_samples", 0),
                "models": model_names,
            }
        )

        if args.output_dir is not None:
            cell_output_dir = args.output_dir / _cell_slug(
                cell.family,
                cell.variant,
                prepared.context_length,
                prepared.prediction_length,
            )
            cell_output_dir.mkdir(parents=True, exist_ok=True)
            (cell_output_dir / "metrics.json").write_text(
                json.dumps(results, indent=2),
                encoding="utf-8",
            )
            if not predictions.empty:
                predictions.to_parquet(cell_output_dir / "predictions.parquet", index=False)

    payload = {
        "selection": {
            "pack": args.pack or None,
            "families": args.family,
            "variants": args.variant,
            "preset": args.preset,
            "models": args.model,
        },
        "cells": cell_summaries,
        "results": all_results,
    }

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "MANIFEST.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        if prediction_frames:
            pd.concat(prediction_frames, ignore_index=True).to_parquet(
                args.output_dir / "predictions.parquet",
                index=False,
            )

    summary = {
        "n_cells": len(cell_summaries),
        "n_results": len(all_results),
        "n_errors": sum(1 for result in all_results if "error" in result),
        "families": sorted({result["family"] for result in all_results}) if all_results else [],
        "models": sorted({result["model_name"] for result in all_results}) if all_results else [],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())