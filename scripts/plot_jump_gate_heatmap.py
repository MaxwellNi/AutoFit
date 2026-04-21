#!/usr/bin/env python3
"""Jump Gate Activation Heatmap — ED-SSM Feature Importance Visualizer.

Runs a trained (or freshly trained) ED-SSM on panel data with
``return_states=True`` to extract:
  - ``trajectory`` (B, L, D·N): full SSM state evolution
  - ``jump_gates`` (B, L): per-timestep jump activation norms

Produces publication-quality figures:
  1. Jump gate activation heatmap (entities × time)
  2. State trajectory PCA (temporal dynamics visualization)
  3. Jump gate vs. target correlation bar chart

Usage:
    python scripts/plot_jump_gate_heatmap.py \\
        --target is_funded \\
        --ablation core_only \\
        --horizon 7 \\
        --output-dir runs/edssm_figures \\
        --max-entities 200
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_LOG = logging.getLogger("jump_gate_viz")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Jump Gate Activation Visualizer")
    parser.add_argument("--target", default="is_funded")
    parser.add_argument("--ablation", default="core_only")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--output-dir", default="runs/edssm_figures")
    parser.add_argument("--max-entities", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--n-epochs", type=int, default=15)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from sklearn.decomposition import PCA

    from scripts.run_v740_alpha_minibenchmark import _build_case_frame, _make_temporal_config
    from scripts.run_v740_alpha_smoke_slice import _prepare_features
    from src.narrative.block3.models.single_model_mainline.event_driven_ssm import EventDrivenSSM
    from src.narrative.block3.models.single_model_mainline.sequential_adapter import (
        EntitySequenceDataset,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _TASK_MAP = {
        "is_funded": "task1_outcome",
        "funding_raised_usd": "task2_outcome",
        "investors_count": "task3_outcome",
    }

    # ── Load data ────────────────────────────────────────────────
    _LOG.info("Loading data ...")
    case = {
        "task": _TASK_MAP[args.target],
        "ablation": args.ablation,
        "target": args.target,
        "horizon": args.horizon,
        "max_entities": args.max_entities,
        "max_rows": args.max_entities * 50,
        "name": "viz",
    }
    temporal_config = _make_temporal_config()
    train, val, test = _build_case_frame(case, temporal_config)
    X_test, y_test = _prepare_features(test, args.target)

    if len(X_test) < 20:
        _LOG.error("Insufficient test data")
        return

    d_features = X_test.shape[1]
    _LOG.info(f"Test set: {len(X_test)} rows, {d_features} features")

    # ── Build entity sequences ───────────────────────────────────
    entity_ids = test["entity_id"].reindex(X_test.index).to_numpy() if "entity_id" in test.columns else None
    dates = test["crawled_date_day"].reindex(X_test.index).to_numpy() if "crawled_date_day" in test.columns else None

    ds = EntitySequenceDataset(
        X_test.to_numpy(dtype=np.float32),
        y_test.to_numpy(dtype=np.float32),
        entity_ids, dates,
        window_size=args.window_size,
    )

    # ── Train lightweight ED-SSM ─────────────────────────────────
    _LOG.info("Training lightweight ED-SSM ...")
    model = EventDrivenSSM(
        d_cont=d_features, d_event=0,
        d_model=args.d_model, d_state=args.d_state,
        d_output=args.d_model, n_layers=2,
    ).to(device)

    # Quick training with auxiliary head
    import torch.nn.functional as F
    aux_head = torch.nn.Linear(args.d_model, 1).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(aux_head.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=True, num_workers=0,
    )
    is_binary = args.target == "is_funded"

    model.train()
    for epoch in range(args.n_epochs):
        for batch in loader:
            x_seq, lengths, targets = batch
            x_seq = x_seq.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)

            out = model(x_seq, lengths=lengths)
            logits = aux_head(out).squeeze(-1)
            if is_binary:
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            else:
                loss = F.mse_loss(logits, targets)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    _LOG.info("Training done")

    # ── Forward pass with return_states ───────────────────────────
    model.eval()
    all_traj, all_jump, all_y = [], [], []
    eval_loader = torch.utils.data.DataLoader(
        ds, batch_size=512, shuffle=False, num_workers=0,
    )

    with torch.no_grad():
        for batch in eval_loader:
            x_seq, lengths, targets = batch
            x_seq = x_seq.to(device)
            lengths = lengths.to(device)

            result = model(x_seq, lengths=lengths, return_states=True)
            all_traj.append(result["trajectory"].cpu().numpy())
            all_jump.append(result["jump_gates"].cpu().numpy())
            all_y.append(targets.numpy())

    trajectory = np.concatenate(all_traj, axis=0)   # (N, L, D·N)
    jump_gates = np.concatenate(all_jump, axis=0)    # (N, L)
    y_arr = np.concatenate(all_y, axis=0)            # (N,)

    _LOG.info(f"trajectory: {trajectory.shape}, jump_gates: {jump_gates.shape}")

    # ── Figure 1: Jump Gate Activation Heatmap ───────────────────
    _LOG.info("Plotting jump gate heatmap ...")
    n_show = min(100, len(jump_gates))

    # Sort by total jump activation (most active on top)
    total_activation = jump_gates.sum(axis=1)
    sort_idx = np.argsort(-total_activation)[:n_show]

    fig, ax = plt.subplots(figsize=(12, 6))
    jg_show = jump_gates[sort_idx]

    # Use log scale if there's variation
    vmax = max(jg_show.max(), 1e-6)
    if vmax > 0:
        im = ax.imshow(
            jg_show + 1e-8, aspect="auto", cmap="hot",
            norm=LogNorm(vmin=1e-8, vmax=vmax),
        )
    else:
        im = ax.imshow(jg_show, aspect="auto", cmap="hot")

    ax.set_xlabel("Timestep (within lookback window)", fontsize=12)
    ax.set_ylabel("Entity (sorted by total activation)", fontsize=12)
    ax.set_title(
        f"ED-SSM Jump Gate Activations\n"
        f"target={args.target}, ablation={args.ablation}, h={args.horizon}",
        fontsize=14,
    )
    plt.colorbar(im, ax=ax, label="Jump gate ‖Δh‖")
    plt.tight_layout()

    fpath = output_dir / f"jump_gate_heatmap_{args.target}_{args.ablation}_h{args.horizon}.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _LOG.info(f"  Saved: {fpath}")

    # ── Figure 2: State Trajectory PCA ───────────────────────────
    _LOG.info("Plotting state trajectory PCA ...")
    # Take last-timestep state for PCA
    last_states = trajectory[:, -1, :]  # (N, D·N)
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(last_states)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        states_2d[:, 0], states_2d[:, 1],
        c=y_arr, cmap="coolwarm", alpha=0.5, s=10,
        vmin=np.percentile(y_arr, 5), vmax=np.percentile(y_arr, 95),
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
    ax.set_title(
        f"ED-SSM Final State Embedding (PCA)\n"
        f"colored by {args.target}",
        fontsize=14,
    )
    plt.colorbar(scatter, ax=ax, label=args.target)
    plt.tight_layout()

    fpath = output_dir / f"state_trajectory_pca_{args.target}_{args.ablation}_h{args.horizon}.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _LOG.info(f"  Saved: {fpath}")

    # ── Figure 3: Jump Activation vs. Target Correlation ─────────
    _LOG.info("Plotting jump activation correlation ...")
    total_jump = jump_gates.sum(axis=1)  # (N,)
    mean_jump_per_step = jump_gates.mean(axis=0)  # (L,)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Total jump vs target scatter
    ax = axes[0]
    ax.scatter(total_jump, y_arr, alpha=0.3, s=8, c="steelblue")
    ax.set_xlabel("Total Jump Activation ‖Δh‖ₜₒₜ", fontsize=12)
    ax.set_ylabel(args.target, fontsize=12)
    corr = np.corrcoef(total_jump, y_arr)[0, 1] if total_jump.std() > 0 else 0
    ax.set_title(f"Jump Activation vs {args.target}\nρ = {corr:.3f}", fontsize=13)

    # Panel B: Mean jump activation per timestep
    ax = axes[1]
    timesteps = np.arange(len(mean_jump_per_step))
    ax.bar(timesteps, mean_jump_per_step, color="coral", alpha=0.8)
    ax.set_xlabel("Timestep (within lookback window)", fontsize=12)
    ax.set_ylabel("Mean Jump Activation", fontsize=12)
    ax.set_title("Temporal Profile of Jump Gate Firing", fontsize=13)

    plt.tight_layout()
    fpath = output_dir / f"jump_correlation_{args.target}_{args.ablation}_h{args.horizon}.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _LOG.info(f"  Saved: {fpath}")

    _LOG.info(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
