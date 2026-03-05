#!/usr/bin/env python3
"""
Block 3 Hyperparameter Grid Search Pipeline.

For each model category, defines a search space and generates SLURM jobs
that try all HP combos via the --model-kwargs-json mechanism.

Strategy:
  1) Coarse grid on core_only + task1_outcome (cheapest signal)
  2) Pick top-3 configs per model by validation RMSE
  3) Fine-grained search around winners on full ablation × task matrix

Usage:
    # Phase 1: Generate coarse grid SLURM jobs
    python scripts/hp_grid_search.py --phase coarse --submit

    # Phase 2: After coarse jobs complete, aggregate and find top configs
    python scripts/hp_grid_search.py --phase analyze --coarse-dir <dir>

    # Phase 3: Generate fine-grained jobs around top configs
    python scripts/hp_grid_search.py --phase fine --top-configs <file> --submit
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# HP Search Spaces by Category
# ============================================================================

# --- TSLib SOTA models ---
TSLIB_HP_SPACE: Dict[str, Dict[str, List[Any]]] = {
    # Shared across all TSLib models
    "__shared__": {
        "learning_rate": [5e-4, 1e-3, 2e-3],
        "batch_size": [32, 64, 128],
        "max_epochs": [50],
    },
    # Model-specific overrides
    "TimeFilter": {
        "d_model": [64, 128, 256],
        "d_ff": [128, 256, 512],
        "n_heads": [4, 8],
        "e_layers": [2, 3],
        "dropout": [0.05, 0.1, 0.2],
    },
    "WPMixer": {
        "d_model": [64, 128, 256],
        "d_ff": [128, 256, 512],
        "n_heads": [4, 8],
        "e_layers": [2, 3],
        "dropout": [0.05, 0.1, 0.2],
    },
    "MultiPatchFormer": {
        "d_model": [64, 128, 256],
        "d_ff": [128, 256, 512],
        "n_heads": [4, 8],
        "e_layers": [2, 3],
        "dropout": [0.05, 0.1, 0.2],
    },
    "MSGNet": {
        "d_model": [64, 128, 256],
        "d_ff": [128, 256, 512],
        "e_layers": [2, 3],
        "dropout": [0.05, 0.1, 0.2],
    },
    "PAttn": {
        "d_model": [64, 128, 256],
        "d_ff": [128, 256, 512],
        "n_heads": [4, 8],
        "e_layers": [2, 3],
        "dropout": [0.05, 0.1, 0.2],
    },
    "MambaSimple": {
        "d_model": [64, 128, 256],
        "d_ff": [128, 256, 512],
        "e_layers": [2, 3, 4],
        "dropout": [0.05, 0.1, 0.2],
    },
    "Koopa": {
        "d_model": [64, 128, 256],
        "d_ff": [128, 256, 512],
        "e_layers": [2, 3],
        "dropout": [0.05, 0.1, 0.2],
    },
    "FreTS": {
        "d_model": [64, 128, 256],
        "d_ff": [128, 256, 512],
        "e_layers": [2, 3],
        "dropout": [0.05, 0.1, 0.2],
    },
    "Crossformer": {
        "d_model": [64, 128, 256],
        "d_ff": [128, 256, 512],
        "n_heads": [4, 8],
        "e_layers": [2, 3],
        "dropout": [0.05, 0.1, 0.2],
    },
    "MICN": {
        "d_model": [64, 128, 256],
        "d_ff": [128, 256, 512],
        "e_layers": [2, 3],
        "dropout": [0.05, 0.1, 0.2],
    },
    "SegRNN": {
        "d_model": [64, 128, 256],
        "d_ff": [128, 256, 512],
        "e_layers": [2, 3],
        "dropout": [0.05, 0.1, 0.2],
    },
    "NonstationaryTransformer": {
        "d_model": [64, 128, 256],
        "d_ff": [128, 256, 512],
        "n_heads": [4, 8],
        "e_layers": [2, 3],
        "dropout": [0.05, 0.1, 0.2],
    },
    "FiLM": {
        "d_model": [64, 128, 256],
        "d_ff": [128, 256, 512],
        "e_layers": [2, 3],
        "dropout": [0.05, 0.1, 0.2],
    },
    "SCINet": {
        "d_model": [64, 128, 256],
        "d_ff": [128, 256, 512],
        "e_layers": [2, 3],
        "dropout": [0.05, 0.1, 0.2],
    },
}

# --- NeuralForecast deep_classical models ---
DEEP_HP_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "__shared__": {
        "max_steps": [500, 1000, 2000],
        "learning_rate": [5e-4, 1e-3, 5e-3],
        "batch_size": [32, 64],
    },
    "NBEATS": {
        "input_size": [30, 60, 90],
        "hidden_size": [128, 256, 512],
        "n_blocks": [1, 3, 5],
    },
    "NHITS": {
        "input_size": [30, 60, 90],
        "hidden_size": [128, 256, 512],
        "n_blocks": [1, 3, 5],
    },
    "TFT": {
        "input_size": [30, 60, 90],
        "hidden_size": [64, 128, 256],
        "n_head": [2, 4, 8],
        "dropout": [0.1, 0.2, 0.3],
    },
    "DeepAR": {
        "input_size": [30, 60, 90],
        "hidden_size": [64, 128, 256],
        "n_layers": [1, 2, 3],
    },
    "GRU": {
        "input_size": [30, 60, 90],
        "hidden_size": [64, 128, 256],
        "n_layers": [1, 2, 3],
    },
    "LSTM": {
        "input_size": [30, 60, 90],
        "hidden_size": [64, 128, 256],
        "n_layers": [1, 2, 3],
    },
    "TCN": {
        "input_size": [30, 60, 90],
        "hidden_size": [64, 128, 256],
        "kernel_size": [2, 3, 5],
    },
    "MLP": {
        "input_size": [30, 60, 90],
        "hidden_size": [128, 256, 512],
        "num_layers": [2, 3, 4],
    },
    "DilatedRNN": {
        "input_size": [30, 60, 90],
        "hidden_size": [64, 128, 256],
        "n_layers": [1, 2, 3],
    },
}

# --- Transformer SOTA (NeuralForecast) ---
TRANSFORMER_HP_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "__shared__": {
        "max_steps": [500, 1000, 2000],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "batch_size": [32, 64],
    },
    "PatchTST": {
        "input_size": [60, 90, 120],
        "hidden_size": [64, 128, 256],
        "n_heads": [4, 8],
        "e_layers": [2, 3],
        "patch_len": [8, 16, 24],
    },
    "iTransformer": {
        "input_size": [60, 90, 120],
        "hidden_size": [64, 128, 256],
        "n_heads": [4, 8],
        "e_layers": [2, 3],
    },
    "TimesNet": {
        "input_size": [60, 90, 120],
        "hidden_size": [32, 64, 128],
        "n_heads": [4, 8],
        "e_layers": [2, 3],
        "top_k": [3, 5],
    },
    "TSMixer": {
        "input_size": [60, 90, 120],
        "hidden_size": [64, 128, 256],
        "n_block": [2, 4, 6],
    },
}

# --- ML Tabular ---
ML_HP_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "LightGBM": {
        "n_estimators": [100, 500, 1000, 2000],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8, -1],
        "num_leaves": [31, 63, 127],
        "min_child_samples": [10, 20, 50],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0],
    },
    "XGBoost": {
        "n_estimators": [100, 500, 1000, 2000],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8, 10],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0],
    },
    "CatBoost": {
        "iterations": [500, 1000, 2000],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 10],
        "bagging_temperature": [0, 1, 5],
    },
    "RandomForest": {
        "n_estimators": [100, 500, 1000],
        "max_depth": [10, 20, 50, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    },
    "HistGradientBoosting": {
        "max_iter": [100, 500, 1000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, None],
        "min_samples_leaf": [5, 10, 20, 50],
        "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    },
}


# ============================================================================
# Grid Generation (Latin Hypercube Sampling for large spaces)
# ============================================================================

def _merge_shared(
    model_name: str,
    space: Dict[str, Dict[str, List[Any]]],
) -> Dict[str, List[Any]]:
    """Merge __shared__ with model-specific params."""
    shared = space.get("__shared__", {})
    specific = space.get(model_name, {})
    merged = {**shared, **specific}
    return merged


def _generate_grid(
    params: Dict[str, List[Any]],
    max_combos: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate HP combos. If full grid > max_combos, use Latin Hypercube
    sampling to pick representative combos.
    """
    keys = sorted(params.keys())
    values = [params[k] for k in keys]
    full_grid = list(itertools.product(*values))

    if len(full_grid) <= max_combos:
        return [dict(zip(keys, combo)) for combo in full_grid]

    # Stratified random sampling
    import random
    rng = random.Random(seed)
    sampled = rng.sample(full_grid, max_combos)
    return [dict(zip(keys, combo)) for combo in sampled]


def generate_hp_configs(
    category: str,
    model_name: str,
    max_combos: int = 50,
) -> List[Dict[str, Any]]:
    """Generate HP configs for a specific model."""
    space_map = {
        "tslib_sota": TSLIB_HP_SPACE,
        "deep_classical": DEEP_HP_SPACE,
        "transformer_sota": TRANSFORMER_HP_SPACE,
        "ml_tabular": ML_HP_SPACE,
    }
    space = space_map.get(category)
    if not space:
        print(f"No HP space defined for category: {category}")
        return []
    if model_name not in space and model_name != "__shared__":
        print(f"No model-specific space for {model_name}, using shared only")

    params = _merge_shared(model_name, space)
    if not params:
        print(f"  Empty search space for {model_name}")
        return []

    combos = _generate_grid(params, max_combos)
    return combos


# ============================================================================
# SLURM Job Generation
# ============================================================================

SLURM_TEMPLATE = r"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=1-12:00:00
#SBATCH --mem=384G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --signal=USR1@120

set -e

# ── Preemption handler ──
handle_preempt() {{
    echo "PREEMPT: SIGUSR1 received at $(date -Iseconds)"
    wait "$HARNESS_PID" 2>/dev/null || true
    echo "PREEMPT: Requeue count: ${{SLURM_RESTART_COUNT:-0}}"
    exit 0
}}
trap handle_preempt USR1

export CONDA_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider"
export PATH="${{CONDA_PREFIX}}/bin:${{PATH}}"
export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="/mnt/aiongpfs/projects/eint/vendor/Time-Series-Library:${{PYTHONPATH:-}}"
cd /mnt/aiongpfs/projects/eint/repo_root

INSIDER_PY="${{CONDA_PREFIX}}/bin/python3"
[[ -x "${{INSIDER_PY}}" ]] || {{ echo "FATAL: insider python missing"; exit 2; }}

echo "============================================================"
echo "HP Grid Search | Job ${{SLURM_JOB_ID}} | Config #{config_idx}"
echo "$(date -Iseconds) | $(hostname)"
echo "Model: {model_name} | Category: {category}"
echo "Task: {task} | Ablation: {ablation}"
echo "HP Config: {hp_json_oneline}"
echo "============================================================"

python3 - <<'PY'
import sys, torch
assert "insider" in sys.executable, "not insider python"
assert sys.version_info >= (3, 11), "python >= 3.11 required"
assert torch.cuda.is_available(), "GPU required"
PY

${{INSIDER_PY}} scripts/assert_block3_execution_contract.py --entrypoint "slurm:${{SLURM_JOB_NAME}}"

"${{INSIDER_PY}}" scripts/run_block3_benchmark_shard.py \
    --task {task} \
    --category {category} \
    --ablation {ablation} \
    --preset full \
    --output-dir {output_dir} \
    --seed 42 \
    --no-verify-first \
    --models {model_name} \
    --model-kwargs-json '{hp_json}' &

HARNESS_PID=$!
wait "$HARNESS_PID"

echo "Done: $(date -Iseconds)"
"""


def generate_coarse_jobs(
    category: str,
    models: List[str],
    max_combos_per_model: int = 30,
    script_dir: Optional[Path] = None,
    log_dir: Optional[str] = None,
    submit: bool = False,
) -> List[Path]:
    """Generate coarse grid search SLURM jobs.

    Phase 1: Only core_only + task1_outcome for fast signal.
    """
    if script_dir is None:
        script_dir = Path(f".slurm_scripts/hp_grid_{category}")
    if log_dir is None:
        log_dir = f"/work/projects/eint/logs/hp_grid_{category}"

    script_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    task = "task1_outcome"
    ablation = "core_only"
    scripts: List[Path] = []

    for model_name in models:
        combos = generate_hp_configs(category, model_name, max_combos_per_model)
        if not combos:
            continue

        print(f"\n{model_name}: {len(combos)} HP configs")

        for idx, hp_config in enumerate(combos):
            job_name = f"hp_{model_name[:10].lower()}_c{idx:02d}"
            hp_json = json.dumps({model_name: hp_config})
            hp_json_oneline = hp_json.replace('"', '\\"')[:120]
            output_dir = (
                f"runs/benchmarks/block3_20260203_225620_phase7/"
                f"hp_grid/{category}/{model_name}/config_{idx:02d}"
            )

            script_content = SLURM_TEMPLATE.format(
                job_name=job_name,
                log_dir=log_dir,
                config_idx=idx,
                model_name=model_name,
                category=category,
                task=task,
                ablation=ablation,
                hp_json_oneline=hp_json_oneline,
                hp_json=hp_json,
                output_dir=output_dir,
            )

            script_path = script_dir / f"{job_name}.sh"
            script_path.write_text(script_content, encoding="utf-8")
            script_path.chmod(0o755)
            scripts.append(script_path)

    print(f"\nTotal: {len(scripts)} SLURM scripts in {script_dir}/")

    if submit:
        print(f"\nSubmitting {len(scripts)} jobs...")
        for s in scripts:
            result = subprocess.run(
                ["sbatch", str(s)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"  {s.name} → Job {job_id}")
            else:
                print(f"  FAILED {s.name}: {result.stderr.strip()}")

    return scripts


def analyze_coarse_results(results_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Analyze coarse grid results and return top configs per model.

    Returns: {model_name: [(config_idx, val_rmse), ...]} sorted by RMSE.
    """
    import pandas as pd

    model_results: Dict[str, List[Tuple[int, float]]] = {}

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        config_scores = []

        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            idx = int(config_dir.name.split("_")[-1])

            # Look for metrics CSV
            metrics_files = list(config_dir.rglob("*metrics*.csv"))
            if not metrics_files:
                continue

            try:
                df = pd.read_csv(metrics_files[0])
                if "rmse" in df.columns:
                    avg_rmse = df["rmse"].mean()
                    config_scores.append((idx, avg_rmse))
            except Exception as e:
                print(f"  Warning: {config_dir}: {e}")

        if config_scores:
            config_scores.sort(key=lambda x: x[1])
            model_results[model_name] = config_scores
            print(f"{model_name}: top-3 configs: {config_scores[:3]}")

    return model_results


def save_top_configs(
    model_results: Dict[str, List[Tuple[int, float]]],
    category: str,
    hp_space: Dict[str, Dict[str, List[Any]]],
    output_file: Path,
    top_k: int = 3,
) -> None:
    """Save top-K HP configs per model for fine-grained search."""
    top_configs: Dict[str, List[Dict[str, Any]]] = {}

    for model_name, scores in model_results.items():
        combos = generate_hp_configs(category, model_name)
        best_indices = [idx for idx, _ in scores[:top_k]]
        top_configs[model_name] = [combos[i] for i in best_indices if i < len(combos)]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(top_configs, f, indent=2, default=str)
    print(f"\nSaved top-{top_k} configs to {output_file}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Block 3 HP Grid Search Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase", choices=["coarse", "analyze", "fine"],
        required=True, help="Search phase",
    )
    parser.add_argument(
        "--category", default="tslib_sota",
        help="Model category (default: tslib_sota)",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model names (default: all in category)",
    )
    parser.add_argument(
        "--max-combos", type=int, default=30,
        help="Max HP combos per model in coarse phase (default: 30)",
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="Actually submit SLURM jobs",
    )
    parser.add_argument(
        "--coarse-dir", type=Path, default=None,
        help="Coarse results directory for analyze phase",
    )
    parser.add_argument(
        "--top-configs", type=Path, default=None,
        help="Top configs JSON for fine phase",
    )

    args = parser.parse_args()

    # Resolve models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        import importlib, sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.narrative.block3.models.registry import MODEL_CATEGORIES
        models = MODEL_CATEGORIES.get(args.category, [])
        if not models:
            print(f"No models found for category: {args.category}")
            sys.exit(1)

    print(f"Category: {args.category}")
    print(f"Models: {models}")
    print(f"Phase: {args.phase}")

    if args.phase == "coarse":
        generate_coarse_jobs(
            category=args.category,
            models=models,
            max_combos_per_model=args.max_combos,
            submit=args.submit,
        )

    elif args.phase == "analyze":
        if not args.coarse_dir:
            args.coarse_dir = Path(
                f"runs/benchmarks/block3_20260203_225620_phase7/"
                f"hp_grid/{args.category}"
            )
        results = analyze_coarse_results(args.coarse_dir)
        output_file = Path(f"configs/hp_top_configs_{args.category}.json")
        space_map = {
            "tslib_sota": TSLIB_HP_SPACE,
            "deep_classical": DEEP_HP_SPACE,
            "transformer_sota": TRANSFORMER_HP_SPACE,
            "ml_tabular": ML_HP_SPACE,
        }
        save_top_configs(results, args.category, space_map[args.category], output_file)

    elif args.phase == "fine":
        if not args.top_configs:
            args.top_configs = Path(f"configs/hp_top_configs_{args.category}.json")
        if not args.top_configs.exists():
            print(f"Top configs not found: {args.top_configs}")
            print("Run --phase analyze first.")
            sys.exit(1)
        # Fine-grained: expand around top configs for all tasks × ablations
        with open(args.top_configs, "r") as f:
            top = json.load(f)
        print(f"Fine-grained search: {len(top)} models, {sum(len(v) for v in top.values())} configs")
        # TODO: Implement fine-grained perturbation around top configs
        print("Fine-grained phase: will submit full matrix for top configs")


if __name__ == "__main__":
    main()
