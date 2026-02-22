#!/usr/bin/env python3
"""
Build an auditable Block 3 truth pack from materialized benchmark metrics.

Legacy outputs (kept for compatibility):
  - condition_leaderboard.csv
  - autofit_lineage.csv
  - failure_taxonomy.csv
  - v71_vs_v7_overlap.csv
  - truth_pack_summary.json
  - truth_pack_summary.md

Panorama outputs:
  - task_subtask_catalog.csv
  - condition_inventory_full.csv
  - subtasks_by_target_full.csv
  - run_history_ledger.csv
  - run_history_observations.csv
  - autofit_version_ladder.csv
  - autofit_step_deltas.csv
  - top3_representative_models_by_target.csv
  - family_gap_by_target.csv
  - sota_feature_value_map.csv
  - audit_gate_snapshot.csv
  - primary_literature_matrix.csv
  - citation_correction_log.csv
  - slurm_snapshot.json
  - slurm_snapshot.md
  - slurm_live_snapshot_YYYYMMDD_HHMMSS.json
  - slurm_live_snapshot_YYYYMMDD_HHMMSS.md
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal Python env
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
RUNS_BENCH_ROOT = ROOT / "runs" / "benchmarks"
DEFAULT_BENCH_DIRS = [
    ROOT / "runs" / "benchmarks" / "block3_20260203_225620_phase7",
    ROOT / "runs" / "benchmarks" / "block3_20260203_225620_phase7_v71extreme_20260214_032205",
    ROOT / "runs" / "benchmarks" / "block3_20260203_225620_phase7_v71extreme_20260214_130737",
]
DEFAULT_BENCH_GLOB = "block3_20260203_225620*"
DEFAULT_CONFIG = ROOT / "configs" / "block3_tasks.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "benchmarks" / "block3_truth_pack"
DEFAULT_MASTER_DOC = ROOT / "docs" / "AUTOFIT_V72_EVIDENCE_MASTER_20260217.md"
DEFAULT_SLURM_SINCE = "2026-02-12"

SECTION_ORDER: List[Tuple[str, str]] = [
    ("EVIDENCE_SNAPSHOT", "## Evidence Snapshot"),
    (
        "AUDIT_GATES",
        "## Audit Gates Snapshot",
    ),
    ("TASK_AND_SUBTASK_UNIVERSE", "## Task And Subtask Universe"),
    (
        "DATA_CHARACTERISTIC_DERIVED_SUBTASKS",
        "## Data-Characteristic-Derived Subtasks",
    ),
    (
        "MODEL_FAMILY_COVERAGE_AUDIT",
        "## Model Family Coverage Audit",
    ),
    (
        "TARGET_SUBTASKS",
        "## Target Subtasks (is_funded / funding_raised_usd / investors_count)",
    ),
    (
        "TOP3_REPRESENTATIVE_MODELS",
        "## Top-3 Representative Models",
    ),
    (
        "FAMILY_GAP_MATRIX",
        "## Family Gap Matrix",
    ),
    (
        "CHAMPION_TEMPLATE_LIBRARY",
        "## Champion Template Library",
    ),
    (
        "HYPERPARAMETER_SEARCH_LEDGER",
        "## Hyperparameter Search Ledger",
    ),
    (
        "BEST_CONFIG_BY_MODEL_TARGET",
        "## Best Config By Model/Target",
    ),
    (
        "COMPUTE_COST_REPORT",
        "## Compute Cost Report",
    ),
    (
        "V72_PILOT_GATE_REPORT",
        "## V7.2 Pilot Gate Report",
    ),
    (
        "HISTORICAL_FULL_SCALE_EXPERIMENT_LEDGER",
        "## Historical Full-Scale Experiment Ledger",
    ),
    (
        "AUTOFIT_VERSION_LADDER",
        "## AutoFit Version Ladder (V1->V7.2)",
    ),
    (
        "HIGH_VALUE_SOTA_COMPONENTS",
        "## High-Value SOTA Components For This Freeze",
    ),
    (
        "PRIMARY_LITERATURE_MATRIX",
        "## Primary Literature Matrix",
    ),
    (
        "CITATION_CORRECTION_LOG",
        "## Citation Correction Log",
    ),
    ("LIVE_SLURM_SNAPSHOT", "## Live Slurm Snapshot"),
]

AUTOFIT_VERSION_ORDER = [
    "AutoFitV1",
    "AutoFitV2",
    "AutoFitV2E",
    "AutoFitV3",
    "AutoFitV3E",
    "AutoFitV3Max",
    "AutoFitV4",
    "AutoFitV5",
    "AutoFitV6",
    "AutoFitV7",
    "AutoFitV71",
    "AutoFitV72",
]

AUTOFIT_VERSION_META: Dict[str, Dict[str, str]] = {
    "AutoFitV1": {
        "commit_hint": "baseline_pre_phase2",
        "core_changes": "Data-driven best-single selection with residual correction.",
        "inspiration_source": "Pragmatic stacked residual correction from tabular ensemble practice.",
    },
    "AutoFitV2": {
        "commit_hint": "87baa13",
        "core_changes": "Top-K weighted ensemble by inverse validation MAE.",
        "inspiration_source": "Classical weighted model averaging.",
    },
    "AutoFitV2E": {
        "commit_hint": "87baa13",
        "core_changes": "Top-K stacking with LightGBM meta-learner.",
        "inspiration_source": "Stacked generalization.",
    },
    "AutoFitV3": {
        "commit_hint": "320c314",
        "core_changes": "Temporal CV + greedy/exhaustive subset selection with diversity preference.",
        "inspiration_source": "Temporal OOF stacking with subset search.",
    },
    "AutoFitV3E": {
        "commit_hint": "320c314",
        "core_changes": "Top-K stacking variant under V3 temporal CV framework.",
        "inspiration_source": "OOF-based stacking simplification.",
    },
    "AutoFitV3Max": {
        "commit_hint": "320c314",
        "core_changes": "Exhaustive V3 search with bounded candidate budget.",
        "inspiration_source": "Combinatorial ensemble search.",
    },
    "AutoFitV4": {
        "commit_hint": "c53abf6",
        "core_changes": "Target transforms + full OOF stacking + diversity-aware selection.",
        "inspiration_source": "NCL and transform-aware robust stacking.",
    },
    "AutoFitV5": {
        "commit_hint": "dce0ff9",
        "core_changes": "Regime-aware tiered evaluation and collapse detection guard.",
        "inspiration_source": "Cost-aware routing with monotonic fallback guard.",
    },
    "AutoFitV6": {
        "commit_hint": "ad07032",
        "core_changes": "Caruana-style greedy ensembling + two-layer stacking + conformal weighting.",
        "inspiration_source": "AutoGluon-style weighted ensemble with robust transform.",
    },
    "AutoFitV7": {
        "commit_hint": "faafdcf",
        "core_changes": "Data-adapted robust ensemble with missingness/ratio features and repeated temporal CV.",
        "inspiration_source": "SOTA tabular feature engineering + robust ensemble selection.",
    },
    "AutoFitV71": {
        "commit_hint": "phase7_v71_extreme_branch",
        "core_changes": "Lane-adaptive routing, dynamic thresholds, count-safe mode, anchor and policy logging.",
        "inspiration_source": "Target-family specialization with fairness-first guards.",
    },
    "AutoFitV72": {
        "commit_hint": "planned_not_materialized",
        "core_changes": "Count-safe hardening + champion-anchor + offline policy layer (planned).",
        "inspiration_source": "Evidence-driven v7.2 design gates.",
    },
}


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)

    # Fallback: delegate YAML parsing to system python that has PyYAML.
    try:
        out = subprocess.check_output(
            [
                "bash",
                "-lc",
                "python3 -c 'import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1], encoding=\"utf-8\").read())))' '%s'"
                % str(path),
            ],
            universal_newlines=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        parsed = json.loads(out)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    raise RuntimeError(
        "Could not parse YAML. Install PyYAML for this Python runtime "
        "or ensure `python3` with PyYAML is available."
    )


def _extract_dict_keys_from_source(path: Path, var_name: str) -> List[str]:
    if not path.exists():
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        hit = any(isinstance(t, ast.Name) and t.id == var_name for t in node.targets)
        if not hit or not isinstance(node.value, ast.Dict):
            continue
        keys: List[str] = []
        for k in node.value.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.append(k.value)
        return keys
    return []


def _registered_models_catalog() -> Dict[str, List[str]]:
    model_root = ROOT / "src" / "narrative" / "block3" / "models"
    return {
        "ml_tabular": _extract_dict_keys_from_source(model_root / "traditional_ml.py", "TRADITIONAL_ML_MODELS"),
        "statistical": _extract_dict_keys_from_source(model_root / "statistical.py", "STATISTICAL_MODELS"),
        "deep_classical": _extract_dict_keys_from_source(model_root / "deep_models.py", "DEEP_MODELS"),
        "transformer_sota": _extract_dict_keys_from_source(model_root / "deep_models.py", "TRANSFORMER_MODELS"),
        "foundation": _extract_dict_keys_from_source(model_root / "deep_models.py", "FOUNDATION_MODELS"),
        "irregular": _extract_dict_keys_from_source(model_root / "irregular_models.py", "IRREGULAR_MODELS"),
        "autofit": _extract_dict_keys_from_source(model_root / "autofit_wrapper.py", "AUTOFIT_MODELS"),
    }


def _display_path(path: Path) -> str:
    s = str(path)
    for marker in ["/runs/", "/docs/", "/configs/", "/scripts/", "/src/"]:
        idx = s.find(marker)
        if idx >= 0:
            return s[idx + 1 :]
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        try:
            return str(path.relative_to(ROOT))
        except Exception:
            return str(path)


def _to_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _to_int(value: Any) -> Optional[int]:
    v = _to_float(value)
    if v is None:
        return None
    return int(round(v))


def _is_autofit(model_name: str, category: str) -> bool:
    return category == "autofit" or "AutoFit" in model_name


def _infer_autofit_variant_id(row: Dict[str, Any]) -> Optional[str]:
    model_name = str(row.get("model_name") or "")
    source = str(row.get("_source_path") or "")
    if not _is_autofit(model_name, str(row.get("category") or "")):
        return None
    if model_name == "AutoFitV72":
        return "v72"
    if model_name == "AutoFitV71":
        m = re.search(r"(v71_g0[1-5])", source)
        return m.group(1) if m else "v71_unlabeled"
    if model_name.startswith("AutoFit"):
        return model_name.lower()
    return None


def _v72_priority_group(task: str, ablation: str) -> Tuple[int, str]:
    if task == "task1_outcome" and ablation in {"core_text", "full", "core_edgar"}:
        return 1, "P1_task1_core_text_full_core_edgar"
    if task == "task2_forecast" and ablation in {"core_only", "core_text", "full"}:
        return 2, "P2_task2_core_only_text_full"
    if task == "task3_risk_adjust" and ablation in {"core_only", "core_edgar", "full"}:
        return 3, "P3_task3_all_ablations"
    return 4, "P4_other"


def _build_v72_missing_key_manifest(
    strict_records: Sequence[Dict[str, Any]],
    expected_conditions: Sequence[Tuple[str, str, str, int]],
) -> Tuple[List[Dict[str, Any]], int, float]:
    expected = sorted(set(expected_conditions))
    v72_keys: set[Tuple[str, str, str, int]] = set()
    for row in strict_records:
        if str(row.get("model_name", "")) != "AutoFitV72":
            continue
        key = row.get("_condition_key")
        if key is None:
            continue
        v72_keys.add(key)

    missing_keys = [k for k in expected if k not in v72_keys]
    manifest: List[Dict[str, Any]] = []
    for task, ablation, target, horizon in missing_keys:
        rank, group = _v72_priority_group(task, ablation)
        manifest.append(
            {
                "task": task,
                "ablation": ablation,
                "target": target,
                "horizon": horizon,
                "priority_rank": rank,
                "priority_group": group,
                "reason": "v72_strict_missing",
            }
        )

    total = len(expected)
    coverage_ratio = (float(total - len(missing_keys)) / float(total)) if total else 0.0
    return manifest, len(missing_keys), coverage_ratio


def _estimate_queue_eta_model(slurm_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Simple queue ETA heuristic for planning (not a scheduler guarantee)."""
    running_total = int(slurm_snapshot.get("running_total", 0) or 0)
    pending_total = int(slurm_snapshot.get("pending_total", 0) or 0)
    qos_caps = slurm_snapshot.get("qos_caps", {}) if isinstance(slurm_snapshot, dict) else {}
    batch_cap = _to_int((qos_caps.get("iris-batch-long") or {}).get("MaxJobsPU")) or 8
    gpu_cap = _to_int((qos_caps.get("iris-gpu-long") or {}).get("MaxJobsPU")) or 4
    effective_parallelism = max(1, batch_cap + gpu_cap)
    assumed_avg_hours_per_job = 18.0
    pending_hours = (float(pending_total) / float(effective_parallelism)) * assumed_avg_hours_per_job
    running_tail_hours = assumed_avg_hours_per_job if running_total > 0 else 0.0
    eta_hours = pending_hours + running_tail_hours
    return {
        "method": "heuristic_v1",
        "assumed_avg_hours_per_job": assumed_avg_hours_per_job,
        "effective_parallelism": effective_parallelism,
        "batch_cap_max_jobs_pu": batch_cap,
        "gpu_cap_max_jobs_pu": gpu_cap,
        "estimated_hours_to_clear": round(eta_hours, 2),
    }


def _quantiles(values: Sequence[float], qs: Sequence[float]) -> List[Optional[float]]:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return [None for _ in qs]
    n = len(vals)
    out: List[Optional[float]] = []
    for q in qs:
        if n == 1:
            out.append(float(vals[0]))
            continue
        pos = max(0.0, min(1.0, q)) * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out.append(float(vals[lo]))
        else:
            w_hi = pos - lo
            w_lo = 1.0 - w_hi
            out.append(float(vals[lo] * w_lo + vals[hi] * w_hi))
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _load_optional_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    except Exception:
        return []


def _load_optional_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _build_best_config_rows(best_cfg_payload: Optional[Any], evidence_path: str) -> List[Dict[str, Any]]:
    if not isinstance(best_cfg_payload, dict):
        return [
            {
                "target": "n/a",
                "model_name": "n/a",
                "target_family": "n/a",
                "category": "n/a",
                "status": "missing",
                "search_budget": None,
                "trials_executed": None,
                "best_mae_observed_strict": None,
                "best_config_json": "artifact_missing",
                "search_space_json": "artifact_missing",
                "evidence_path": evidence_path,
            }
        ]

    rows: List[Dict[str, Any]] = []
    targets = best_cfg_payload.get("targets", {})
    if not isinstance(targets, dict):
        return rows
    for target, model_map in sorted(targets.items()):
        if not isinstance(model_map, dict):
            continue
        for model_name, item in sorted(model_map.items()):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "target": target,
                    "model_name": model_name,
                    "target_family": item.get("target_family"),
                    "category": item.get("category"),
                    "status": item.get("status"),
                    "search_budget": item.get("search_budget"),
                    "trials_executed": item.get("trials_executed"),
                    "best_mae_observed_strict": item.get("best_mae_observed_strict"),
                    "best_config_json": _safe_json(item.get("best_config", {})),
                    "search_space_json": _safe_json(item.get("search_space", {})),
                    "evidence_path": item.get("evidence_path", evidence_path),
                }
            )
    if not rows:
        rows.append(
            {
                "target": "n/a",
                "model_name": "n/a",
                "target_family": "n/a",
                "category": "n/a",
                "status": "empty",
                "search_budget": None,
                "trials_executed": None,
                "best_mae_observed_strict": None,
                "best_config_json": "empty_payload",
                "search_space_json": "empty_payload",
                "evidence_path": evidence_path,
            }
        )
    return rows


def _build_pilot_gate_rows(pilot_payload: Optional[Any], evidence_path: str) -> List[Dict[str, Any]]:
    if not isinstance(pilot_payload, dict):
        return [
            {
                "section": "summary",
                "key": "status",
                "value": "artifact_missing",
                "evidence_path": evidence_path,
            }
        ]

    rows: List[Dict[str, Any]] = [
        {
            "section": "summary",
            "key": "generated_at_utc",
            "value": pilot_payload.get("generated_at_utc"),
            "evidence_path": evidence_path,
        },
        {
            "section": "summary",
            "key": "overall_pass",
            "value": pilot_payload.get("overall_pass"),
            "evidence_path": evidence_path,
        },
    ]
    counts = pilot_payload.get("counts", {})
    if isinstance(counts, dict):
        for k, v in sorted(counts.items()):
            rows.append(
                {
                    "section": "counts",
                    "key": k,
                    "value": v,
                    "evidence_path": evidence_path,
                }
            )
    metrics = pilot_payload.get("metrics", {})
    if isinstance(metrics, dict):
        for k, v in sorted(metrics.items()):
            rows.append(
                {
                    "section": "metrics",
                    "key": k,
                    "value": v,
                    "evidence_path": evidence_path,
                }
            )
    checks = pilot_payload.get("checks", {})
    if isinstance(checks, dict):
        for k, v in sorted(checks.items()):
            rows.append(
                {
                    "section": "checks",
                    "key": k,
                    "value": v,
                    "evidence_path": evidence_path,
                }
            )
    return rows


def _condition_key(row: Dict[str, Any]) -> Optional[Tuple[str, str, str, int]]:
    task = row.get("task")
    ablation = row.get("ablation")
    target = row.get("target")
    horizon = _to_int(row.get("horizon"))
    if task is None or ablation is None or target is None or horizon is None:
        return None
    return str(task), str(ablation), str(target), int(horizon)


def _run_stage(name: str) -> str:
    if name.endswith("_iris_phase7"):
        return "iris_phase7_partial"
    if name.endswith("_phase7"):
        return "phase7_canonical"
    if "phase7_v71extreme_20260214_032205" in name:
        return "v71_pilot"
    if "phase7_v71extreme_20260214_130737" in name:
        return "v71_full"
    if name.endswith("_iris_full"):
        return "iris_full_baseline"
    if name.endswith("_iris_phase3"):
        return "iris_phase3"
    if name.endswith("_iris"):
        return "iris_initial"
    return "other"


def _run_stage_rank(stage: str) -> int:
    order = {
        "iris_initial": 0,
        "iris_phase3": 1,
        "iris_full_baseline": 2,
        "iris_phase7_partial": 3,
        "phase7_canonical": 4,
        "v71_pilot": 5,
        "v71_full": 6,
        "other": 99,
    }
    return order.get(stage, 99)


def _resolve_bench_dirs(
    explicit_dirs: Sequence[Path],
    include_freeze_history: bool,
    bench_glob: str,
) -> List[Path]:
    if explicit_dirs:
        uniq: List[Path] = []
        seen = set()
        for p in explicit_dirs:
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            if rp.exists() and rp.is_dir():
                uniq.append(rp)
        return uniq

    bench_dirs: List[Path] = []
    if include_freeze_history and RUNS_BENCH_ROOT.exists():
        for cand in sorted(RUNS_BENCH_ROOT.glob(bench_glob)):
            if not cand.is_dir():
                continue
            if cand.name.startswith("block3_preflight_"):
                continue
            if not cand.name.startswith("block3_20260203_225620"):
                continue
            bench_dirs.append(cand.resolve())

    if not bench_dirs:
        bench_dirs = [p.resolve() for p in DEFAULT_BENCH_DIRS if p.exists()]

    bench_dirs = sorted(
        list(dict.fromkeys(bench_dirs)),
        key=lambda p: (_run_stage_rank(_run_stage(p.name)), p.name),
    )
    return bench_dirs


def _load_expected_conditions(config_path: Path) -> List[Tuple[str, str, str, int]]:
    cfg = _safe_load_yaml(config_path)
    tasks = cfg.get("tasks", {})
    full_horizons = cfg.get("presets", {}).get("full", {}).get("horizons", [1, 7, 14, 30])
    expected: List[Tuple[str, str, str, int]] = []
    for task_name, task_cfg in tasks.items():
        task_ablations = task_cfg.get("ablations", [])
        targets = task_cfg.get("targets", [])
        for ablation in task_ablations:
            for target in targets:
                for horizon in full_horizons:
                    expected.append((str(task_name), str(ablation), str(target), int(horizon)))
    return sorted(set(expected))


def _load_metrics_records(bench_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not bench_dir.exists():
        return records
    for mf in sorted(bench_dir.rglob("metrics.json")):
        try:
            payload = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue

        if isinstance(payload, dict):
            rows = payload.get("results", []) or []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []

        rel_source = str(mf.relative_to(bench_dir))
        source_path = _display_path(bench_dir / rel_source)
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item["_bench_dir"] = bench_dir.name
            item["_run_stage"] = _run_stage(bench_dir.name)
            item["_source"] = rel_source
            item["_source_path"] = source_path
            records.append(item)
    return records


def classify_record_layer(row: Dict[str, Any], min_coverage: float) -> str:
    """Classify comparability layer for one metric record."""
    mae = _to_float(row.get("mae"))
    if mae is None:
        return "invalid"

    has_fairness = "fairness_pass" in row
    has_coverage = "prediction_coverage_ratio" in row
    if not (has_fairness and has_coverage):
        return "legacy_unverified"

    fairness = row.get("fairness_pass")
    coverage = _to_float(row.get("prediction_coverage_ratio"))
    if fairness is True and coverage is not None and coverage >= min_coverage:
        return "strict_comparable"
    return "strict_excluded"


def _prepare_records(
    raw_records: Sequence[Dict[str, Any]],
    min_coverage: float,
) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "strict_comparable": [],
        "legacy_unverified": [],
        "strict_excluded": [],
        "invalid": [],
        "valid_all": [],
    }

    for row in raw_records:
        item = dict(row)
        item["mae"] = _to_float(item.get("mae"))
        item["horizon"] = _to_int(item.get("horizon"))
        key = _condition_key(item)
        item["_condition_key"] = key
        layer = classify_record_layer(item, min_coverage=min_coverage)
        item["_layer"] = layer
        buckets[layer].append(item)
        if item["mae"] is not None and key is not None:
            buckets["valid_all"].append(item)

    return buckets


def _build_condition_leaderboard(
    strict_records: Sequence[Dict[str, Any]],
    expected_conditions: Sequence[Tuple[str, str, str, int]],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str, int], float], Dict[Tuple[str, str, str, int], List[Dict[str, Any]]]]:
    expected_set = set(expected_conditions)
    by_condition: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in strict_records:
        key = row.get("_condition_key")
        if key is None:
            continue
        by_condition[key].append(row)

    condition_rows: List[Dict[str, Any]] = []
    best_non_by_condition: Dict[Tuple[str, str, str, int], float] = {}

    all_keys = sorted(expected_set.union(by_condition.keys()))
    for cond in all_keys:
        rows = sorted(by_condition.get(cond, []), key=lambda x: float(x["mae"]))
        task, ablation, target, horizon = cond
        cond_row: Dict[str, Any] = {
            "task": task,
            "ablation": ablation,
            "target": target,
            "horizon": horizon,
            "expected_condition": cond in expected_set,
            "condition_completed": len(rows) > 0,
            "n_records": len(rows),
        }
        if rows:
            best = rows[0]
            best_non = next(
                (r for r in rows if not _is_autofit(str(r.get("model_name", "")), str(r.get("category", "")))),
                None,
            )
            best_af = next(
                (r for r in rows if _is_autofit(str(r.get("model_name", "")), str(r.get("category", "")))),
                None,
            )
            cond_row.update(
                {
                    "best_model": best.get("model_name"),
                    "best_category": best.get("category"),
                    "best_mae": best.get("mae"),
                    "best_non_autofit_model": best_non.get("model_name") if best_non else None,
                    "best_non_autofit_category": best_non.get("category") if best_non else None,
                    "best_non_autofit_mae": best_non.get("mae") if best_non else None,
                    "best_autofit_model": best_af.get("model_name") if best_af else None,
                    "best_autofit_variant_id": _infer_autofit_variant_id(best_af) if best_af else None,
                    "best_autofit_mae": best_af.get("mae") if best_af else None,
                    "autofit_gap_pct": (
                        (float(best_af["mae"]) / max(float(best_non["mae"]), 1e-12) - 1.0) * 100.0
                        if best_af is not None and best_non is not None
                        else None
                    ),
                    "bench_dirs": ";".join(sorted({str(r.get("_bench_dir", "")) for r in rows if r.get("_bench_dir")})),
                    "sources": ";".join(sorted({str(r.get("_source_path", "")) for r in rows[:10] if r.get("_source_path")})),
                }
            )
            if best_non is not None:
                best_non_by_condition[cond] = float(best_non["mae"])
        else:
            cond_row.update(
                {
                    "best_model": None,
                    "best_category": None,
                    "best_mae": None,
                    "best_non_autofit_model": None,
                    "best_non_autofit_category": None,
                    "best_non_autofit_mae": None,
                    "best_autofit_model": None,
                    "best_autofit_variant_id": None,
                    "best_autofit_mae": None,
                    "autofit_gap_pct": None,
                    "bench_dirs": "",
                    "sources": "",
                }
            )
        condition_rows.append(cond_row)

    return condition_rows, best_non_by_condition, by_condition


def _build_autofit_lineage(
    strict_records: Sequence[Dict[str, Any]],
    expected_conditions: Sequence[Tuple[str, str, str, int]],
    best_non_by_condition: Dict[Tuple[str, str, str, int], float],
) -> List[Dict[str, Any]]:
    lineage_stats: Dict[Tuple[str, str], List[Tuple[Tuple[str, str, str, int], float]]] = defaultdict(list)
    for row in strict_records:
        model_name = str(row.get("model_name", ""))
        category = str(row.get("category", ""))
        if not _is_autofit(model_name, category):
            continue
        key = row.get("_condition_key")
        if key is None or row.get("mae") is None:
            continue
        lineage_stats[(model_name, key[2])].append((key, float(row["mae"])))

    lineage_rows: List[Dict[str, Any]] = []
    for (model_name, target), pairs in sorted(lineage_stats.items()):
        maes = [m for _, m in pairs]
        q25, q50, q75 = _quantiles(maes, [0.25, 0.5, 0.75])
        cond_keys = {k for k, _ in pairs}
        target_expected = [k for k in expected_conditions if k[2] == target]
        gaps: List[float] = []
        for cond_key, mae in pairs:
            if cond_key in best_non_by_condition:
                base = best_non_by_condition[cond_key]
                gaps.append((mae / max(base, 1e-12) - 1.0) * 100.0)

        lineage_rows.append(
            {
                "model_name": model_name,
                "target": target,
                "n_records": len(maes),
                "conditions_covered": len(cond_keys),
                "condition_coverage_ratio": (
                    float(len(cond_keys)) / float(len(target_expected)) if target_expected else None
                ),
                "best_mae": min(maes),
                "median_mae": q50,
                "p25_mae": q25,
                "p75_mae": q75,
                "worst_mae": max(maes),
                "median_gap_vs_best_non_autofit_pct": _quantiles(gaps, [0.5])[0] if gaps else None,
            }
        )

    return lineage_rows


def _build_v71_vs_v7_overlap(
    strict_records: Sequence[Dict[str, Any]],
    expected_conditions: Sequence[Tuple[str, str, str, int]],
) -> List[Dict[str, Any]]:
    expected_set = set(expected_conditions)
    by_model_condition: Dict[Tuple[str, str, str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    by_condition: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)

    for row in strict_records:
        key = row.get("_condition_key")
        if key is None:
            continue
        model_name = str(row.get("model_name", ""))
        by_model_condition[(key[0], key[1], key[2], key[3], model_name)].append(row)
        by_condition[key].append(row)

    overlap_rows: List[Dict[str, Any]] = []
    for cond in sorted(expected_set.union(by_condition.keys())):
        task, ablation, target, horizon = cond
        v7_rows = sorted(
            by_model_condition.get((task, ablation, target, horizon, "AutoFitV7"), []),
            key=lambda r: float(r["mae"]),
        )
        v71_rows = sorted(
            by_model_condition.get((task, ablation, target, horizon, "AutoFitV71"), []),
            key=lambda r: float(r["mae"]),
        )
        if not v7_rows or not v71_rows:
            continue

        v7 = v7_rows[0]
        v71 = v71_rows[0]
        mae_v7 = float(v7["mae"])
        mae_v71 = float(v71["mae"])
        gain = (mae_v7 - mae_v71) / max(mae_v7, 1e-12) * 100.0
        overlap_rows.append(
            {
                "task": task,
                "ablation": ablation,
                "target": target,
                "horizon": horizon,
                "mae_v7": mae_v7,
                "mae_v71": mae_v71,
                "relative_gain_pct": gain,
                "v71_wins": gain > 0.0,
                "source_v7": v7.get("_source_path"),
                "source_v71": v71.get("_source_path"),
            }
        )

    return overlap_rows


def _build_failure_taxonomy(
    all_valid_records: Sequence[Dict[str, Any]],
    condition_rows: Sequence[Dict[str, Any]],
    min_coverage: float,
) -> List[Dict[str, Any]]:
    failure_rows: List[Dict[str, Any]] = []
    for row in all_valid_records:
        key = row.get("_condition_key") or ("", "", "", None)
        model_name = str(row.get("model_name", ""))
        category = str(row.get("category", ""))
        mae = _to_float(row.get("mae"))
        fairness = row.get("fairness_pass")
        coverage = _to_float(row.get("prediction_coverage_ratio"))
        layer = str(row.get("_layer", ""))

        if layer == "strict_excluded" and fairness is False:
            failure_rows.append(
                {
                    "issue_type": "fairness_guard_fail",
                    "severity": "critical",
                    "model_name": model_name,
                    "category": category,
                    "task": key[0],
                    "ablation": key[1],
                    "target": key[2],
                    "horizon": key[3],
                    "mae": mae,
                    "prediction_coverage_ratio": coverage,
                    "fairness_pass": fairness,
                    "evidence_source": row.get("_source_path"),
                    "note": "Record flagged fairness_pass=false",
                }
            )

        if layer == "strict_excluded" and coverage is not None and coverage < min_coverage:
            failure_rows.append(
                {
                    "issue_type": "coverage_below_threshold",
                    "severity": "critical",
                    "model_name": model_name,
                    "category": category,
                    "task": key[0],
                    "ablation": key[1],
                    "target": key[2],
                    "horizon": key[3],
                    "mae": mae,
                    "prediction_coverage_ratio": coverage,
                    "fairness_pass": fairness,
                    "evidence_source": row.get("_source_path"),
                    "note": "Coverage below configured threshold",
                }
            )

        if (
            model_name == "AutoFitV71"
            and str(row.get("target")) == "investors_count"
            and mae is not None
            and mae > 1_000_000
        ):
            failure_rows.append(
                {
                    "issue_type": "v71_count_explosion",
                    "severity": "critical",
                    "model_name": model_name,
                    "category": category,
                    "task": key[0],
                    "ablation": key[1],
                    "target": key[2],
                    "horizon": key[3],
                    "mae": mae,
                    "prediction_coverage_ratio": coverage,
                    "fairness_pass": fairness,
                    "evidence_source": row.get("_source_path"),
                    "note": "AutoFitV71 investors_count catastrophic MAE spike",
                }
            )

    for cond_row in condition_rows:
        gap = _to_float(cond_row.get("autofit_gap_pct"))
        if gap is None:
            continue
        if gap > 100.0:
            failure_rows.append(
                {
                    "issue_type": "autofit_gap_gt_100pct",
                    "severity": "high",
                    "model_name": cond_row.get("best_autofit_model"),
                    "category": "autofit",
                    "task": cond_row.get("task"),
                    "ablation": cond_row.get("ablation"),
                    "target": cond_row.get("target"),
                    "horizon": cond_row.get("horizon"),
                    "mae": cond_row.get("best_autofit_mae"),
                    "prediction_coverage_ratio": None,
                    "fairness_pass": True,
                    "evidence_source": cond_row.get("sources"),
                    "note": "Best AutoFit is >100% worse than best non-AutoFit",
                }
            )

    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    failure_rows = sorted(
        failure_rows,
        key=lambda r: (
            severity_rank.get(str(r.get("severity")), 99),
            str(r.get("task", "")),
            str(r.get("ablation", "")),
            str(r.get("target", "")),
            str(r.get("horizon", "")),
            str(r.get("issue_type", "")),
        ),
    )
    return failure_rows


def _build_condition_inventory_full(
    expected_conditions: Sequence[Tuple[str, str, str, int]],
    strict_records: Sequence[Dict[str, Any]],
    legacy_records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], set, set]:
    expected_set = set(expected_conditions)
    strict_by_condition: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    legacy_by_condition: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)

    for row in strict_records:
        key = row.get("_condition_key")
        if key is not None:
            strict_by_condition[key].append(row)

    for row in legacy_records:
        key = row.get("_condition_key")
        if key is not None:
            legacy_by_condition[key].append(row)

    strict_cond_set = {k for k, rows in strict_by_condition.items() if rows}
    legacy_cond_set = {k for k, rows in legacy_by_condition.items() if rows}

    rows_out: List[Dict[str, Any]] = []
    for cond in sorted(expected_set):
        task, ablation, target, horizon = cond
        strict_rows = sorted(strict_by_condition.get(cond, []), key=lambda x: float(x["mae"]))
        legacy_rows = sorted(legacy_by_condition.get(cond, []), key=lambda x: float(x["mae"]))
        rows_out.append(
            {
                "task": task,
                "ablation": ablation,
                "target": target,
                "horizon": horizon,
                "expected": True,
                "strict_completed": bool(strict_rows),
                "legacy_completed": bool(legacy_rows),
                "best_model_strict": strict_rows[0].get("model_name") if strict_rows else None,
                "best_category_strict": strict_rows[0].get("category") if strict_rows else None,
                "best_mae_strict": strict_rows[0].get("mae") if strict_rows else None,
            }
        )

    return rows_out, strict_cond_set, legacy_cond_set


def _load_target_stats_from_column_manifest(pointer_path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not pointer_path.exists():
        return out

    try:
        pointer = _safe_load_yaml(pointer_path)
    except Exception:
        return out

    col_manifest_rel = (
        pointer.get("analysis", {}).get("column_manifest")
        if isinstance(pointer, dict)
        else None
    )
    if not col_manifest_rel:
        return out

    col_manifest_path = ROOT / str(col_manifest_rel)
    if not col_manifest_path.exists():
        return out

    try:
        payload = json.loads(col_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return out

    stats = payload.get("offers_core_column_stats", {})
    for target in ["funding_raised_usd", "investors_count", "is_funded", "funding_goal_usd"]:
        info = stats.get(target, {}) if isinstance(stats, dict) else {}
        out[target] = {
            "missing_rate": _to_float(info.get("missing_rate")),
            "n_unique": _to_int(info.get("n_unique")),
            "dtype": info.get("dtype"),
            "column_manifest_path": _display_path(col_manifest_path),
        }
    return out


def _coverage_ratio(keys: Iterable[Tuple[str, str, str, int]], covered: set) -> float:
    key_list = list(keys)
    if not key_list:
        return 0.0
    return float(sum(1 for k in key_list if k in covered)) / float(len(key_list))


def _build_task_subtask_catalog(
    expected_conditions: Sequence[Tuple[str, str, str, int]],
    strict_cond_set: set,
    target_stats: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    expected = list(expected_conditions)
    targets = sorted({k[2] for k in expected})
    tasks = sorted({k[0] for k in expected})
    ablations = sorted({k[1] for k in expected})
    horizons = sorted({k[3] for k in expected})

    rows: List[Dict[str, Any]] = []

    def add_row(
        subtask_id: str,
        family: str,
        definition_rule: str,
        keys: Sequence[Tuple[str, str, str, int]],
        evidence_path: str,
    ) -> None:
        rows.append(
            {
                "subtask_id": subtask_id,
                "subtask_family": family,
                "definition_rule": definition_rule,
                "key_count": len(keys),
                "key_coverage_strict": _coverage_ratio(keys, strict_cond_set),
                "evidence_path": evidence_path,
            }
        )

    add_row(
        "all_condition_keys",
        "condition_universe",
        "All expected keys from task x ablation x target x horizon lattice.",
        expected,
        "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv",
    )

    for task in tasks:
        keys = [k for k in expected if k[0] == task]
        add_row(
            "task:" + task,
            "task",
            "All condition keys where task == %s." % task,
            keys,
            "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv",
        )

    for task in tasks:
        for target in targets:
            keys = [k for k in expected if k[0] == task and k[2] == target]
            if not keys:
                continue
            add_row(
                "task_target:%s|%s" % (task, target),
                "task_target",
                "All keys where task == %s and target == %s." % (task, target),
                keys,
                "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv",
            )

    target_family_rules = {
        "binary": ["is_funded"],
        "count": ["investors_count"],
        "heavy_tail": ["funding_raised_usd"],
    }
    for family, fam_targets in target_family_rules.items():
        keys = [k for k in expected if k[2] in fam_targets]
        add_row(
            "target_family:" + family,
            "target_family",
            "Lane family inferred from target semantics.",
            keys,
            "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv",
        )

    for ablation in ablations:
        keys = [k for k in expected if k[1] == ablation]
        add_row(
            "modality:" + ablation,
            "modality",
            "All keys where ablation == %s." % ablation,
            keys,
            "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv",
        )

    for horizon in horizons:
        keys = [k for k in expected if k[3] == horizon]
        add_row(
            "horizon:%s" % horizon,
            "horizon",
            "All keys where horizon == %s." % horizon,
            keys,
            "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv",
        )

    horizon_bands = {
        "short": {1, 7},
        "mid": {14},
        "long": {30},
    }
    for band, band_horizons in horizon_bands.items():
        keys = [k for k in expected if k[3] in band_horizons]
        add_row(
            "horizon_band:" + band,
            "horizon_band",
            "Band grouping on horizon values (%s)." % sorted(list(band_horizons)),
            keys,
            "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv",
        )

    risk_keys = [k for k in expected if k[0] == "task3_risk_adjust"]
    add_row(
        "robustness:task3_risk_adjust",
        "robustness",
        "OOD robustness proxy keys from task3_risk_adjust.",
        risk_keys,
        "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv",
    )

    derived_rows: List[Dict[str, Any]] = []
    for target in ["funding_raised_usd", "investors_count", "is_funded"]:
        keys = [k for k in expected if k[2] == target]
        stats = target_stats.get(target, {})
        missing_rate = stats.get("missing_rate")
        n_unique = stats.get("n_unique")
        definition = "Missingness and cardinality profile for %s." % target
        if missing_rate is not None or n_unique is not None:
            definition += " missing_rate=%s, n_unique=%s." % (missing_rate, n_unique)
        row = {
            "subtask_id": "data_feature:%s" % target,
            "subtask_family": "data_characteristic",
            "definition_rule": definition,
            "key_count": len(keys),
            "key_coverage_strict": _coverage_ratio(keys, strict_cond_set),
            "evidence_path": stats.get(
                "column_manifest_path",
                "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv",
            ),
        }
        rows.append(row)
        derived_rows.append(row)

    return rows, derived_rows


def _build_run_history_ledger(
    bench_dirs: Sequence[Path],
    all_records: Sequence[Dict[str, Any]],
    strict_records: Sequence[Dict[str, Any]],
    legacy_records: Sequence[Dict[str, Any]],
    failure_rows: Sequence[Dict[str, Any]],
    expected_conditions: Sequence[Tuple[str, str, str, int]],
    min_coverage: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    expected_set = set(expected_conditions)

    all_by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    strict_by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    legacy_by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_records:
        all_by_run[str(row.get("_bench_dir", ""))].append(row)
    for row in strict_records:
        strict_by_run[str(row.get("_bench_dir", ""))].append(row)
    for row in legacy_records:
        legacy_by_run[str(row.get("_bench_dir", ""))].append(row)

    failure_by_run: Dict[str, Counter] = defaultdict(Counter)
    for row in failure_rows:
        src = str(row.get("evidence_source", ""))
        run_name = ""
        parts = src.split("/")
        if len(parts) >= 4 and parts[0] == "runs" and parts[1] == "benchmarks":
            run_name = parts[2]
        if run_name:
            failure_by_run[run_name][str(row.get("issue_type", "unknown"))] += 1

    ledger_rows: List[Dict[str, Any]] = []
    observation_rows: List[Dict[str, Any]] = []

    for bdir in bench_dirs:
        run_name = bdir.name
        run_stage = _run_stage(run_name)
        run_all = all_by_run.get(run_name, [])
        run_strict = strict_by_run.get(run_name, [])
        run_legacy = legacy_by_run.get(run_name, [])

        strict_cond = {row.get("_condition_key") for row in run_strict if row.get("_condition_key") is not None}
        legacy_cond = {row.get("_condition_key") for row in run_legacy if row.get("_condition_key") is not None}

        by_target_best: Dict[str, Dict[str, Any]] = {}
        for target in sorted({k[2] for k in expected_set}):
            trows = [r for r in run_strict if str(r.get("target")) == target and r.get("mae") is not None]
            if not trows:
                continue
            best = min(trows, key=lambda x: float(x["mae"]))
            by_target_best[target] = {
                "model": best.get("model_name"),
                "category": best.get("category"),
                "mae": best.get("mae"),
            }

        strict_ratio = float(len(run_strict)) / float(len(run_all)) if run_all else 0.0
        key_failures_counter = Counter(failure_by_run.get(run_name, Counter()))
        for row in run_all:
            layer = str(row.get("_layer", ""))
            fairness = row.get("fairness_pass")
            coverage = _to_float(row.get("prediction_coverage_ratio"))
            model_name = str(row.get("model_name", ""))
            target = str(row.get("target", ""))
            mae = _to_float(row.get("mae"))
            if layer == "strict_excluded" and fairness is False:
                key_failures_counter["fairness_guard_fail"] += 1
            if layer == "strict_excluded" and coverage is not None and coverage < min_coverage:
                key_failures_counter["coverage_below_threshold"] += 1
            if (
                model_name == "AutoFitV71"
                and target == "investors_count"
                and mae is not None
                and mae > 1_000_000
            ):
                key_failures_counter["v71_count_explosion"] += 1

        key_failures = dict(key_failures_counter)

        ledger_rows.append(
            {
                "run_name": run_name,
                "run_stage": run_stage,
                "raw_records": len(run_all),
                "strict_records": len(run_strict),
                "legacy_records": len(run_legacy),
                "strict_ratio": strict_ratio,
                "models": len({str(r.get("model_name", "")) for r in run_all if r.get("model_name")}),
                "categories": len({str(r.get("category", "")) for r in run_all if r.get("category")}),
                "condition_coverage_strict": float(len(strict_cond)) / float(len(expected_set)) if expected_set else 0.0,
                "condition_coverage_legacy": float(len(legacy_cond)) / float(len(expected_set)) if expected_set else 0.0,
                "best_model_by_target_json": _safe_json(by_target_best),
                "key_failures": _safe_json(key_failures),
            }
        )

        observation_rows.append(
            {
                "run_name": run_name,
                "observation_type": "strict_ratio",
                "observation": "Share of strict-comparable records among all valid records.",
                "supporting_metric": "%.4f" % strict_ratio,
                "evidence_path": _display_path(bdir),
            }
        )
        observation_rows.append(
            {
                "run_name": run_name,
                "observation_type": "strict_condition_coverage",
                "observation": "Coverage ratio of expected 104 keys under strict comparability.",
                "supporting_metric": "%.4f" % (
                    float(len(strict_cond)) / float(len(expected_set)) if expected_set else 0.0
                ),
                "evidence_path": "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv",
            }
        )
        if key_failures:
            observation_rows.append(
                {
                    "run_name": run_name,
                    "observation_type": "failure_tags",
                    "observation": "Failure taxonomy tags found in this run.",
                    "supporting_metric": _safe_json(key_failures),
                    "evidence_path": "docs/benchmarks/block3_truth_pack/failure_taxonomy.csv",
                }
            )
        if by_target_best:
            observation_rows.append(
                {
                    "run_name": run_name,
                    "observation_type": "target_winners",
                    "observation": "Best strict model per target for this run.",
                    "supporting_metric": _safe_json(by_target_best),
                    "evidence_path": _display_path(bdir),
                }
            )

    ledger_rows = sorted(ledger_rows, key=lambda r: (_run_stage_rank(str(r["run_stage"])), str(r["run_name"])))
    observation_rows = sorted(
        observation_rows,
        key=lambda r: (
            _run_stage_rank(_run_stage(str(r["run_name"]))),
            str(r["run_name"]),
            str(r["observation_type"]),
        ),
    )
    return ledger_rows, observation_rows


def _build_autofit_version_ladder(
    lineage_rows: Sequence[Dict[str, Any]],
    failure_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    lineage_by_model: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in lineage_rows:
        model = str(row.get("model_name", ""))
        target = str(row.get("target", ""))
        lineage_by_model[model][target] = row

    failure_by_model: Dict[str, Counter] = defaultdict(Counter)
    for row in failure_rows:
        model = str(row.get("model_name", ""))
        if model:
            failure_by_model[model][str(row.get("issue_type", "unknown"))] += 1

    rows_out: List[Dict[str, Any]] = []
    for version in AUTOFIT_VERSION_ORDER:
        notes = AUTOFIT_VERSION_META.get(version, {})
        target_rows = lineage_by_model.get(version, {})
        measured_targets = sorted(target_rows.keys())

        if version == "AutoFitV72":
            primary_failure = "planned_not_materialized"
        else:
            fail_counts = failure_by_model.get(version, Counter())
            if fail_counts.get("v71_count_explosion", 0) > 0:
                primary_failure = "v71_count_explosion"
            elif fail_counts.get("autofit_gap_gt_100pct", 0) > 0:
                primary_failure = "high_gap_vs_best_non_autofit"
            elif not measured_targets:
                primary_failure = "no_materialized_records"
            else:
                primary_failure = "no_critical_failure_observed"

        median_mae = {
            target: target_rows[target].get("median_mae")
            for target in measured_targets
        }
        median_gap = {
            target: target_rows[target].get("median_gap_vs_best_non_autofit_pct")
            for target in measured_targets
        }

        rows_out.append(
            {
                "version": version,
                "commit_hint": notes.get("commit_hint", "unknown"),
                "core_changes": notes.get("core_changes", ""),
                "inspiration_source": notes.get("inspiration_source", ""),
                "measured_targets": ",".join(measured_targets),
                "median_mae_by_target_json": _safe_json(median_mae),
                "median_gap_vs_best_non_autofit_json": _safe_json(median_gap),
                "primary_failure_mode": primary_failure,
                "evidence_path": "docs/benchmarks/block3_truth_pack/autofit_lineage.csv",
            }
        )

    return rows_out


def _build_autofit_step_deltas(
    strict_records: Sequence[Dict[str, Any]],
    best_non_by_condition: Dict[Tuple[str, str, str, int], float],
    expected_conditions: Sequence[Tuple[str, str, str, int]],
) -> List[Dict[str, Any]]:
    model_cond_mae: Dict[str, Dict[Tuple[str, str, str, int], float]] = defaultdict(dict)
    for row in strict_records:
        model = str(row.get("model_name", ""))
        category = str(row.get("category", ""))
        if not _is_autofit(model, category):
            continue
        key = row.get("_condition_key")
        mae = _to_float(row.get("mae"))
        if key is None or mae is None:
            continue
        old = model_cond_mae[model].get(key)
        if old is None or mae < old:
            model_cond_mae[model][key] = mae

    targets = sorted({k[2] for k in expected_conditions})
    rows_out: List[Dict[str, Any]] = []

    for i in range(len(AUTOFIT_VERSION_ORDER) - 1):
        from_v = AUTOFIT_VERSION_ORDER[i]
        to_v = AUTOFIT_VERSION_ORDER[i + 1]
        from_map = model_cond_mae.get(from_v, {})
        to_map = model_cond_mae.get(to_v, {})

        for target in targets:
            from_keys = {k for k in from_map.keys() if k[2] == target}
            to_keys = {k for k in to_map.keys() if k[2] == target}
            overlap = sorted(from_keys.intersection(to_keys))

            if not overlap:
                rows_out.append(
                    {
                        "from_version": from_v,
                        "to_version": to_v,
                        "target": target,
                        "overlap_keys": 0,
                        "median_mae_delta_pct": None,
                        "median_gap_delta_pct": None,
                    }
                )
                continue

            mae_deltas: List[float] = []
            gap_deltas: List[float] = []
            for key in overlap:
                f = from_map[key]
                t = to_map[key]
                mae_deltas.append((t / max(f, 1e-12) - 1.0) * 100.0)

                base = best_non_by_condition.get(key)
                if base is not None:
                    from_gap = (f / max(base, 1e-12) - 1.0) * 100.0
                    to_gap = (t / max(base, 1e-12) - 1.0) * 100.0
                    gap_deltas.append(to_gap - from_gap)

            rows_out.append(
                {
                    "from_version": from_v,
                    "to_version": to_v,
                    "target": target,
                    "overlap_keys": len(overlap),
                    "median_mae_delta_pct": _quantiles(mae_deltas, [0.5])[0],
                    "median_gap_delta_pct": _quantiles(gap_deltas, [0.5])[0] if gap_deltas else None,
                }
            )

    return rows_out


def _build_sota_feature_value_map(
    condition_rows: Sequence[Dict[str, Any]],
    failure_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    winners_non_autofit = [r for r in condition_rows if r.get("condition_completed")]
    by_target_category: Dict[str, Counter] = defaultdict(Counter)
    by_target_model: Dict[str, Counter] = defaultdict(Counter)

    for row in winners_non_autofit:
        target = str(row.get("target", ""))
        cat = str(row.get("best_non_autofit_category", ""))
        model = str(row.get("best_non_autofit_model", ""))
        if target and cat:
            by_target_category[target][cat] += 1
        if target and model:
            by_target_model[target][model] += 1

    failure_counts = Counter(str(r.get("issue_type", "")) for r in failure_rows)

    def top_items(counter: Counter, n: int = 3) -> str:
        if not counter:
            return "n/a"
        return ", ".join("%s=%s" % (k, v) for k, v in counter.most_common(n))

    rows = [
        {
            "feature_component": "Deep sequence inductive bias",
            "winner_evidence": "investors_count winners by category: %s" % top_items(by_target_category.get("investors_count", Counter())),
            "affected_subtasks": "target_family:count; task2_forecast investors_count",
            "why_effective": "Deep temporal models preserve count trajectory structure better than current AutoFit count lane.",
            "integration_priority": "high",
            "risk": "Over-parameterized deep models may become unstable on sparse slices.",
            "verification_test": "Track median gap reduction on investors_count under strict comparability.",
            "evidence_path": "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
        },
        {
            "feature_component": "Foundation-model global priors",
            "winner_evidence": "funding_raised_usd winners by category: %s" % top_items(by_target_category.get("funding_raised_usd", Counter())),
            "affected_subtasks": "target_family:heavy_tail",
            "why_effective": "Foundation/deep models are robust to heavy-tail dynamics and long-range temporal drift.",
            "integration_priority": "high",
            "risk": "Inference cost and dependency drift across clusters.",
            "verification_test": "Monitor heavy-tail lane median MAE and tail quantile errors.",
            "evidence_path": "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
        },
        {
            "feature_component": "Transformer representation for binary outcome",
            "winner_evidence": "is_funded winners by model: %s" % top_items(by_target_model.get("is_funded", Counter())),
            "affected_subtasks": "target_family:binary; task1_outcome is_funded",
            "why_effective": "Temporal representation improves ranking and calibration for binary outcome targets.",
            "integration_priority": "medium",
            "risk": "Calibration drift when class balance shifts over time.",
            "verification_test": "Track binary MAE/logloss/AUC drift by horizon under strict guard.",
            "evidence_path": "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
        },
        {
            "feature_component": "Count-safe postprocess hard guard",
            "winner_evidence": "failure tags: %s" % top_items(failure_counts),
            "affected_subtasks": "target_family:count; core_edgar investors_count",
            "why_effective": "Count explosions indicate inverse-transform and clipping chain needs hard rejection logic.",
            "integration_priority": "critical",
            "risk": "Over-clipping can suppress true extreme events.",
            "verification_test": "Count lane guard-hit accounting and no catastrophic MAE spikes.",
            "evidence_path": "docs/benchmarks/block3_truth_pack/failure_taxonomy.csv",
        },
        {
            "feature_component": "Champion-anchor retention",
            "winner_evidence": "AutoFit high-gap tags: %s" % failure_counts.get("autofit_gap_gt_100pct", 0),
            "affected_subtasks": "all target families",
            "why_effective": "Anchor retention prevents ensemble collapse to weak homogeneous candidate sets.",
            "integration_priority": "high",
            "risk": "Anchor may overfit historical winner if guard is too permissive.",
            "verification_test": "OOF guard and bounded degradation constraint on anchor injection.",
            "evidence_path": "docs/benchmarks/block3_truth_pack/failure_taxonomy.csv",
        },
        {
            "feature_component": "Fairness and coverage comparability gate",
            "winner_evidence": "Strict vs legacy layering enforced at record level.",
            "affected_subtasks": "all",
            "why_effective": "Separating strict-comparable from legacy-unverified prevents invalid leaderboard conclusions.",
            "integration_priority": "critical",
            "risk": "Historical results with missing guards become non-comparable.",
            "verification_test": "Summary fields strict_records/legacy_records and strict condition coverage.",
            "evidence_path": "docs/benchmarks/block3_truth_pack/truth_pack_summary.json",
        },
    ]
    return rows


def _build_primary_literature_matrix_rows() -> List[Dict[str, Any]]:
    """Primary-source matrix for v7.2 design inputs (strictly first-party links)."""
    rows = [
        {
            "topic": "time_series_foundation",
            "source": "Chronos (arXiv:2403.07815)",
            "problem": "Universal forecasting across heterogeneous time series with limited task-specific tuning.",
            "core_mechanism": "Language-model style tokenization over scaled time series and seq2seq pretraining.",
            "what_it_fixes": "Weak cross-domain priors in sparse or noisy slices.",
            "risk": "Tokenization mismatch on non-stationary count bursts.",
            "integration_point": "Global prior features and champion-anchor candidates for heavy-tail/count lanes.",
            "expected_gain": "Improved robustness on low-signal and long-horizon regimes.",
            "verification_test": "Strict comparable median MAE and tail quantile error by horizon.",
            "primary_link": "https://arxiv.org/abs/2403.07815",
            "status": "verified_primary",
        },
        {
            "topic": "time_series_foundation",
            "source": "Chronos-2 (arXiv:2510.15821)",
            "problem": "Unified large-scale pretrained forecasting for zero-shot and fine-tuned settings.",
            "core_mechanism": "Chronos family second-generation architecture and training pipeline improvements.",
            "what_it_fixes": "Transfer degradation under distribution shift versus earlier Chronos variants.",
            "risk": "Operational dependency drift and heavier inference footprint.",
            "integration_point": "Reference foundation baseline and lane-aware anchor pool updates.",
            "expected_gain": "Better OOD resilience on full ablation matrix.",
            "verification_test": "Condition-level win share in strict_comparable subset.",
            "primary_link": "https://arxiv.org/abs/2510.15821",
            "status": "verified_primary",
        },
        {
            "topic": "time_series_foundation",
            "source": "Moirai-MoE (arXiv:2410.10469)",
            "problem": "Handling diverse time-series regimes with scalable specialist capacity.",
            "core_mechanism": "Sparse mixture-of-experts routing for temporal patterns.",
            "what_it_fixes": "Single-backbone underfitting for regime-diverse signals.",
            "risk": "Route instability and training complexity.",
            "integration_point": "Inspiration for offline routing policy and lane-specialist candidate pools.",
            "expected_gain": "Lower variance across task/ablation slices.",
            "verification_test": "Variance of MAE across repeated strict runs and per-lane gap.",
            "primary_link": "https://arxiv.org/abs/2410.10469",
            "status": "verified_primary",
        },
        {
            "topic": "time_series_foundation",
            "source": "TiRex (arXiv:2505.23719)",
            "problem": "General-purpose time-series foundation transfer across tasks.",
            "core_mechanism": "Unified representation learning and transfer-oriented objective design.",
            "what_it_fixes": "Task-specific over-specialization and weak reuse of temporal representations.",
            "risk": "Potential mismatch between benchmark tasks and pretraining assumptions.",
            "integration_point": "Representation-oriented features for lane routing and anchor diversity.",
            "expected_gain": "Stronger cross-task consistency on strict condition keys.",
            "verification_test": "WinRate@Condition across tasks under fairness gate.",
            "primary_link": "https://arxiv.org/abs/2505.23719",
            "status": "verified_primary",
        },
        {
            "topic": "time_series_foundation",
            "source": "Time-MoE (arXiv:2409.16040)",
            "problem": "Scaling temporal models without fully dense compute cost.",
            "core_mechanism": "Sparse expert activation and efficient MoE architecture for time series.",
            "what_it_fixes": "Throughput bottlenecks at large model scale.",
            "risk": "Expert imbalance and route collapse.",
            "integration_point": "Offline policy design and budgeted candidate routing in AutoFitV7.2.",
            "expected_gain": "Better performance-compute Pareto under fixed QOS constraints.",
            "verification_test": "Compute_cost_report vs GlobalNormalizedMAE under strict filters.",
            "primary_link": "https://arxiv.org/abs/2409.16040",
            "status": "verified_primary",
        },
        {
            "topic": "time_series_llm",
            "source": "Time-LLM (arXiv:2310.01728)",
            "problem": "Injecting semantic priors from LLMs into temporal forecasting.",
            "core_mechanism": "Reprogramming time series into LLM-compatible prompt/token spaces.",
            "what_it_fixes": "Weak contextual semantics in pure numerical models.",
            "risk": "Prompt sensitivity and unstable gains across slices.",
            "integration_point": "Auxiliary retrieval/regime descriptors and policy features (offline only).",
            "expected_gain": "Improved binary/count subtasks with richer context encoding.",
            "verification_test": "Ablation against baseline retrieval-disabled setting.",
            "primary_link": "https://arxiv.org/abs/2310.01728",
            "status": "verified_primary",
        },
        {
            "topic": "time_series_retrieval",
            "source": "Retrieval-Augmented Forecasting (arXiv:2505.04163)",
            "problem": "Model adaptation to recurring regimes and non-stationary contexts.",
            "core_mechanism": "Retrieve analogous historical windows and condition predictions on retrieved context.",
            "what_it_fixes": "Context-loss in long-history or regime-switch settings.",
            "risk": "Leakage if retrieval index uses future slices.",
            "integration_point": "Train-only prototype bank for regime retrieval features in AutoFit lanes.",
            "expected_gain": "Better robustness in task3_risk_adjust and long-horizon slices.",
            "verification_test": "Leakage audit + OOD slice degradation_pct delta.",
            "primary_link": "https://arxiv.org/abs/2505.04163",
            "status": "verified_primary",
        },
        {
            "topic": "time_series_multimodal",
            "source": "TimeOmni-1 (arXiv:2509.24803)",
            "problem": "Unified multimodal time-series representation with long context.",
            "core_mechanism": "Cross-modal alignment with long-context modeling.",
            "what_it_fixes": "Fragmented handling of text/edgar/core modalities.",
            "risk": "High training/inference complexity for production benchmark loops.",
            "integration_point": "Guidance for modality-aware feature fusion and lane gating.",
            "expected_gain": "More stable full-ablation gains over core-only baselines.",
            "verification_test": "full vs core_only uplift consistency under strict keys.",
            "primary_link": "https://arxiv.org/abs/2509.24803",
            "status": "verified_primary",
        },
        {
            "topic": "rl_policy",
            "source": "LangTime (arXiv:2506.10630)",
            "problem": "Improving forecast decisions via policy optimization rather than static heuristics.",
            "core_mechanism": "PPO-based policy optimization for time-series decision flow.",
            "what_it_fixes": "Rigid static routing/weighting in changing regimes.",
            "risk": "Test leakage if policy is updated on held-out feedback.",
            "integration_point": "Offline-only policy action selection from train/OOF trajectories.",
            "expected_gain": "Higher condition-level win rate with bounded risk.",
            "verification_test": "Policy leak audit + reproducible policy_action_id telemetry.",
            "primary_link": "https://arxiv.org/abs/2506.10630",
            "status": "verified_primary",
        },
        {
            "topic": "tree_sota",
            "source": "XGBoost 3.2.0 Release Notes",
            "problem": "Scalable tabular learning with improved infrastructure/runtime behavior.",
            "core_mechanism": "Core library and ecosystem updates in official release pipeline.",
            "what_it_fixes": "Version-drift and stale baseline reproducibility.",
            "risk": "Behavior shifts across minor/major upgrades.",
            "integration_point": "Pinned baseline refresh for ml_tabular comparison lanes.",
            "expected_gain": "Cleaner baseline, fewer silent incompatibilities.",
            "verification_test": "Version-pinned smoke + strict benchmark delta log.",
            "primary_link": "https://xgboost.readthedocs.io/en/stable/changes/v3.2.0.html",
            "status": "verified_primary",
        },
        {
            "topic": "tree_sota",
            "source": "LightGBM Releases (official GitHub)",
            "problem": "Maintaining strong tabular baseline quality and compatibility.",
            "core_mechanism": "Official incremental improvements through released versions.",
            "what_it_fixes": "Outdated baseline behavior and dependency mismatch.",
            "risk": "Regression risk if upgrading without gate checks.",
            "integration_point": "Version-audited baseline model in ml_tabular family.",
            "expected_gain": "Stable and reproducible tabular baseline comparisons.",
            "verification_test": "Gate S smoke + strict comparable delta tracking.",
            "primary_link": "https://github.com/microsoft/LightGBM/releases",
            "status": "verified_primary",
        },
        {
            "topic": "tabular_non_tree",
            "source": "TabPFN docs (official)",
            "problem": "Fast strong tabular generalization without heavy per-task tuning.",
            "core_mechanism": "Prior-data fitted network inference for tabular tasks.",
            "what_it_fixes": "Weak performance of homogeneous tree-only candidate sets on some slices.",
            "risk": "Dependency and hardware compatibility limits.",
            "integration_point": "Optional candidate injection in binary/heavy-tail lanes with graceful fallback.",
            "expected_gain": "Diversity uplift and anti-collapse benefits.",
            "verification_test": "Admission gate + fallback_fraction telemetry.",
            "primary_link": "https://priorlabs.ai/docs/getting-started",
            "status": "verified_primary",
        },
    ]
    return rows


def _build_citation_correction_rows() -> List[Dict[str, Any]]:
    """Corrections for previously cited but mismatched/unverified references."""
    return [
        {
            "reference_item": "Chronos-2",
            "previous_claim": "arXiv:2503.06548",
            "verification_result": "Mismatch: arXiv:2503.06548 is unrelated to Chronos-2.",
            "action": "Use verified Chronos-2 link.",
            "primary_link": "https://arxiv.org/abs/2510.15821",
            "status": "corrected",
        },
        {
            "reference_item": "TiRex",
            "previous_claim": "arXiv:2502.13995",
            "verification_result": "Mismatch: arXiv:2502.13995 is unrelated.",
            "action": "Use verified TiRex link.",
            "primary_link": "https://arxiv.org/abs/2505.23719",
            "status": "corrected",
        },
        {
            "reference_item": "TimeOmni-1",
            "previous_claim": "arXiv:2502.15638",
            "verification_result": "Mismatch: arXiv:2502.15638 is unrelated.",
            "action": "Use verified TimeOmni-1 link.",
            "primary_link": "https://arxiv.org/abs/2509.24803",
            "status": "corrected",
        },
        {
            "reference_item": "DRLTSF",
            "previous_claim": "arXiv:2508.07481",
            "verification_result": "Unverified title/ID mapping from primary source.",
            "action": "Keep as hypothesis only; exclude from design-critical evidence.",
            "primary_link": "https://arxiv.org/search/?query=DRLTSF&searchtype=all",
            "status": "hypothesis",
        },
        {
            "reference_item": "Moirai2 naming",
            "previous_claim": "arXiv:2410.10469 referenced as Moirai2",
            "verification_result": "Primary title identifies Moirai-MoE; mapping to 'Moirai2' label should be treated as alias.",
            "action": "Use primary title in evidence table and keep alias explicitly marked.",
            "primary_link": "https://arxiv.org/abs/2410.10469",
            "status": "clarified",
        },
    ]


def _load_audit_gate_snapshot(output_dir: Path) -> List[Dict[str, Any]]:
    """Load latest audit status (if present) for master-doc gate snapshot section."""
    rows: List[Dict[str, Any]] = []
    data_path = output_dir / "data_integrity_audit_latest.json"
    inv_path = output_dir / "investors_count_stability_audit_latest.json"

    if data_path.exists():
        try:
            payload = json.loads(data_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "audit": "data_integrity",
                    "generated_at_utc": payload.get("generated_at_utc"),
                    "overall_pass": payload.get("overall_pass"),
                    "key_signal": _safe_json(payload.get("split", {}).get("checks", {})),
                    "evidence_path": _display_path(data_path),
                }
            )
            rows.append(
                {
                    "audit": "data_integrity.freeze_gate",
                    "generated_at_utc": payload.get("generated_at_utc"),
                    "overall_pass": payload.get("freeze_gate", {}).get("all_pass"),
                    "key_signal": _safe_json(
                        {
                            "exit_code": payload.get("freeze_gate", {}).get("exit_code"),
                            "n_checks": len(payload.get("freeze_gate", {}).get("checks", [])),
                            "fallback_mode": payload.get("freeze_gate", {}).get("fallback_mode"),
                        }
                    ),
                    "evidence_path": _display_path(data_path),
                }
            )
        except Exception:
            pass

    if inv_path.exists():
        try:
            payload = json.loads(inv_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "audit": "investors_count_stability",
                    "generated_at_utc": payload.get("generated_at_utc"),
                    "overall_pass": payload.get("overall_pass"),
                    "key_signal": _safe_json(
                        {
                            "strict_record_count": payload.get("strict_record_count"),
                            "catastrophic_spikes": payload.get("catastrophic_spikes"),
                            "guard_telemetry": payload.get("guard_telemetry", {}),
                        }
                    ),
                    "evidence_path": _display_path(inv_path),
                }
            )
        except Exception:
            pass

    if not rows:
        rows.append(
            {
                "audit": "gate_snapshot",
                "generated_at_utc": None,
                "overall_pass": None,
                "key_signal": "No latest audit artifacts found.",
                "evidence_path": _display_path(output_dir),
            }
        )
    return rows


def _target_family(target: str) -> str:
    if target == "is_funded":
        return "binary"
    if target == "investors_count":
        return "count"
    if target == "funding_raised_usd":
        return "heavy_tail"
    return "unknown"


def _horizon_band_from_value(horizon: Optional[int]) -> str:
    if horizon is None:
        return "unknown"
    h = int(horizon)
    if h <= 7:
        return "short"
    if h <= 14:
        return "mid"
    return "long"


def _build_subtasks_by_target_full(
    condition_inventory_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    for row in condition_inventory_rows:
        target = str(row.get("target", ""))
        if target not in {"is_funded", "funding_raised_usd", "investors_count"}:
            continue
        task = str(row.get("task", ""))
        ablation = str(row.get("ablation", ""))
        horizon = _to_int(row.get("horizon"))
        if horizon is None:
            continue
        rows_out.append(
            {
                "subtask_id": f"{task}__{ablation}__{target}__h{horizon}",
                "task": task,
                "ablation": ablation,
                "target": target,
                "target_family": _target_family(target),
                "horizon": horizon,
                "strict_completed": bool(row.get("strict_completed")),
                "legacy_completed": bool(row.get("legacy_completed")),
                "best_model_strict": row.get("best_model_strict"),
                "best_category_strict": row.get("best_category_strict"),
                "best_mae_strict": row.get("best_mae_strict"),
                "evidence_path": "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv",
            }
        )
    return sorted(
        rows_out,
        key=lambda r: (str(r["target"]), str(r["task"]), str(r["ablation"]), int(r["horizon"])),
    )


def _build_top3_representative_models_by_target(
    condition_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_target_model: Dict[str, Counter] = defaultdict(Counter)
    by_target_category: Dict[str, Dict[str, str]] = defaultdict(dict)
    completed_by_target: Counter = Counter()

    for row in condition_rows:
        if not bool(row.get("condition_completed")):
            continue
        target = str(row.get("target", ""))
        model = str(row.get("best_model", ""))
        category = str(row.get("best_category", ""))
        if not target or not model:
            continue
        completed_by_target[target] += 1
        by_target_model[target][model] += 1
        if category:
            by_target_category[target][model] = category

    rows_out: List[Dict[str, Any]] = []
    for target in sorted(by_target_model.keys()):
        total = max(int(completed_by_target.get(target, 0)), 1)
        top3 = by_target_model[target].most_common(3)
        for rank, (model_name, wins) in enumerate(top3, start=1):
            rows_out.append(
                {
                    "target": target,
                    "target_family": _target_family(target),
                    "rank": rank,
                    "model_name": model_name,
                    "category": by_target_category[target].get(model_name),
                    "win_count": wins,
                    "win_rate": float(wins) / float(total),
                    "total_conditions": total,
                    "evidence_path": "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
                }
            )

    return rows_out


def _build_family_gap_by_target(
    strict_records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_target_global_best: Dict[str, Tuple[float, str, str]] = {}
    by_target_category_best: Dict[Tuple[str, str], Tuple[float, str, str]] = {}

    for row in strict_records:
        target = str(row.get("target", ""))
        category = str(row.get("category", ""))
        model_name = str(row.get("model_name", ""))
        source = str(row.get("_source_path", ""))
        mae = _to_float(row.get("mae"))
        if not target or not category or not model_name or mae is None:
            continue

        g = by_target_global_best.get(target)
        if g is None or mae < g[0]:
            by_target_global_best[target] = (mae, model_name, category)

        ck = (target, category)
        c = by_target_category_best.get(ck)
        if c is None or mae < c[0]:
            by_target_category_best[ck] = (mae, model_name, source)

    rows_out: List[Dict[str, Any]] = []
    for (target, category), (best_mae, best_model, evidence_source) in sorted(by_target_category_best.items()):
        global_best = by_target_global_best.get(target)
        if global_best is None:
            continue
        global_best_mae, global_best_model, global_best_category = global_best
        gap_pct = (best_mae / max(global_best_mae, 1e-12) - 1.0) * 100.0
        rows_out.append(
            {
                "target": target,
                "target_family": _target_family(target),
                "category": category,
                "category_best_model": best_model,
                "category_best_mae": best_mae,
                "global_best_model": global_best_model,
                "global_best_category": global_best_category,
                "global_best_mae": global_best_mae,
                "gap_vs_global_best_pct": gap_pct,
                "evidence_path": evidence_source or "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
            }
        )

    return rows_out


def _build_champion_template_library(
    condition_rows: Sequence[Dict[str, Any]],
    failure_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped_winners: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    condition_counts: Counter = Counter()
    for row in condition_rows:
        if not bool(row.get("condition_completed")):
            continue
        target = str(row.get("target", ""))
        ablation = str(row.get("ablation", ""))
        horizon = _to_int(row.get("horizon"))
        best_model = str(row.get("best_model", ""))
        if not target or not ablation or horizon is None or not best_model:
            continue
        key = (_target_family(target), _horizon_band_from_value(horizon), ablation)
        grouped_winners[key][best_model] += 1
        condition_counts[key] += 1

    failure_counter: Counter = Counter()
    for row in failure_rows:
        target = str(row.get("target", ""))
        if not target:
            continue
        family = _target_family(target)
        issue = str(row.get("issue_type", ""))
        if issue:
            failure_counter[(family, issue)] += 1

    rows_out: List[Dict[str, Any]] = []
    for key in sorted(grouped_winners.keys()):
        family, horizon_band, ablation = key
        winners = grouped_winners[key]
        total = int(condition_counts.get(key, 0))
        if total <= 0:
            continue
        ranked = winners.most_common(3)
        primary_anchor = ranked[0][0] if ranked else None
        backup = [m for m, _ in ranked[1:]]
        dist = {
            model: {
                "wins": int(cnt),
                "win_rate": float(cnt) / float(total),
            }
            for model, cnt in ranked
        }
        related_failures = {
            issue: int(cnt)
            for (fam, issue), cnt in failure_counter.items()
            if fam == family
        }
        rows_out.append(
            {
                "template_id": f"{family}__{horizon_band}__{ablation}",
                "target_family": family,
                "horizon_band": horizon_band,
                "ablation": ablation,
                "primary_anchor": primary_anchor,
                "backup_anchors": ",".join(backup),
                "n_conditions": total,
                "winner_distribution_json": _safe_json(dist),
                "failure_signals_json": _safe_json(related_failures),
                "evidence_path": "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
            }
        )
    return rows_out


def _build_model_family_coverage_audit(
    strict_records: Sequence[Dict[str, Any]],
    condition_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    registered = {
        k: sorted(v)
        for k, v in _registered_models_catalog().items()
    }

    models_by_category: Dict[str, set] = defaultdict(set)
    for row in strict_records:
        category = str(row.get("category", ""))
        model_name = str(row.get("model_name", ""))
        if category and model_name:
            models_by_category[category].add(model_name)

    wins_by_category: Counter = Counter()
    for row in condition_rows:
        if not bool(row.get("condition_completed")):
            continue
        category = str(row.get("best_category", ""))
        if category:
            wins_by_category[category] += 1

    rows_out: List[Dict[str, Any]] = []
    categories = sorted(
        set(models_by_category.keys()).union(wins_by_category.keys()).union(registered.keys())
    )
    total_wins = sum(wins_by_category.values())
    for category in categories:
        model_set = sorted(models_by_category.get(category, set()))
        reg_models = registered.get(category, [])
        missing_models = sorted(set(reg_models) - set(model_set))
        rows_out.append(
            {
                "category": category,
                "n_models_registered": len(reg_models),
                "n_models_observed_strict": len(model_set),
                "strict_model_coverage_ratio": (
                    float(len(model_set)) / float(len(reg_models))
                    if len(reg_models) > 0 else None
                ),
                "models_observed_strict": ";".join(model_set),
                "missing_models_strict": ";".join(missing_models),
                "condition_wins": int(wins_by_category.get(category, 0)),
                "condition_win_share": (
                    float(wins_by_category.get(category, 0)) / float(total_wins)
                    if total_wins > 0 else 0.0
                ),
                "evidence_path": "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
            }
        )
    return rows_out


def _run_command(cmd: str, timeout: int = 30) -> str:
    try:
        out = subprocess.check_output(
            ["bash", "-lc", cmd],
            universal_newlines=True,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return out
    except Exception:
        return ""


def _capture_slurm_snapshot(slurm_since: str) -> Dict[str, Any]:
    snapshot_ts = datetime.now(timezone.utc).isoformat()

    squeue_out = _run_command('squeue -u $USER -h -o "%T %j %P %R"', timeout=20)
    running_total = 0
    pending_total = 0
    running_by_partition: Counter = Counter()
    pending_by_partition: Counter = Counter()
    pending_reason: Counter = Counter()
    prefix_status_squeue: Dict[str, Counter] = {
        "p7": Counter(),
        "p7r": Counter(),
        "p7x": Counter(),
        "p7xF": Counter(),
    }

    for line in squeue_out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        status, job_name, partition, reason = parts
        if status == "RUNNING":
            running_total += 1
            running_by_partition[partition] += 1
        elif status == "PENDING":
            pending_total += 1
            pending_by_partition[partition] += 1
            pending_reason[reason] += 1

        for prefix in ["p7xF", "p7x", "p7r", "p7"]:
            if job_name.startswith(prefix + "_"):
                prefix_status_squeue[prefix][status] += 1
                break

    sacct_out = _run_command(
        "sacct -u $USER -S %s -n -X -o JobName,State" % slurm_since,
        timeout=25,
    )
    prefix_status_sacct: Dict[str, Counter] = {
        "p7": Counter(),
        "p7r": Counter(),
        "p7x": Counter(),
        "p7xF": Counter(),
    }
    for line in sacct_out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        job_name = parts[0]
        state = parts[1].split("+")[0]
        for prefix in ["p7xF", "p7x", "p7r", "p7"]:
            if job_name.startswith(prefix + "_"):
                prefix_status_sacct[prefix][state] += 1
                break

    qos_out = _run_command(
        "sacctmgr show qos iris-batch-long,iris-gpu-long format=Name,MaxJobsPU,MaxWall,Priority -P -n",
        timeout=15,
    )
    qos_caps: Dict[str, Dict[str, str]] = {}
    for line in qos_out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        name, max_jobs_pu, max_wall, priority = [p.strip() for p in parts[:4]]
        qos_caps[name] = {
            "MaxJobsPU": max_jobs_pu,
            "MaxWall": max_wall,
            "Priority": priority,
        }

    return {
        "snapshot_ts": snapshot_ts,
        "running_total": running_total,
        "pending_total": pending_total,
        "running_by_partition": dict(running_by_partition),
        "pending_by_partition": dict(pending_by_partition),
        "pending_reason_topk": [
            {"reason": reason, "count": count}
            for reason, count in pending_reason.most_common(10)
        ],
        "qos_caps": qos_caps,
        "prefix_status_squeue": {k: dict(v) for k, v in prefix_status_squeue.items()},
        "prefix_status_sacct": {k: dict(v) for k, v in prefix_status_sacct.items()},
        "commands": {
            "squeue": 'squeue -u $USER -h -o "%T %j %P %R"',
            "sacct": "sacct -u $USER -S %s -n -X -o JobName,State" % slurm_since,
            "qos": "sacctmgr show qos iris-batch-long,iris-gpu-long format=Name,MaxJobsPU,MaxWall,Priority -P -n",
        },
    }


def _format_md_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if abs(value) >= 1_000_000:
            return "%.3f" % value
        if abs(value) >= 1_000:
            return "%.3f" % value
        return "%.6f" % value
    if isinstance(value, (dict, list)):
        return _safe_json(value)
    return str(value)


def _render_markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    cols = list(columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---" for _ in cols]) + "|"
    body: List[str] = [header, sep]
    for row in rows:
        body.append(
            "| "
            + " | ".join(_format_md_value(row.get(col)) for col in cols)
            + " |"
        )
    return "\n".join(body)


def _render_slurm_snapshot_md(snapshot: Dict[str, Any]) -> str:
    summary_rows = [
        {
            "metric": "snapshot_ts",
            "value": snapshot.get("snapshot_ts"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
        },
        {
            "metric": "running_total",
            "value": snapshot.get("running_total"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
        },
        {
            "metric": "pending_total",
            "value": snapshot.get("pending_total"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
        },
        {
            "metric": "running_by_partition",
            "value": snapshot.get("running_by_partition"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
        },
        {
            "metric": "pending_by_partition",
            "value": snapshot.get("pending_by_partition"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
        },
    ]

    reason_rows = [
        {
            "reason": r.get("reason"),
            "count": r.get("count"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
        }
        for r in snapshot.get("pending_reason_topk", [])
    ]

    prefix_rows: List[Dict[str, Any]] = []
    for prefix, state_map in sorted(snapshot.get("prefix_status_squeue", {}).items()):
        prefix_rows.append(
            {
                "prefix": prefix,
                "state_counts": state_map,
                "source": "squeue",
                "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
            }
        )
    for prefix, state_map in sorted(snapshot.get("prefix_status_sacct", {}).items()):
        prefix_rows.append(
            {
                "prefix": prefix,
                "state_counts": state_map,
                "source": "sacct",
                "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
            }
        )

    lines = [
        "# Block3 Slurm Snapshot",
        "",
        "## Snapshot Summary",
        "",
        _render_markdown_table(summary_rows, ["metric", "value", "evidence_path"]),
        "",
        "## Pending Reason Top-K",
        "",
        _render_markdown_table(reason_rows, ["reason", "count", "evidence_path"]) if reason_rows else "_No pending reasons captured._",
        "",
        "## Prefix Status",
        "",
        _render_markdown_table(prefix_rows, ["prefix", "source", "state_counts", "evidence_path"]) if prefix_rows else "_No prefix status captured._",
        "",
        "## Collection Commands",
        "",
        "```bash",
        snapshot.get("commands", {}).get("squeue", ""),
        snapshot.get("commands", {}).get("sacct", ""),
        snapshot.get("commands", {}).get("qos", ""),
        "```",
        "",
    ]
    return "\n".join(lines)


def _ensure_markers(doc_text: str) -> str:
    out = doc_text
    for section_name, heading in SECTION_ORDER:
        begin = "<!-- BEGIN AUTO:%s -->" % section_name
        end = "<!-- END AUTO:%s -->" % section_name
        if begin in out and end in out:
            continue
        addition = "\n\n%s\n\n%s\n%s\n" % (heading, begin, end)
        out += addition
    return out


def _replace_marker_section(doc_text: str, section_name: str, content: str) -> str:
    begin = "<!-- BEGIN AUTO:%s -->" % section_name
    end = "<!-- END AUTO:%s -->" % section_name
    b = doc_text.find(begin)
    e = doc_text.find(end)
    if b == -1 or e == -1 or e < b:
        return doc_text
    before = doc_text[: b + len(begin)]
    after = doc_text[e:]
    payload = "\n" + content.strip() + "\n"
    return before + payload + after


def _build_master_template() -> str:
    parts = [
        "# AutoFit V7.2 Evidence Master (2026-02-17)",
        "",
        "Scope: Block 3 full-scale benchmark evidence consolidation in one document.",
        "",
        "Method: strict comparability filter (`fairness_pass=true` and `prediction_coverage_ratio>=0.98`) with explicit legacy-unverified separation.",
        "",
        "Refresh command:",
        "",
        "```bash",
        "python scripts/build_block3_truth_pack.py --include-freeze-history --update-master-doc",
        "```",
    ]
    for section_name, heading in SECTION_ORDER:
        parts.extend(
            [
                "",
                heading,
                "",
                "<!-- BEGIN AUTO:%s -->" % section_name,
                "<!-- END AUTO:%s -->" % section_name,
            ]
        )

    parts.extend(
        [
            "",
            "## Validation Gates / Risk & Rollback",
            "",
            "### Stage S (Smoke/Audit)",
            "",
            "1. Preflight gate must pass.",
            "2. Leakage audit must pass.",
            "3. Coverage/fairness audit must pass.",
            "4. Count-safe and policy logging tests must pass.",
            "",
            "### Stage P (Pilot)",
            "",
            "1. `fairness_pass = 100%`",
            "2. `investors_count` median gap vs V7 reduced by at least 50%",
            "3. `GlobalNormalizedMAE` improved by at least 8% vs V7",
            "",
            "### Stage F (Full)",
            "",
            "1. Condition-level win rate vs V7 at least 70%",
            "2. No target with median degradation worse than 3%",
            "3. No new fairness/leakage anomalies",
            "",
            "### Risks",
            "",
            "1. Count-safe clipping can over-constrain true extremes; keep lane-level toggle.",
            "2. Anchor injection can overfit historical winners; keep bounded OOF degradation guard.",
            "3. Policy layer can create hidden leakage paths; keep offline-only policy and explicit logging.",
            "4. Dependency drift can break reproducibility; keep release-note and seed audits.",
            "",
        ]
    )
    return "\n".join(parts)


def _build_master_sections(
    summary: Dict[str, Any],
    audit_gate_rows: Sequence[Dict[str, Any]],
    task_subtask_rows: Sequence[Dict[str, Any]],
    condition_inventory_rows: Sequence[Dict[str, Any]],
    derived_rows: Sequence[Dict[str, Any]],
    model_family_coverage_rows: Sequence[Dict[str, Any]],
    target_subtask_rows: Sequence[Dict[str, Any]],
    top3_rows: Sequence[Dict[str, Any]],
    family_gap_rows: Sequence[Dict[str, Any]],
    champion_template_rows: Sequence[Dict[str, Any]],
    hyperparam_ledger_rows: Sequence[Dict[str, Any]],
    best_config_rows: Sequence[Dict[str, Any]],
    compute_cost_rows: Sequence[Dict[str, Any]],
    pilot_gate_rows: Sequence[Dict[str, Any]],
    ledger_rows: Sequence[Dict[str, Any]],
    observation_rows: Sequence[Dict[str, Any]],
    ladder_rows: Sequence[Dict[str, Any]],
    delta_rows: Sequence[Dict[str, Any]],
    sota_rows: Sequence[Dict[str, Any]],
    primary_literature_rows: Sequence[Dict[str, Any]],
    citation_correction_rows: Sequence[Dict[str, Any]],
    slurm_snapshot: Dict[str, Any],
) -> Dict[str, str]:
    evidence_rows = [
        {
            "metric": "bench_dirs",
            "value": ";".join(summary.get("bench_dirs", [])),
            "evidence_path": "docs/benchmarks/block3_truth_pack/truth_pack_summary.json",
        },
        {
            "metric": "raw_records",
            "value": summary.get("raw_records"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/truth_pack_summary.json",
        },
        {
            "metric": "strict_records",
            "value": summary.get("strict_records"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/truth_pack_summary.json",
        },
        {
            "metric": "legacy_unverified_records",
            "value": summary.get("legacy_unverified_records"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/truth_pack_summary.json",
        },
        {
            "metric": "strict_condition_completion",
            "value": summary.get("strict_condition_completion"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/truth_pack_summary.json",
        },
        {
            "metric": "legacy_condition_completion",
            "value": summary.get("legacy_condition_completion"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/truth_pack_summary.json",
        },
        {
            "metric": "v71_win_rate_vs_v7",
            "value": summary.get("v71_win_rate_vs_v7"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/v71_vs_v7_overlap.csv",
        },
        {
            "metric": "v71_median_relative_gain_vs_v7_pct",
            "value": summary.get("v71_median_relative_gain_vs_v7_pct"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/v71_vs_v7_overlap.csv",
        },
        {
            "metric": "critical_failures",
            "value": summary.get("critical_failures"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/failure_taxonomy.csv",
        },
        {
            "metric": "v72_pilot_overall_pass",
            "value": summary.get("v72_pilot_overall_pass"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json",
        },
        {
            "metric": "v72_pilot_overlap_keys",
            "value": summary.get("v72_pilot_overlap_keys"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json",
        },
    ]

    cond_rows_for_doc = []
    for row in condition_inventory_rows:
        item = dict(row)
        item["evidence_path"] = "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv"
        cond_rows_for_doc.append(item)

    ledger_rows_for_doc = []
    for row in ledger_rows:
        item = dict(row)
        item["evidence_path"] = "runs/benchmarks/%s" % row.get("run_name")
        ledger_rows_for_doc.append(item)

    delta_rows_for_doc = []
    for row in delta_rows:
        item = dict(row)
        item["evidence_path"] = "docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv"
        delta_rows_for_doc.append(item)

    slurm_summary_rows = [
        {
            "metric": "snapshot_ts",
            "value": slurm_snapshot.get("snapshot_ts"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
        },
        {
            "metric": "running_total",
            "value": slurm_snapshot.get("running_total"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
        },
        {
            "metric": "pending_total",
            "value": slurm_snapshot.get("pending_total"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
        },
        {
            "metric": "running_by_partition",
            "value": slurm_snapshot.get("running_by_partition"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
        },
        {
            "metric": "pending_by_partition",
            "value": slurm_snapshot.get("pending_by_partition"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
        },
    ]

    sections: Dict[str, str] = {}
    sections["EVIDENCE_SNAPSHOT"] = "\n".join(
        [
            "Generated from strict/legacy layered truth pack outputs under `docs/benchmarks/block3_truth_pack/`.",
            "",
            _render_markdown_table(evidence_rows, ["metric", "value", "evidence_path"]),
        ]
    )

    sections["AUDIT_GATES"] = "\n".join(
        [
            "Gate snapshot is sourced from latest read-only audit artifacts.",
            "",
            _render_markdown_table(
                audit_gate_rows,
                [
                    "audit",
                    "generated_at_utc",
                    "overall_pass",
                    "key_signal",
                    "evidence_path",
                ],
            ),
        ]
    )

    sections["TASK_AND_SUBTASK_UNIVERSE"] = "\n".join(
        [
            "### Task/Subtask Catalog",
            "",
            _render_markdown_table(
                task_subtask_rows,
                [
                    "subtask_id",
                    "subtask_family",
                    "definition_rule",
                    "key_count",
                    "key_coverage_strict",
                    "evidence_path",
                ],
            ),
            "",
            "### Full Condition Inventory",
            "",
            _render_markdown_table(
                cond_rows_for_doc,
                [
                    "task",
                    "ablation",
                    "target",
                    "horizon",
                    "expected",
                    "strict_completed",
                    "legacy_completed",
                    "best_model_strict",
                    "best_category_strict",
                    "best_mae_strict",
                    "evidence_path",
                ],
            ),
        ]
    )

    sections["DATA_CHARACTERISTIC_DERIVED_SUBTASKS"] = "\n".join(
        [
            "Derived subtasks are created from target lane semantics and target missingness/cardinality profile.",
            "",
            _render_markdown_table(
                derived_rows,
                [
                    "subtask_id",
                    "subtask_family",
                    "definition_rule",
                    "key_count",
                    "key_coverage_strict",
                    "evidence_path",
                ],
            ),
        ]
    )

    sections["MODEL_FAMILY_COVERAGE_AUDIT"] = "\n".join(
        [
            _render_markdown_table(
                model_family_coverage_rows,
                [
                    "category",
                    "n_models_registered",
                    "n_models_observed_strict",
                    "strict_model_coverage_ratio",
                    "models_observed_strict",
                    "missing_models_strict",
                    "condition_wins",
                    "condition_win_share",
                    "evidence_path",
                ],
            )
        ]
    )

    sections["TARGET_SUBTASKS"] = "\n".join(
        [
            _render_markdown_table(
                target_subtask_rows,
                [
                    "subtask_id",
                    "task",
                    "ablation",
                    "target",
                    "target_family",
                    "horizon",
                    "strict_completed",
                    "legacy_completed",
                    "best_model_strict",
                    "best_category_strict",
                    "best_mae_strict",
                    "evidence_path",
                ],
            )
        ]
    )

    sections["TOP3_REPRESENTATIVE_MODELS"] = "\n".join(
        [
            _render_markdown_table(
                top3_rows,
                [
                    "target",
                    "target_family",
                    "rank",
                    "model_name",
                    "category",
                    "win_count",
                    "win_rate",
                    "total_conditions",
                    "evidence_path",
                ],
            )
        ]
    )

    sections["FAMILY_GAP_MATRIX"] = "\n".join(
        [
            _render_markdown_table(
                family_gap_rows,
                [
                    "target",
                    "target_family",
                    "category",
                    "category_best_model",
                    "category_best_mae",
                    "global_best_model",
                    "global_best_category",
                    "global_best_mae",
                    "gap_vs_global_best_pct",
                    "evidence_path",
                ],
            )
        ]
    )

    sections["CHAMPION_TEMPLATE_LIBRARY"] = "\n".join(
        [
            _render_markdown_table(
                champion_template_rows,
                [
                    "template_id",
                    "target_family",
                    "horizon_band",
                    "ablation",
                    "primary_anchor",
                    "backup_anchors",
                    "n_conditions",
                    "winner_distribution_json",
                    "failure_signals_json",
                    "evidence_path",
                ],
            )
        ]
    )

    sections["HYPERPARAMETER_SEARCH_LEDGER"] = "\n".join(
        [
            _render_markdown_table(
                hyperparam_ledger_rows,
                [
                    "target",
                    "target_family",
                    "priority_rank",
                    "model_name",
                    "category",
                    "search_budget",
                    "trials_executed",
                    "status",
                    "best_mae_observed_strict",
                    "best_config_json",
                    "search_space_json",
                    "selection_scope",
                    "evidence_path",
                ],
            )
        ]
    )

    sections["BEST_CONFIG_BY_MODEL_TARGET"] = "\n".join(
        [
            _render_markdown_table(
                best_config_rows,
                [
                    "target",
                    "model_name",
                    "target_family",
                    "category",
                    "status",
                    "search_budget",
                    "trials_executed",
                    "best_mae_observed_strict",
                    "best_config_json",
                    "search_space_json",
                    "evidence_path",
                ],
            )
        ]
    )

    sections["COMPUTE_COST_REPORT"] = "\n".join(
        [
            _render_markdown_table(
                compute_cost_rows,
                [
                    "model_name",
                    "category",
                    "target",
                    "strict_records",
                    "train_time_median_seconds",
                    "inference_time_median_seconds",
                    "evidence_path",
                ],
            )
        ]
    )

    sections["V72_PILOT_GATE_REPORT"] = "\n".join(
        [
            _render_markdown_table(
                pilot_gate_rows,
                [
                    "section",
                    "key",
                    "value",
                    "evidence_path",
                ],
            )
        ]
    )

    sections["HISTORICAL_FULL_SCALE_EXPERIMENT_LEDGER"] = "\n".join(
        [
            "### Run Ledger",
            "",
            _render_markdown_table(
                ledger_rows_for_doc,
                [
                    "run_name",
                    "run_stage",
                    "raw_records",
                    "strict_records",
                    "legacy_records",
                    "strict_ratio",
                    "models",
                    "categories",
                    "condition_coverage_strict",
                    "condition_coverage_legacy",
                    "best_model_by_target_json",
                    "key_failures",
                    "evidence_path",
                ],
            ),
            "",
            "### Observations",
            "",
            _render_markdown_table(
                observation_rows,
                [
                    "run_name",
                    "observation_type",
                    "observation",
                    "supporting_metric",
                    "evidence_path",
                ],
            ),
        ]
    )

    sections["AUTOFIT_VERSION_LADDER"] = "\n".join(
        [
            "### Version Ladder",
            "",
            _render_markdown_table(
                ladder_rows,
                [
                    "version",
                    "commit_hint",
                    "core_changes",
                    "inspiration_source",
                    "measured_targets",
                    "median_mae_by_target_json",
                    "median_gap_vs_best_non_autofit_json",
                    "primary_failure_mode",
                    "evidence_path",
                ],
            ),
            "",
            "### Step Deltas",
            "",
            _render_markdown_table(
                delta_rows_for_doc,
                [
                    "from_version",
                    "to_version",
                    "target",
                    "overlap_keys",
                    "median_mae_delta_pct",
                    "median_gap_delta_pct",
                    "evidence_path",
                ],
            ),
        ]
    )

    sections["HIGH_VALUE_SOTA_COMPONENTS"] = "\n".join(
        [
            _render_markdown_table(
                sota_rows,
                [
                    "feature_component",
                    "winner_evidence",
                    "affected_subtasks",
                    "why_effective",
                    "integration_priority",
                    "risk",
                    "verification_test",
                    "evidence_path",
                ],
            )
        ]
    )

    sections["PRIMARY_LITERATURE_MATRIX"] = "\n".join(
        [
            "Strict-primary sources only (arXiv originals, official release notes, official docs).",
            "",
            _render_markdown_table(
                primary_literature_rows,
                [
                    "topic",
                    "source",
                    "problem",
                    "core_mechanism",
                    "what_it_fixes",
                    "risk",
                    "integration_point",
                    "expected_gain",
                    "verification_test",
                    "primary_link",
                    "status",
                ],
            ),
        ]
    )

    sections["CITATION_CORRECTION_LOG"] = "\n".join(
        [
            "Unverified or mismatched legacy references are explicitly corrected or marked as hypothesis.",
            "",
            _render_markdown_table(
                citation_correction_rows,
                [
                    "reference_item",
                    "previous_claim",
                    "verification_result",
                    "action",
                    "primary_link",
                    "status",
                ],
            ),
        ]
    )

    prefix_rows = []
    for prefix, counts in sorted(slurm_snapshot.get("prefix_status_squeue", {}).items()):
        prefix_rows.append(
            {
                "prefix": prefix,
                "source": "squeue",
                "state_counts": counts,
                "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
            }
        )
    for prefix, counts in sorted(slurm_snapshot.get("prefix_status_sacct", {}).items()):
        prefix_rows.append(
            {
                "prefix": prefix,
                "source": "sacct",
                "state_counts": counts,
                "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
            }
        )

    pending_rows = [
        {
            "reason": r.get("reason"),
            "count": r.get("count"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
        }
        for r in slurm_snapshot.get("pending_reason_topk", [])
    ]

    sections["LIVE_SLURM_SNAPSHOT"] = "\n".join(
        [
            "Snapshot timestamp is absolute and all queue conclusions are time-bounded.",
            "",
            _render_markdown_table(slurm_summary_rows, ["metric", "value", "evidence_path"]),
            "",
            "### Pending Reasons",
            "",
            _render_markdown_table(pending_rows, ["reason", "count", "evidence_path"]) if pending_rows else "_No pending reasons captured._",
            "",
            "### Prefix Status (p7/p7r/p7x/p7xF)",
            "",
            _render_markdown_table(prefix_rows, ["prefix", "source", "state_counts", "evidence_path"]) if prefix_rows else "_No prefix status captured._",
            "",
            "### Collection Commands",
            "",
            "```bash",
            slurm_snapshot.get("commands", {}).get("squeue", ""),
            slurm_snapshot.get("commands", {}).get("sacct", ""),
            slurm_snapshot.get("commands", {}).get("qos", ""),
            "```",
        ]
    )

    return sections


def _update_master_doc(
    master_doc_path: Path,
    sections: Dict[str, str],
) -> None:
    # Deterministic regeneration prevents section drift from manual edits.
    text = _build_master_template()
    for section_name, _heading in SECTION_ORDER:
        payload = sections.get(section_name, "")
        text = _replace_marker_section(text, section_name, payload)

    master_doc_path.parent.mkdir(parents=True, exist_ok=True)
    master_doc_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_summary_docs(
    summary: Dict[str, Any],
    slurm_snapshot: Dict[str, Any],
    master_doc_path: Path,
) -> None:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    master_path_display = _display_path(master_doc_path)

    status_path = ROOT / "docs" / "BLOCK3_MODEL_STATUS.md"
    results_path = ROOT / "docs" / "BLOCK3_RESULTS.md"

    status_text = "\n".join(
        [
            "# Block 3 Model Benchmark Status",
            "",
            "> Last Updated: %s" % now_iso,
            "> Single source of truth: `%s`" % master_path_display,
            "",
            "## Snapshot",
            "",
            "| Metric | Value | Evidence |",
            "|---|---:|---|",
            "| raw_records | %s | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |" % summary.get("raw_records"),
            "| strict_records | %s | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |" % summary.get("strict_records"),
            "| legacy_unverified_records | %s | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |" % summary.get("legacy_unverified_records"),
            "| strict_condition_completion | %s | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |" % summary.get("strict_condition_completion"),
            "| running_total | %s | `docs/benchmarks/block3_truth_pack/slurm_snapshot.json` |" % slurm_snapshot.get("running_total"),
            "| pending_total | %s | `docs/benchmarks/block3_truth_pack/slurm_snapshot.json` |" % slurm_snapshot.get("pending_total"),
            "",
            "## Notes",
            "",
            "1. Detailed tables are centralized in the master evidence document.",
            "2. This file is intentionally lightweight to avoid drift.",
            "",
        ]
    )

    results_text = "\n".join(
        [
            "# Block 3 Benchmark Results",
            "",
            "> Last Updated: %s" % now_iso,
            "> Single source of truth: `%s`" % master_path_display,
            "",
            "## Snapshot",
            "",
            "| Metric | Value | Evidence |",
            "|---|---:|---|",
            "| strict_records | %s | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |" % summary.get("strict_records"),
            "| strict_completion_ratio | %s | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |" % summary.get("strict_condition_completion_ratio"),
            "| v71_win_rate_vs_v7 | %s | `docs/benchmarks/block3_truth_pack/v71_vs_v7_overlap.csv` |" % summary.get("v71_win_rate_vs_v7"),
            "| v71_median_relative_gain_vs_v7_pct | %s | `docs/benchmarks/block3_truth_pack/v71_vs_v7_overlap.csv` |" % summary.get("v71_median_relative_gain_vs_v7_pct"),
            "| critical_failures | %s | `docs/benchmarks/block3_truth_pack/failure_taxonomy.csv` |" % summary.get("critical_failures"),
            "| running_total | %s | `docs/benchmarks/block3_truth_pack/slurm_snapshot.json` |" % slurm_snapshot.get("running_total"),
            "| pending_total | %s | `docs/benchmarks/block3_truth_pack/slurm_snapshot.json` |" % slurm_snapshot.get("pending_total"),
            "",
            "## Notes",
            "",
            "1. Use the master evidence document for full task/subtask, ladder, and SOTA analysis tables.",
            "2. This file keeps only high-level snapshot metrics.",
            "",
        ]
    )

    status_path.write_text(status_text, encoding="utf-8")
    results_path.write_text(results_text, encoding="utf-8")


def build_truth_pack(
    bench_dirs: Sequence[Path],
    config_path: Path,
    output_dir: Path,
    min_coverage: float,
    capture_slurm: bool,
    slurm_since: str,
    update_master_doc: bool,
    master_doc_path: Path,
    duplicate_jobs_removed: int = 0,
) -> Dict[str, Any]:
    raw_records: List[Dict[str, Any]] = []
    for bdir in bench_dirs:
        raw_records.extend(_load_metrics_records(bdir))

    buckets = _prepare_records(raw_records, min_coverage=min_coverage)
    strict_records = buckets["strict_comparable"]
    legacy_records = buckets["legacy_unverified"]
    strict_excluded_records = buckets["strict_excluded"]
    all_valid_records = buckets["valid_all"]

    expected_conditions = _load_expected_conditions(config_path)

    condition_rows, best_non_by_condition, _ = _build_condition_leaderboard(
        strict_records=strict_records,
        expected_conditions=expected_conditions,
    )
    lineage_rows = _build_autofit_lineage(
        strict_records=strict_records,
        expected_conditions=expected_conditions,
        best_non_by_condition=best_non_by_condition,
    )
    overlap_rows = _build_v71_vs_v7_overlap(
        strict_records=strict_records,
        expected_conditions=expected_conditions,
    )
    failure_rows = _build_failure_taxonomy(
        all_valid_records=all_valid_records,
        condition_rows=condition_rows,
        min_coverage=min_coverage,
    )

    condition_inventory_rows, strict_cond_set, legacy_cond_set = _build_condition_inventory_full(
        expected_conditions=expected_conditions,
        strict_records=strict_records,
        legacy_records=legacy_records,
    )

    target_stats = _load_target_stats_from_column_manifest(
        ROOT / "docs" / "audits" / "FULL_SCALE_POINTER.yaml"
    )
    task_subtask_rows, derived_rows = _build_task_subtask_catalog(
        expected_conditions=expected_conditions,
        strict_cond_set=strict_cond_set,
        target_stats=target_stats,
    )

    run_history_ledger_rows, run_history_obs_rows = _build_run_history_ledger(
        bench_dirs=bench_dirs,
        all_records=all_valid_records,
        strict_records=strict_records,
        legacy_records=legacy_records,
        failure_rows=failure_rows,
        expected_conditions=expected_conditions,
        min_coverage=min_coverage,
    )

    ladder_rows = _build_autofit_version_ladder(
        lineage_rows=lineage_rows,
        failure_rows=failure_rows,
    )
    delta_rows = _build_autofit_step_deltas(
        strict_records=strict_records,
        best_non_by_condition=best_non_by_condition,
        expected_conditions=expected_conditions,
    )
    model_family_coverage_rows = _build_model_family_coverage_audit(
        strict_records=strict_records,
        condition_rows=condition_rows,
    )
    target_subtask_rows = _build_subtasks_by_target_full(
        condition_inventory_rows=condition_inventory_rows,
    )
    top3_rows = _build_top3_representative_models_by_target(
        condition_rows=condition_rows,
    )
    family_gap_rows = _build_family_gap_by_target(
        strict_records=strict_records,
    )
    champion_template_rows = _build_champion_template_library(
        condition_rows=condition_rows,
        failure_rows=failure_rows,
    )
    v72_missing_manifest_rows, v72_missing_keys, v72_coverage_ratio = _build_v72_missing_key_manifest(
        strict_records=strict_records,
        expected_conditions=expected_conditions,
    )
    sota_rows = _build_sota_feature_value_map(
        condition_rows=condition_rows,
        failure_rows=failure_rows,
    )
    primary_literature_rows = _build_primary_literature_matrix_rows()
    citation_correction_rows = _build_citation_correction_rows()

    slurm_snapshot = _capture_slurm_snapshot(slurm_since=slurm_since) if capture_slurm else {
        "snapshot_ts": datetime.now(timezone.utc).isoformat(),
        "running_total": 0,
        "pending_total": 0,
        "running_by_partition": {},
        "pending_by_partition": {},
        "pending_reason_topk": [],
        "qos_caps": {},
        "prefix_status_squeue": {},
        "prefix_status_sacct": {},
        "commands": {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_gate_rows = _load_audit_gate_snapshot(output_dir)

    # Legacy outputs retained for compatibility
    condition_path = output_dir / "condition_leaderboard.csv"
    lineage_path = output_dir / "autofit_lineage.csv"
    failures_path = output_dir / "failure_taxonomy.csv"
    overlap_path = output_dir / "v71_vs_v7_overlap.csv"

    _write_csv(
        condition_path,
        condition_rows,
        [
            "task",
            "ablation",
            "target",
            "horizon",
            "expected_condition",
            "condition_completed",
            "n_records",
            "best_model",
            "best_category",
            "best_mae",
            "best_non_autofit_model",
            "best_non_autofit_category",
            "best_non_autofit_mae",
            "best_autofit_model",
            "best_autofit_variant_id",
            "best_autofit_mae",
            "autofit_gap_pct",
            "bench_dirs",
            "sources",
        ],
    )
    _write_csv(
        lineage_path,
        lineage_rows,
        [
            "model_name",
            "target",
            "n_records",
            "conditions_covered",
            "condition_coverage_ratio",
            "best_mae",
            "median_mae",
            "p25_mae",
            "p75_mae",
            "worst_mae",
            "median_gap_vs_best_non_autofit_pct",
        ],
    )
    _write_csv(
        failures_path,
        failure_rows,
        [
            "issue_type",
            "severity",
            "model_name",
            "category",
            "task",
            "ablation",
            "target",
            "horizon",
            "mae",
            "prediction_coverage_ratio",
            "fairness_pass",
            "evidence_source",
            "note",
        ],
    )
    _write_csv(
        overlap_path,
        overlap_rows,
        [
            "task",
            "ablation",
            "target",
            "horizon",
            "mae_v7",
            "mae_v71",
            "relative_gain_pct",
            "v71_wins",
            "source_v7",
            "source_v71",
        ],
    )

    # Panorama outputs
    task_subtask_path = output_dir / "task_subtask_catalog.csv"
    condition_inventory_path = output_dir / "condition_inventory_full.csv"
    run_history_ledger_path = output_dir / "run_history_ledger.csv"
    run_history_obs_path = output_dir / "run_history_observations.csv"
    ladder_path = output_dir / "autofit_version_ladder.csv"
    delta_path = output_dir / "autofit_step_deltas.csv"
    v72_missing_manifest_path = output_dir / "missing_key_manifest.csv"
    model_family_coverage_path = output_dir / "model_family_coverage_audit.csv"
    target_subtasks_path = output_dir / "subtasks_by_target_full.csv"
    top3_path = output_dir / "top3_representative_models_by_target.csv"
    family_gap_path = output_dir / "family_gap_by_target.csv"
    champion_template_path = output_dir / "champion_template_library.csv"
    sota_path = output_dir / "sota_feature_value_map.csv"
    primary_lit_path = output_dir / "primary_literature_matrix.csv"
    citation_correction_path = output_dir / "citation_correction_log.csv"
    audit_gate_path = output_dir / "audit_gate_snapshot.csv"
    hyperparam_ledger_path = output_dir / "hyperparam_search_ledger.csv"
    best_config_path = output_dir / "best_config_by_model_target.json"
    compute_cost_path = output_dir / "compute_cost_report.csv"
    pilot_gate_path = output_dir / "v72_pilot_gate_report.json"
    slurm_json_path = output_dir / "slurm_snapshot.json"
    slurm_md_path = output_dir / "slurm_snapshot.md"

    _write_csv(
        task_subtask_path,
        task_subtask_rows,
        [
            "subtask_id",
            "subtask_family",
            "definition_rule",
            "key_count",
            "key_coverage_strict",
            "evidence_path",
        ],
    )
    _write_csv(
        condition_inventory_path,
        condition_inventory_rows,
        [
            "task",
            "ablation",
            "target",
            "horizon",
            "expected",
            "strict_completed",
            "legacy_completed",
            "best_model_strict",
            "best_category_strict",
            "best_mae_strict",
        ],
    )
    _write_csv(
        run_history_ledger_path,
        run_history_ledger_rows,
        [
            "run_name",
            "run_stage",
            "raw_records",
            "strict_records",
            "legacy_records",
            "strict_ratio",
            "models",
            "categories",
            "condition_coverage_strict",
            "condition_coverage_legacy",
            "best_model_by_target_json",
            "key_failures",
        ],
    )
    _write_csv(
        run_history_obs_path,
        run_history_obs_rows,
        [
            "run_name",
            "observation_type",
            "observation",
            "supporting_metric",
            "evidence_path",
        ],
    )
    _write_csv(
        ladder_path,
        ladder_rows,
        [
            "version",
            "commit_hint",
            "core_changes",
            "inspiration_source",
            "measured_targets",
            "median_mae_by_target_json",
            "median_gap_vs_best_non_autofit_json",
            "primary_failure_mode",
            "evidence_path",
        ],
    )
    _write_csv(
        delta_path,
        delta_rows,
        [
            "from_version",
            "to_version",
            "target",
            "overlap_keys",
            "median_mae_delta_pct",
            "median_gap_delta_pct",
        ],
    )
    _write_csv(
        model_family_coverage_path,
        model_family_coverage_rows,
        [
            "category",
            "n_models_registered",
            "n_models_observed_strict",
            "strict_model_coverage_ratio",
            "models_observed_strict",
            "missing_models_strict",
            "condition_wins",
            "condition_win_share",
            "evidence_path",
        ],
    )
    _write_csv(
        target_subtasks_path,
        target_subtask_rows,
        [
            "subtask_id",
            "task",
            "ablation",
            "target",
            "target_family",
            "horizon",
            "strict_completed",
            "legacy_completed",
            "best_model_strict",
            "best_category_strict",
            "best_mae_strict",
            "evidence_path",
        ],
    )
    _write_csv(
        top3_path,
        top3_rows,
        [
            "target",
            "target_family",
            "rank",
            "model_name",
            "category",
            "win_count",
            "win_rate",
            "total_conditions",
            "evidence_path",
        ],
    )
    _write_csv(
        family_gap_path,
        family_gap_rows,
        [
            "target",
            "target_family",
            "category",
            "category_best_model",
            "category_best_mae",
            "global_best_model",
            "global_best_category",
            "global_best_mae",
            "gap_vs_global_best_pct",
            "evidence_path",
        ],
    )
    _write_csv(
        champion_template_path,
        champion_template_rows,
        [
            "template_id",
            "target_family",
            "horizon_band",
            "ablation",
            "primary_anchor",
            "backup_anchors",
            "n_conditions",
            "winner_distribution_json",
            "failure_signals_json",
            "evidence_path",
        ],
    )
    _write_csv(
        v72_missing_manifest_path,
        v72_missing_manifest_rows,
        [
            "task",
            "ablation",
            "target",
            "horizon",
            "priority_rank",
            "priority_group",
            "reason",
        ],
    )
    _write_csv(
        sota_path,
        sota_rows,
        [
            "feature_component",
            "winner_evidence",
            "affected_subtasks",
            "why_effective",
            "integration_priority",
            "risk",
            "verification_test",
            "evidence_path",
        ],
    )
    _write_csv(
        primary_lit_path,
        primary_literature_rows,
        [
            "topic",
            "source",
            "problem",
            "core_mechanism",
            "what_it_fixes",
            "risk",
            "integration_point",
            "expected_gain",
            "verification_test",
            "primary_link",
            "status",
        ],
    )
    _write_csv(
        citation_correction_path,
        citation_correction_rows,
        [
            "reference_item",
            "previous_claim",
            "verification_result",
            "action",
            "primary_link",
            "status",
        ],
    )
    _write_csv(
        audit_gate_path,
        audit_gate_rows,
        [
            "audit",
            "generated_at_utc",
            "overall_pass",
            "key_signal",
            "evidence_path",
        ],
    )

    hyperparam_ledger_rows = _load_optional_csv(hyperparam_ledger_path)
    if not hyperparam_ledger_rows:
        hyperparam_ledger_rows = [
            {
                "target": "n/a",
                "target_family": "n/a",
                "priority_rank": None,
                "model_name": "n/a",
                "category": "n/a",
                "search_budget": None,
                "trials_executed": None,
                "status": "artifact_missing",
                "best_mae_observed_strict": None,
                "best_config_json": "artifact_missing",
                "search_space_json": "artifact_missing",
                "selection_scope": "train_val_oof_only",
                "evidence_path": _display_path(hyperparam_ledger_path),
            }
        ]

    best_config_rows = _build_best_config_rows(
        _load_optional_json(best_config_path),
        evidence_path=_display_path(best_config_path),
    )

    compute_cost_rows = _load_optional_csv(compute_cost_path)
    if not compute_cost_rows:
        compute_cost_rows = [
            {
                "model_name": "n/a",
                "category": "n/a",
                "target": "n/a",
                "strict_records": 0,
                "train_time_median_seconds": None,
                "inference_time_median_seconds": None,
                "evidence_path": _display_path(compute_cost_path),
            }
        ]

    pilot_gate_payload = _load_optional_json(pilot_gate_path)
    pilot_gate_rows = _build_pilot_gate_rows(
        pilot_gate_payload,
        evidence_path=_display_path(pilot_gate_path),
    )

    slurm_json_path.write_text(json.dumps(slurm_snapshot, indent=2), encoding="utf-8")
    slurm_md_path.write_text(_render_slurm_snapshot_md(slurm_snapshot), encoding="utf-8")
    snapshot_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slurm_live_json_path = output_dir / f"slurm_live_snapshot_{snapshot_stamp}.json"
    slurm_live_md_path = output_dir / f"slurm_live_snapshot_{snapshot_stamp}.md"
    slurm_live_json_path.write_text(json.dumps(slurm_snapshot, indent=2), encoding="utf-8")
    slurm_live_md_path.write_text(_render_slurm_snapshot_md(slurm_snapshot), encoding="utf-8")

    expected_total = len(expected_conditions)
    strict_completed_conditions = sum(1 for r in condition_inventory_rows if r.get("strict_completed"))
    legacy_completed_conditions = sum(1 for r in condition_inventory_rows if r.get("legacy_completed"))

    overlap_gains = [
        _to_float(r.get("relative_gain_pct"))
        for r in overlap_rows
        if _to_float(r.get("relative_gain_pct")) is not None
    ]
    overlap_win_rate = (
        float(sum(1 for g in overlap_gains if g is not None and g > 0.0)) / float(len(overlap_gains))
        if overlap_gains
        else None
    )
    overlap_median = _quantiles([g for g in overlap_gains if g is not None], [0.5])[0] if overlap_gains else None
    queue_eta_model = _estimate_queue_eta_model(slurm_snapshot)

    summary = {
        "bench_dirs": [_display_path(p) for p in bench_dirs],
        "config_path": _display_path(config_path),
        "min_coverage": min_coverage,
        "raw_records": len(all_valid_records),
        "strict_records": len(strict_records),
        "legacy_unverified_records": len(legacy_records),
        "strict_excluded_records": len(strict_excluded_records),
        "filtered_records": len(strict_records),
        "expected_conditions": expected_total,
        "strict_completed_conditions": strict_completed_conditions,
        "legacy_completed_conditions": legacy_completed_conditions,
        "strict_condition_completion": "%s/%s" % (strict_completed_conditions, expected_total),
        "legacy_condition_completion": "%s/%s" % (legacy_completed_conditions, expected_total),
        "strict_condition_completion_ratio": (
            float(strict_completed_conditions) / float(expected_total) if expected_total else None
        ),
        "legacy_condition_completion_ratio": (
            float(legacy_completed_conditions) / float(expected_total) if expected_total else None
        ),
        "overlap_rows": len(overlap_rows),
        "v71_win_rate_vs_v7": overlap_win_rate,
        "v71_median_relative_gain_vs_v7_pct": overlap_median,
        "critical_failures": sum(1 for r in failure_rows if r.get("severity") == "critical"),
        "high_failures": sum(1 for r in failure_rows if r.get("severity") == "high"),
        "v72_pilot_overall_pass": (
            pilot_gate_payload.get("overall_pass")
            if isinstance(pilot_gate_payload, dict)
            else None
        ),
        "v72_pilot_overlap_keys": (
            pilot_gate_payload.get("counts", {}).get("overlap_keys_v7_v72_non_autofit")
            if isinstance(pilot_gate_payload, dict)
            else None
        ),
        "v72_missing_keys": v72_missing_keys,
        "v72_coverage_ratio": v72_coverage_ratio,
        "duplicate_jobs_removed": int(max(0, duplicate_jobs_removed)),
        "queue_eta_model": queue_eta_model,
        "slurm_snapshot_path": _display_path(slurm_json_path),
        "slurm_live_snapshot_path": _display_path(slurm_live_json_path),
        "outputs": {
            "condition_leaderboard": _display_path(condition_path),
            "autofit_lineage": _display_path(lineage_path),
            "failure_taxonomy": _display_path(failures_path),
            "v71_vs_v7_overlap": _display_path(overlap_path),
            "missing_key_manifest": _display_path(v72_missing_manifest_path),
            "task_subtask_catalog": _display_path(task_subtask_path),
            "condition_inventory_full": _display_path(condition_inventory_path),
            "run_history_ledger": _display_path(run_history_ledger_path),
            "run_history_observations": _display_path(run_history_obs_path),
            "autofit_version_ladder": _display_path(ladder_path),
            "autofit_step_deltas": _display_path(delta_path),
            "model_family_coverage_audit": _display_path(model_family_coverage_path),
            "subtasks_by_target_full": _display_path(target_subtasks_path),
            "top3_representative_models_by_target": _display_path(top3_path),
            "family_gap_by_target": _display_path(family_gap_path),
            "champion_template_library": _display_path(champion_template_path),
            "sota_feature_value_map": _display_path(sota_path),
            "primary_literature_matrix": _display_path(primary_lit_path),
            "citation_correction_log": _display_path(citation_correction_path),
            "audit_gate_snapshot": _display_path(audit_gate_path),
            "hyperparam_search_ledger": _display_path(hyperparam_ledger_path),
            "best_config_by_model_target": _display_path(best_config_path),
            "compute_cost_report": _display_path(compute_cost_path),
            "v72_pilot_gate_report": _display_path(pilot_gate_path),
            "slurm_snapshot_json": _display_path(slurm_json_path),
            "slurm_snapshot_md": _display_path(slurm_md_path),
            "slurm_live_snapshot_json": _display_path(slurm_live_json_path),
            "slurm_live_snapshot_md": _display_path(slurm_live_md_path),
        },
    }

    summary_json_path = output_dir / "truth_pack_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_md = [
        "# Block3 Truth Pack Summary",
        "",
        "- Raw records: **%s**" % summary["raw_records"],
        "- Strict comparable records: **%s**" % summary["strict_records"],
        "- Legacy unverified records: **%s**" % summary["legacy_unverified_records"],
        "- Strict excluded records: **%s**" % summary["strict_excluded_records"],
        "- Expected condition keys: **%s**" % summary["expected_conditions"],
        "- Strict condition completion: **%s**" % summary["strict_condition_completion"],
        "- Legacy condition completion: **%s**" % summary["legacy_condition_completion"],
        (
            "- V7.1 vs V7 overlap: **%s** (win_rate=%.3f, median_gain=%.3f%%)"
            % (
                summary["overlap_rows"],
                summary["v71_win_rate_vs_v7"],
                summary["v71_median_relative_gain_vs_v7_pct"],
            )
            if summary["v71_win_rate_vs_v7"] is not None and summary["v71_median_relative_gain_vs_v7_pct"] is not None
            else "- V7.1 vs V7 overlap: **%s**" % summary["overlap_rows"]
        ),
        "- Critical failures tagged: **%s**" % summary["critical_failures"],
        "- High-severity failures tagged: **%s**" % summary["high_failures"],
        "- AutoFitV72 missing keys: **%s**" % summary["v72_missing_keys"],
        "- AutoFitV72 coverage ratio: **%.4f**" % summary["v72_coverage_ratio"],
        "- Duplicate jobs removed (reported): **%s**" % summary["duplicate_jobs_removed"],
        "- Queue ETA model: `%s`" % json.dumps(summary["queue_eta_model"], sort_keys=True),
        "- Slurm snapshot: `%s`" % summary["slurm_snapshot_path"],
        "",
        "## Output Files",
        "",
    ]
    for key, rel_path in sorted(summary["outputs"].items()):
        summary_md.append("- `%s`: `%s`" % (key, rel_path))

    summary_md_path = output_dir / "truth_pack_summary.md"
    summary_md_path.write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    if update_master_doc:
        if not master_doc_path.exists():
            master_doc_path.write_text(_build_master_template(), encoding="utf-8")

        sections = _build_master_sections(
            summary=summary,
            audit_gate_rows=audit_gate_rows,
            task_subtask_rows=task_subtask_rows,
            condition_inventory_rows=condition_inventory_rows,
            derived_rows=derived_rows,
            model_family_coverage_rows=model_family_coverage_rows,
            target_subtask_rows=target_subtask_rows,
            top3_rows=top3_rows,
            family_gap_rows=family_gap_rows,
            champion_template_rows=champion_template_rows,
            hyperparam_ledger_rows=hyperparam_ledger_rows,
            best_config_rows=best_config_rows,
            compute_cost_rows=compute_cost_rows,
            pilot_gate_rows=pilot_gate_rows,
            ledger_rows=run_history_ledger_rows,
            observation_rows=run_history_obs_rows,
            ladder_rows=ladder_rows,
            delta_rows=delta_rows,
            sota_rows=sota_rows,
            primary_literature_rows=primary_literature_rows,
            citation_correction_rows=citation_correction_rows,
            slurm_snapshot=slurm_snapshot,
        )
        _update_master_doc(master_doc_path=master_doc_path, sections=sections)
        _write_summary_docs(summary=summary, slurm_snapshot=slurm_snapshot, master_doc_path=master_doc_path)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Block3 auditable truth pack.")
    parser.add_argument(
        "--bench-dir",
        action="append",
        default=[],
        help="Benchmark directory. Repeatable; overrides --include-freeze-history lookup.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to block3 task config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated truth pack files.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.98,
        help="Minimum prediction coverage for strict comparable records.",
    )
    parser.add_argument(
        "--include-freeze-history",
        dest="include_freeze_history",
        action="store_true",
        default=True,
        help="Discover benchmark runs with --bench-glob under runs/benchmarks.",
    )
    parser.add_argument(
        "--no-include-freeze-history",
        dest="include_freeze_history",
        action="store_false",
        help="Disable automatic run discovery and rely on explicit --bench-dir or defaults.",
    )
    parser.add_argument(
        "--bench-glob",
        type=str,
        default=DEFAULT_BENCH_GLOB,
        help="Glob pattern under runs/benchmarks for discovery.",
    )
    parser.add_argument(
        "--capture-slurm",
        dest="capture_slurm",
        action="store_true",
        default=True,
        help="Capture Slurm snapshot into truth pack outputs.",
    )
    parser.add_argument(
        "--no-capture-slurm",
        dest="capture_slurm",
        action="store_false",
        help="Disable Slurm snapshot collection.",
    )
    parser.add_argument(
        "--slurm-since",
        type=str,
        default=DEFAULT_SLURM_SINCE,
        help="Since date for sacct aggregation (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--update-master-doc",
        action="store_true",
        default=False,
        help="Update docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md with auto sections.",
    )
    parser.add_argument(
        "--master-doc",
        type=Path,
        default=DEFAULT_MASTER_DOC,
        help="Master doc path to update when --update-master-doc is set.",
    )
    parser.add_argument(
        "--duplicate-jobs-removed",
        type=int,
        default=0,
        help="Number of duplicate Slurm jobs removed before this snapshot (for audit trace).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explicit_dirs = [Path(p).resolve() for p in args.bench_dir]
    bench_dirs = _resolve_bench_dirs(
        explicit_dirs=explicit_dirs,
        include_freeze_history=bool(args.include_freeze_history),
        bench_glob=str(args.bench_glob),
    )

    summary = build_truth_pack(
        bench_dirs=bench_dirs,
        config_path=args.config.resolve(),
        output_dir=args.output_dir.resolve(),
        min_coverage=float(args.min_coverage),
        capture_slurm=bool(args.capture_slurm),
        slurm_since=str(args.slurm_since),
        update_master_doc=bool(args.update_master_doc),
        master_doc_path=args.master_doc.resolve(),
        duplicate_jobs_removed=int(args.duplicate_jobs_removed),
    )
    print("Block3 truth pack generated:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
