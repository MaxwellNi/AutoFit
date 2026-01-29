#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _split_entities(
    entities: List[str], split_seed: int, train_frac: float = 0.8, val_frac: float = 0.1
) -> Tuple[set, set, set]:
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(entities)
    n = len(perm)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_ids = set(perm[:n_train])
    val_ids = set(perm[n_train : n_train + n_val])
    test_ids = set(perm[n_train + n_val :])
    return train_ids, val_ids, test_ids


def _compute_ratio_w(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    goal = pd.to_numeric(df["funding_goal_usd"], errors="coerce")
    raised = pd.to_numeric(df["funding_raised_usd"], errors="coerce")
    ratio = pd.Series(np.nan, index=df.index, dtype=float)
    valid = np.isfinite(goal) & (goal != 0) & np.isfinite(raised)
    ratio.loc[valid] = raised[valid] / goal[valid]
    df["funding_ratio"] = ratio
    ratio_finite = ratio[np.isfinite(ratio)]
    if not ratio_finite.empty:
        q_low, q_high = ratio_finite.quantile([0.01, 0.99])
        df["funding_ratio_w"] = ratio.clip(lower=q_low, upper=q_high)
    else:
        df["funding_ratio_w"] = np.nan
    return df


def _label_stats(values: List[float]) -> Dict[str, Any]:
    arr = np.array(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p99": float(np.percentile(finite, 99)),
    }


def _build_sample_labels(
    df: pd.DataFrame,
    *,
    label_horizon: int,
    label_goal_min: float,
    split_seed: int,
    seq_len: int,
    min_label_delta_days: float,
    min_ratio_delta_abs: float,
    min_ratio_delta_rel: float,
    strict_future: bool,
    edgar_valid_mask: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    df = df[df["funding_goal_usd"] >= label_goal_min].copy()
    df = df.sort_values(["entity_id", "snapshot_ts"], kind="stable")
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)

    entities = df["entity_id"].dropna().unique().tolist()
    rng = np.random.RandomState(split_seed)
    rng.shuffle(entities)
    train_ids, val_ids, test_ids = _split_entities(entities, split_seed)

    static_ratio_tol = 1e-6
    drop_counts = {
        "dropped_due_to_insufficient_future": 0,
        "dropped_due_to_static_ratio": 0,
        "dropped_due_to_min_delta_days": 0,
        "dropped_due_to_small_ratio_delta_abs": 0,
        "dropped_due_to_small_ratio_delta_rel": 0,
        "dropped_due_to_nonfinite_ratio": 0,
        "dropped_due_to_nonfinite_label": 0,
    }

    labels: Dict[str, List[float]] = {"train": [], "val": [], "test": []}
    edgar_split_valid: Dict[str, List[bool]] = {"train": [], "val": [], "test": []}

    for entity_id, group in df.groupby("entity_id", sort=False):
        group = group.reset_index()
        if len(group) == 0:
            continue
        for input_end in range(len(group)):
            label_idx = input_end + label_horizon
            if label_idx >= len(group):
                drop_counts["dropped_due_to_insufficient_future"] += 1
                if strict_future:
                    continue
                label_idx = len(group) - 1

            window_start = max(0, input_end - seq_len + 1)
            if label_idx < window_start:
                drop_counts["dropped_due_to_insufficient_future"] += 1
                continue

            input_raised = _safe_float(group["funding_raised_usd"].iloc[input_end])
            input_goal = _safe_float(group["funding_goal_usd"].iloc[input_end])
            label_raised = _safe_float(group["funding_raised_usd"].iloc[label_idx])
            label_goal = _safe_float(group["funding_goal_usd"].iloc[label_idx])

            input_ratio = np.nan
            label_ratio = np.nan
            if np.isfinite(input_goal) and input_goal != 0.0 and np.isfinite(input_raised):
                input_ratio = input_raised / input_goal
            if np.isfinite(label_goal) and label_goal != 0.0 and np.isfinite(label_raised):
                label_ratio = label_raised / label_goal
            if not (np.isfinite(input_ratio) and np.isfinite(label_ratio)):
                drop_counts["dropped_due_to_nonfinite_ratio"] += 1
                continue

            if abs(label_ratio - input_ratio) < static_ratio_tol:
                drop_counts["dropped_due_to_static_ratio"] += 1
                continue

            input_end_ts = group["snapshot_ts"].iloc[input_end]
            label_ts = group["snapshot_ts"].iloc[label_idx]
            delta_days = None
            if pd.notna(input_end_ts) and pd.notna(label_ts):
                delta_days = (label_ts - input_end_ts).total_seconds() / 86400.0
            if delta_days is not None and min_label_delta_days > 0 and delta_days < min_label_delta_days:
                drop_counts["dropped_due_to_min_delta_days"] += 1
                continue

            delta_abs = abs(label_ratio - input_ratio)
            delta_rel = delta_abs / max(1.0, abs(input_ratio))
            if min_ratio_delta_abs > 0 and delta_abs < min_ratio_delta_abs:
                drop_counts["dropped_due_to_small_ratio_delta_abs"] += 1
                continue
            if min_ratio_delta_rel > 0 and delta_rel < min_ratio_delta_rel:
                drop_counts["dropped_due_to_small_ratio_delta_rel"] += 1
                continue

            y_value = _safe_float(group["funding_ratio_w"].iloc[label_idx])
            if not np.isfinite(y_value):
                drop_counts["dropped_due_to_nonfinite_label"] += 1
                continue

            split = "train" if entity_id in train_ids else "val" if entity_id in val_ids else "test"
            labels[split].append(float(y_value))
            if edgar_valid_mask is not None:
                row_idx = int(group["index"].iloc[label_idx])
                edgar_split_valid[split].append(bool(edgar_valid_mask.iloc[row_idx]))

    label_stats = {k: _label_stats(v) for k, v in labels.items()}
    split_counts = {k: int(len(v)) for k, v in labels.items()}

    edgar_split_rates = None
    if edgar_valid_mask is not None:
        edgar_split_rates = {}
        for split, vals in edgar_split_valid.items():
            if not vals:
                edgar_split_rates[split] = None
            else:
                edgar_split_rates[split] = float(np.mean(vals))

    return {
        "label_stats": label_stats,
        "split_counts": split_counts,
        "drop_counts": drop_counts,
        "edgar_split_valid_rate": edgar_split_rates,
    }


def _load_manifest(orch_dir: Path) -> Dict[str, Any]:
    manifest_path = orch_dir / "MANIFEST_full.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing MANIFEST_full.json: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _read_status_pre(bench_dir: Path) -> Optional[Dict[str, str]]:
    status_path = bench_dir / "STATUS_PRE.txt"
    if not status_path.exists():
        return None
    data: Dict[str, str] = {}
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        data[key.strip()] = val.strip()
    return data


def _fail(orch_dir: Path, message: str, command: str) -> None:
    failure_path = orch_dir / "FAILURE.md"
    failure_path.write_text(
        f"# FAILURE\n\n{message}\n\nReproduce:\n```\n{command}\n```\n", encoding="utf-8"
    )
    raise SystemExit(message)


def _compare_configs(configs: Dict[str, Dict[str, Any]], allowed_keys: List[str]) -> List[Dict[str, Any]]:
    diffs: List[Dict[str, Any]] = []
    exp_names = sorted(configs.keys())
    if not exp_names:
        return diffs
    base = configs[exp_names[0]]

    def _normalize(key: str, value: Any) -> Any:
        if key in {"models", "fusion_types", "module_variants", "seeds"} and isinstance(value, list):
            return list(value)
        return value

    for key in allowed_keys:
        base_val = _normalize(key, base.get(key))
        for exp_name in exp_names[1:]:
            other_val = _normalize(key, configs[exp_name].get(key))
            if base_val != other_val:
                diffs.append(
                    {
                        "key": key,
                        "base_exp_name": exp_names[0],
                        "base_value": base_val,
                        "other_exp_name": exp_name,
                        "other_value": other_val,
                    }
                )
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit run comparability for a benchmark stamp.")
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--bench_list", required=True)
    parser.add_argument("--artifacts", required=False, default=None)
    args = parser.parse_args()

    root = Path("/home/pni/project/repo_root")
    stamp = args.stamp
    orch_dir = root / "runs" / "orchestrator" / stamp
    bench_list = Path(args.bench_list)
    if not bench_list.exists():
        _fail(orch_dir, f"missing bench list: {bench_list}", f"cat {bench_list}")

    bench_dirs = [Path(p) for p in bench_list.read_text(encoding="utf-8").splitlines() if p.strip()]
    if len(bench_dirs) != 12:
        _fail(orch_dir, f"bench_dirs_all.txt count={len(bench_dirs)} (expected 12)", f"cat {bench_list}")

    manifest = _load_manifest(orch_dir)
    offers_core = manifest.get("offers_core")
    offers_core_sha = manifest.get("offers_core_sha256")
    selection_hash = manifest.get("selection_hash")
    expected_edgar_col_hash = None
    status_pre_list = manifest.get("status_pre") or []
    if status_pre_list:
        expected_edgar_col_hash = status_pre_list[0].get("col_hash")
    edgar_manifest = manifest.get("edgar_manifest")
    if not offers_core or not offers_core_sha or not selection_hash or not edgar_manifest:
        _fail(
            orch_dir,
            "MANIFEST_full.json missing required keys (offers_core/offers_core_sha256/selection_hash/edgar_manifest)",
            f"cat {orch_dir/'MANIFEST_full.json'}",
        )

    analysis_dir = orch_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_json = analysis_dir / "comparability_report.json"
    report_md = analysis_dir / "comparability_report.md"

    # Load configs and metrics
    configs: Dict[str, Dict[str, Any]] = {}
    metrics_by_exp: Dict[str, Dict[str, Any]] = {}
    status_by_exp: Dict[str, Dict[str, Any]] = {}
    status_pre_by_exp: Dict[str, Optional[Dict[str, str]]] = {}
    bench_by_exp: Dict[str, str] = {}
    factor_rows: List[Dict[str, Any]] = []

    for bench_dir in bench_dirs:
        metrics_path = bench_dir / "metrics.json"
        config_path = bench_dir / "configs" / "resolved_config.yaml"
        status_path = bench_dir / "STATUS_RUN.json"
        selection_hash_path = bench_dir / "selection_hash.txt"
        if not metrics_path.exists():
            _fail(orch_dir, f"missing metrics.json: {metrics_path}", f"ls -lah {bench_dir}")
        if not config_path.exists():
            _fail(orch_dir, f"missing resolved_config.yaml: {config_path}", f"ls -lah {bench_dir}")
        if not status_path.exists():
            _fail(orch_dir, f"missing STATUS_RUN.json: {status_path}", f"ls -lah {bench_dir}")
        if not selection_hash_path.exists():
            _fail(orch_dir, f"missing selection_hash.txt: {selection_hash_path}", f"ls -lah {bench_dir}")
        bench_selection_hash = selection_hash_path.read_text(encoding="utf-8").strip()
        if bench_selection_hash != selection_hash:
            _fail(
                orch_dir,
                f"selection_hash mismatch in {bench_dir.name}: {bench_selection_hash} != {selection_hash}",
                f"cat {selection_hash_path}",
            )

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        status = json.loads(status_path.read_text(encoding="utf-8"))

        exp_name = metrics.get("exp_name") or config.get("exp_name")
        if not exp_name:
            _fail(orch_dir, f"missing exp_name in {metrics_path}", f"head -n 5 {metrics_path}")
        configs[exp_name] = config
        metrics_by_exp[exp_name] = metrics
        status_by_exp[exp_name] = status
        status_pre_by_exp[exp_name] = _read_status_pre(bench_dir)
        bench_by_exp[exp_name] = str(bench_dir)

        factor_rows.append(
            {
                "exp_name": exp_name,
                "label_horizon": int(config.get("label_horizon")),
                "strict_future": bool(config.get("strict_future")),
                "use_edgar": bool(config.get("use_edgar")),
            }
        )

    # Factor coverage check
    combos = {(r["label_horizon"], int(r["strict_future"]), int(r["use_edgar"])) for r in factor_rows}
    expected = {(h, sf, ue) for h in [14, 30, 45] for sf in [0, 1] for ue in [0, 1]}
    missing = sorted(expected - combos)
    extra = sorted(combos - expected)
    if missing or extra:
        report = {
            "stamp": stamp,
            "factor_coverage": {"expected": sorted(expected), "observed": sorted(combos)},
            "missing_combos": missing,
            "extra_combos": extra,
        }
        report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report_md.write_text(
            f"# Comparability Report (stamp={stamp})\n\n"
            f"Missing combos: {missing}\nExtra combos: {extra}\n",
            encoding="utf-8",
        )
        _fail(
            orch_dir,
            f"factor coverage mismatch missing={missing} extra={extra}",
            f"python scripts/audit_run_comparability.py --stamp {stamp} --bench_list {bench_list} --artifacts {args.artifacts or ''}",
        )

    # Config consistency
    compare_keys = [
        "offers_core",
        "offers_static",
        "selected_entities_json",
        "selected_entities_hash",
        "plan",
        "strict_matrix",
        "models",
        "fusion_types",
        "module_variants",
        "seeds",
        "sample_strategy",
        "sample_seed",
        "split_seed",
        "label_goal_min",
        "seq_len",
        "pred_len",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "min_label_delta_days",
        "min_ratio_delta_abs",
        "min_ratio_delta_rel",
    ]
    diffs = _compare_configs(configs, compare_keys)

    # Preload offers_core + selection
    first_cfg = configs[sorted(configs.keys())[0]]
    selected_json = first_cfg.get("selected_entities_json")
    if not selected_json:
        _fail(orch_dir, "selected_entities_json missing in resolved_config.yaml", f"cat {bench_list}")
    selected_path = root / selected_json
    if not selected_path.exists():
        _fail(orch_dir, f"missing selected_entities_json: {selected_path}", f"ls -lah {selected_path.parent}")
    selected_entities = json.loads(selected_path.read_text(encoding="utf-8"))
    selected_set = set(str(e) for e in selected_entities)

    offers_path = root / offers_core
    if not offers_path.exists():
        _fail(orch_dir, f"missing offers_core: {offers_path}", f"ls -lah {offers_path.parent}")

    df_base = pd.read_parquet(offers_path)
    if "entity_id" not in df_base.columns:
        _fail(orch_dir, "offers_core missing entity_id column", f"python scripts/run_full_benchmark.py --help")
    df_base = df_base[df_base["entity_id"].astype(str).isin(selected_set)].copy()
    df_base = _compute_ratio_w(df_base)

    edgar_context = None
    edgar_feature_cols: List[str] = []
    edgar_valid_mask = None
    if any(bool(cfg.get("use_edgar")) for cfg in configs.values()):
        edgar_cfg = next(cfg for cfg in configs.values() if bool(cfg.get("use_edgar")))
        edgar_path = edgar_cfg.get("edgar_features")
        if not edgar_path:
            _fail(orch_dir, "use_edgar=1 but edgar_features missing in resolved_config", f"cat {bench_list}")
        edgar_full_path = root / str(edgar_path)
        if not edgar_full_path.exists():
            _fail(orch_dir, f"missing edgar_features: {edgar_full_path}", f"ls -lah {edgar_full_path.parent}")
        edgar_df = pd.read_parquet(edgar_full_path)
        join_keys = ["offer_id", "cutoff_ts"]
        missing_keys = [k for k in join_keys if k not in df_base.columns or k not in edgar_df.columns]
        if missing_keys:
            _fail(orch_dir, f"edgar join missing keys: {missing_keys}", f"python scripts/run_full_benchmark.py --help")
        if "edgar_filed_date" in edgar_df.columns:
            edgar_df = edgar_df.sort_values("edgar_filed_date")
        edgar_df = edgar_df.drop_duplicates(subset=join_keys, keep="last")
        df_edgar = df_base.merge(edgar_df, on=join_keys, how="left", validate="many_to_one")
        numeric_cols = [
            c for c in edgar_df.columns if c not in join_keys and pd.api.types.is_numeric_dtype(edgar_df[c])
        ]
        edgar_feature_cols = [c for c in numeric_cols if c in df_edgar.columns]
        if not edgar_feature_cols:
            _fail(orch_dir, "No numeric edgar feature columns after join", f"python scripts/run_full_benchmark.py --help")
        edgar_valid_mask = df_edgar[edgar_feature_cols].notna().any(axis=1)
        edgar_context = {"df": df_edgar, "edgar_valid_mask": edgar_valid_mask}

    # Per-run stats
    runs: List[Dict[str, Any]] = []
    for exp_name, cfg in configs.items():
        use_edgar = bool(cfg.get("use_edgar"))
        label_horizon = int(cfg.get("label_horizon"))
        strict_future = bool(cfg.get("strict_future"))
        seq_len = int(cfg.get("seq_len"))
        label_goal_min = float(cfg.get("label_goal_min"))
        min_label_delta_days = float(cfg.get("min_label_delta_days", 0.0))
        min_ratio_delta_abs = float(cfg.get("min_ratio_delta_abs", 0.0))
        min_ratio_delta_rel = float(cfg.get("min_ratio_delta_rel", 0.0))
        split_seed = int(cfg.get("split_seed", 42))

        df_use = edgar_context["df"] if use_edgar else df_base
        ev_mask = edgar_context["edgar_valid_mask"] if use_edgar else None
        stats = _build_sample_labels(
            df_use,
            label_horizon=label_horizon,
            label_goal_min=label_goal_min,
            split_seed=split_seed,
            seq_len=seq_len,
            min_label_delta_days=min_label_delta_days,
            min_ratio_delta_abs=min_ratio_delta_abs,
            min_ratio_delta_rel=min_ratio_delta_rel,
            strict_future=strict_future,
            edgar_valid_mask=ev_mask,
        )

        metrics = metrics_by_exp[exp_name]
        results = metrics.get("results", []) or []
        rmse_values: List[float] = []
        for row in results:
            try:
                value = float(row.get("rmse"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                rmse_values.append(value)
        best_rmse = min(rmse_values) if rmse_values else None

        best_val_values: List[float] = []
        for row in results:
            try:
                value = float(row.get("best_val_loss"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                best_val_values.append(value)
        best_val = min(best_val_values) if best_val_values else None
        results_count = int(metrics.get("results_count", len(results)))

        status_pre = status_pre_by_exp.get(exp_name)
        edgar_col_hash = status_pre.get("col_hash") if status_pre else None
        edgar_col_hash_match = None
        if use_edgar and expected_edgar_col_hash and edgar_col_hash:
            edgar_col_hash_match = bool(edgar_col_hash == expected_edgar_col_hash)
        if use_edgar and expected_edgar_col_hash and edgar_col_hash is None:
            edgar_col_hash_match = False

        edgar_join_rate = status_by_exp[exp_name].get("edgar_join_valid_rate") if use_edgar else None
        edgar_join_rate = float(edgar_join_rate) if edgar_join_rate is not None else None
        edgar_overall_rate = float(ev_mask.mean()) if use_edgar and ev_mask is not None else None

        runs.append(
            {
                "exp_name": exp_name,
                "bench_dir": bench_by_exp[exp_name],
                "label_horizon": label_horizon,
                "strict_future": strict_future,
                "use_edgar": use_edgar,
                "offers_core": cfg.get("offers_core"),
                "offers_core_sha256": offers_core_sha,
                "selection_hash": selection_hash,
                "edgar_col_hash": edgar_col_hash,
                "edgar_col_hash_match": edgar_col_hash_match,
                "results_count": results_count,
                "best_rmse": best_rmse,
                "best_val_loss": best_val,
                "rmse_over_test_std": (
                    float(best_rmse) / float(stats["label_stats"]["test"]["std"])
                    if best_rmse is not None and stats["label_stats"]["test"]["std"]
                    else None
                ),
                "integrity_sources": {
                    "offers_core_sha256": "MANIFEST_full.json",
                    "selection_hash": "MANIFEST_full.json (validated vs bench selection_hash.txt)",
                    "edgar_col_hash": "STATUS_PRE.txt" if use_edgar else None,
                },
                "label_stats": stats["label_stats"],
                "split_counts": stats["split_counts"],
                "drop_counts": stats["drop_counts"],
                "edgar_join_valid_rate": edgar_join_rate,
                "edgar_overall_valid_rate": edgar_overall_rate,
                "edgar_split_valid_rate": stats["edgar_split_valid_rate"],
            }
        )

    # Metrics validity
    metric_errors = []
    for run in runs:
        if run["results_count"] < 10:
            metric_errors.append({"exp_name": run["exp_name"], "issue": "results_count<10"})
        if run["best_rmse"] is None or not np.isfinite(run["best_rmse"]):
            metric_errors.append({"exp_name": run["exp_name"], "issue": "nonfinite_rmse"})
        if run["best_val_loss"] is None or not np.isfinite(run["best_val_loss"]):
            metric_errors.append({"exp_name": run["exp_name"], "issue": "nonfinite_best_val"})
        if run["use_edgar"] and run["edgar_col_hash_match"] is False:
            metric_errors.append({"exp_name": run["exp_name"], "issue": "edgar_col_hash_mismatch"})

    # Anomaly analysis for RMSE scale
    rmse_scale_notes = []
    test_p99s = [r["label_stats"]["test"]["p99"] for r in runs if r["label_stats"]["test"]["p99"] is not None]
    median_p99 = float(np.median(test_p99s)) if test_p99s else None
    for run in runs:
        test_stats = run["label_stats"]["test"]
        if run["best_rmse"] is None or test_stats["p99"] is None or median_p99 is None:
            continue
        if test_stats["p99"] < median_p99 / 10:
            rmse_scale_notes.append(
                {
                    "exp_name": run["exp_name"],
                    "best_rmse": run["best_rmse"],
                    "test_p99": test_stats["p99"],
                    "median_test_p99": median_p99,
                    "note": "label scale (test p99) is much smaller than median",
                }
            )

    # Pairwise label consistency (edgar on/off with same horizon/strict_future)
    by_factor: Dict[Tuple[int, bool], Dict[str, Dict[str, Any]]] = {}
    for run in runs:
        key = (run["label_horizon"], run["strict_future"])
        by_factor.setdefault(key, {})["edgar_on" if run["use_edgar"] else "edgar_off"] = run

    pairwise_label_diffs: List[Dict[str, Any]] = []
    for (horizon, strict_future), pair in sorted(by_factor.items()):
        if "edgar_on" not in pair or "edgar_off" not in pair:
            continue
        on_run = pair["edgar_on"]
        off_run = pair["edgar_off"]
        pair_diffs = {}
        for split in ["train", "val", "test"]:
            on_stats = on_run["label_stats"][split]
            off_stats = off_run["label_stats"][split]
            pair_diffs[split] = {
                "mean_diff": _safe_float(on_stats["mean"]) - _safe_float(off_stats["mean"]),
                "std_diff": _safe_float(on_stats["std"]) - _safe_float(off_stats["std"]),
                "p99_diff": _safe_float(on_stats["p99"]) - _safe_float(off_stats["p99"]),
                "count_diff": int(on_run["split_counts"][split]) - int(off_run["split_counts"][split]),
            }
        pairwise_label_diffs.append(
            {
                "label_horizon": horizon,
                "strict_future": strict_future,
                "edgar_on": on_run["exp_name"],
                "edgar_off": off_run["exp_name"],
                "diffs": pair_diffs,
            }
        )

    report = {
        "stamp": stamp,
        "bench_count": len(runs),
        "factor_coverage": {"expected": sorted(expected), "observed": sorted(combos)},
        "config_diffs": diffs,
        "metric_errors": metric_errors,
        "runs": runs,
        "rmse_scale_notes": rmse_scale_notes,
        "pairwise_label_diffs": pairwise_label_diffs,
        "artifacts": {
            "offers_core": offers_core,
            "offers_core_sha256": offers_core_sha,
            "selection_hash": selection_hash,
            "edgar_manifest": edgar_manifest,
        },
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown report
    lines = [f"# Comparability Report (stamp={stamp})", ""]
    lines.append("## Coverage")
    lines.append(f"- expected combos: {sorted(expected)}")
    lines.append(f"- observed combos: {sorted(combos)}")
    lines.append("")

    if diffs:
        lines.append("## Config Diffs (FAIL)")
        for diff in diffs:
            lines.append(
                f"- {diff['key']}: {diff['base_exp_name']}={diff['base_value']} vs {diff['other_exp_name']}={diff['other_value']}"
            )
        lines.append("")
    else:
        lines.append("## Config Consistency")
        lines.append("- No diffs found across non-factor hyperparameters.")
        lines.append("")

    lines.append("## Metrics Summary (best RMSE)")
    lines.append("| exp_name | use_edgar | horizon | strict_future | best_rmse | best_val_loss | results |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for run in runs:
        lines.append(
            f"| {run['exp_name']} | {run['use_edgar']} | {run['label_horizon']} | {run['strict_future']} | "
            f"{run['best_rmse']} | {run['best_val_loss']} | {run['results_count']} |"
        )
    lines.append("")

    lines.append("## RMSE vs Label Std (test)")
    lines.append("| exp_name | best_rmse | test_std | rmse_over_test_std |")
    lines.append("| --- | --- | --- | --- |")
    for run in runs:
        test_std = run["label_stats"]["test"]["std"]
        lines.append(
            f"| {run['exp_name']} | {run['best_rmse']} | {test_std} | {run['rmse_over_test_std']} |"
        )
    lines.append("")

    lines.append("## Label Stats (train/val/test)")
    lines.append("| exp_name | split | count | mean | std | min | p50 | p90 | p99 | max |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for run in runs:
        for split in ["train", "val", "test"]:
            stats = run["label_stats"][split]
            lines.append(
                f"| {run['exp_name']} | {split} | {stats['count']} | {stats['mean']} | {stats['std']} | "
                f"{stats['min']} | {stats['p50']} | {stats['p90']} | {stats['p99']} | {stats['max']} |"
            )
    lines.append("")

    lines.append("## EDGAR Join Health (use_edgar=1)")
    lines.append("| exp_name | overall_valid_rate | status_join_valid_rate | split_train | split_val | split_test | implicit_filter |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for run in runs:
        if not run["use_edgar"]:
            continue
        split_rates = run.get("edgar_split_valid_rate") or {}
        overall = run.get("edgar_overall_valid_rate")
        status_rate = run.get("edgar_join_valid_rate")
        implicit_filter = None
        if overall is not None and split_rates.get("train") is not None:
            implicit_filter = bool(split_rates["train"] == 1.0 and overall < 1.0)
        lines.append(
            f"| {run['exp_name']} | {overall} | {status_rate} | {split_rates.get('train')} | "
            f"{split_rates.get('val')} | {split_rates.get('test')} | {implicit_filter} |"
        )
    lines.append("")

    lines.append("## RMSE Scale Notes")
    if rmse_scale_notes:
        for note in rmse_scale_notes:
            lines.append(
                f"- {note['exp_name']}: rmse={note['best_rmse']} vs test_p99={note['test_p99']} "
                f"(median_p99={note['median_test_p99']}) -> {note['note']}"
            )
    else:
        lines.append("- No label-scale outliers detected by test p99 heuristic.")
    lines.append("")

    lines.append("## Pairwise Label Consistency (edgar on/off)")
    if pairwise_label_diffs:
        for entry in pairwise_label_diffs:
            lines.append(
                f"- horizon={entry['label_horizon']} strict_future={entry['strict_future']} "
                f"edgar_on={entry['edgar_on']} edgar_off={entry['edgar_off']}"
            )
            for split, diff in entry["diffs"].items():
                lines.append(
                    f"  - {split}: mean_diff={diff['mean_diff']} std_diff={diff['std_diff']} "
                    f"p99_diff={diff['p99_diff']} count_diff={diff['count_diff']}"
                )
    else:
        lines.append("- No edgar on/off pairs available.")
    lines.append("")

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if diffs or metric_errors:
        _fail(
            orch_dir,
            f"comparability check failed: diffs={len(diffs)} metric_errors={len(metric_errors)}",
            f"python scripts/audit_run_comparability.py --stamp {stamp} --bench_list {bench_list} --artifacts {args.artifacts or ''}",
        )

    # Update MANIFEST_full.json with analysis hashes
    manifest["analysis_reports"] = {
        "comparability_report_json": str(report_json),
        "comparability_report_json_sha256": _sha256(report_json),
        "comparability_report_md": str(report_md),
        "comparability_report_md_sha256": _sha256(report_md),
    }
    (orch_dir / "MANIFEST_full.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
