#!/usr/bin/env python3
"""Build offline policy training dataset for AutoFit V7.2 routing/HPO."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRUTH_PACK = ROOT / "docs" / "benchmarks" / "block3_truth_pack"
DEFAULT_OUTPUT = DEFAULT_TRUTH_PACK / "v72_policy_dataset.csv"


def _target_family(target: str) -> str:
    if target == "is_funded":
        return "binary"
    if target == "investors_count":
        return "count"
    return "heavy_tail"


def _horizon_band(h: int) -> str:
    if h in {1, 7}:
        return "short"
    if h == 14:
        return "mid"
    return "long"


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_failure_index(path: Path) -> Dict[Tuple[str, str, str, int], Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[Tuple[str, str, str, int], Dict[str, str]] = {}
    for row in _load_csv(path):
        try:
            key = (
                str(row.get("task", "")),
                str(row.get("ablation", "")),
                str(row.get("target", "")),
                int(float(row.get("horizon", "0"))),
            )
            out[key] = row
        except Exception:
            continue
    return out


def _suggest_action(best_model: str, target_family: str) -> Dict[str, str]:
    if target_family == "count":
        return {
            "template_id": "count_nbeats_nhits_kan",
            "candidate_subset": "NBEATS,NHITS,KAN,LightGBMTweedie,XGBoostPoisson",
            "count_family": "auto",
            "binary_calibration_mode": "na",
            "top_k": "10",
        }
    if target_family == "binary":
        return {
            "template_id": "binary_patchtst_nhits",
            "candidate_subset": "PatchTST,NHITS,TimeMixer,iTransformer",
            "count_family": "na",
            "binary_calibration_mode": "auto",
            "top_k": "8",
        }
    if best_model in {"Chronos", "Moirai", "TimesFM"}:
        template = "heavy_tail_foundation_mix"
        subset = "Chronos,Moirai,TimesFM,NHITS,PatchTST"
    else:
        template = "heavy_tail_deep_mix"
        subset = "NHITS,PatchTST,NBEATS,Chronos"
    return {
        "template_id": template,
        "candidate_subset": subset,
        "count_family": "na",
        "binary_calibration_mode": "na",
        "top_k": "8",
    }


def build_rows(tp_dir: Path) -> List[Dict[str, str]]:
    cond_path = tp_dir / "condition_leaderboard.csv"
    fail_path = tp_dir / "failure_taxonomy.csv"
    if not cond_path.exists():
        raise SystemExit(f"Missing condition_leaderboard.csv: {cond_path}")

    cond_rows = _load_csv(cond_path)
    fail_index = _load_failure_index(fail_path)

    out: List[Dict[str, str]] = []
    for row in cond_rows:
        try:
            task = str(row.get("task", ""))
            ablation = str(row.get("ablation", ""))
            target = str(row.get("target", ""))
            horizon = int(float(row.get("horizon", "0")))
            best_model = str(row.get("best_model", ""))
            autofit_gap = float(row.get("autofit_gap_pct", "0") or 0.0)
        except Exception:
            continue
        if not task or not ablation or not target or horizon <= 0:
            continue

        family = _target_family(target)
        hb = _horizon_band(horizon)
        miss_bucket = "unknown"
        # Lightweight proxy from ablation.
        if ablation == "core_only":
            miss_bucket = "high"
        elif ablation == "core_text":
            miss_bucket = "mid"
        elif ablation == "core_edgar":
            miss_bucket = "mid"
        else:
            miss_bucket = "low"
        route_key = f"lane={family}|hb={hb}|ablation={ablation}|miss={miss_bucket}"

        fail = fail_index.get((task, ablation, target, horizon), {})
        issue = str(fail.get("issue_type", "")).strip()
        guard_penalty = 1.0 if issue else 0.0
        compute_penalty = 0.2 if family in {"count", "binary"} else 0.35
        reward = -autofit_gap - compute_penalty - guard_penalty

        action = _suggest_action(best_model=best_model, target_family=family)
        out.append(
            {
                "task": task,
                "ablation": ablation,
                "target": target,
                "horizon": str(horizon),
                "target_family": family,
                "horizon_band": hb,
                "missingness_bucket": miss_bucket,
                "route_key": route_key,
                "template_id": action["template_id"],
                "candidate_subset": action["candidate_subset"],
                "count_family": action["count_family"],
                "binary_calibration_mode": action["binary_calibration_mode"],
                "top_k": action["top_k"],
                "best_model": best_model,
                "autofit_gap_pct": f"{autofit_gap:.6f}",
                "guard_penalty": f"{guard_penalty:.6f}",
                "compute_penalty": f"{compute_penalty:.6f}",
                "reward": f"{reward:.6f}",
                "failure_issue_type": issue,
                "evidence_path": "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
            }
        )
    return out


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "task",
        "ablation",
        "target",
        "horizon",
        "target_family",
        "horizon_band",
        "missingness_bucket",
        "route_key",
        "template_id",
        "candidate_subset",
        "count_family",
        "binary_calibration_mode",
        "top_k",
        "best_model",
        "autofit_gap_pct",
        "guard_penalty",
        "compute_penalty",
        "reward",
        "failure_issue_type",
        "evidence_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V7.2 offline policy dataset.")
    parser.add_argument("--truth-pack-dir", type=Path, default=DEFAULT_TRUTH_PACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = build_rows(args.truth_pack_dir.resolve())
    _write_csv(args.output.resolve(), rows)
    print(f"Wrote {len(rows)} rows: {args.output.resolve()}")


if __name__ == "__main__":
    main()
