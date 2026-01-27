#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

GOAL_RE = re.compile(r"goal(?P<val>\d+(?:\.\d+)?)", re.IGNORECASE)
HORIZON_RE = re.compile(r"(?:horizon|h)(?P<val>\d+)\b", re.IGNORECASE)


def _parse_goal(text: str | None) -> float | None:
    if not text:
        return None
    match = GOAL_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group("val"))
    except (TypeError, ValueError):
        return None


def _parse_horizon(text: str | None) -> int | None:
    if not text:
        return None
    match = HORIZON_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group("val"))
    except (TypeError, ValueError):
        return None


def _load_config(bench_dir: Path) -> Dict[str, Any] | None:
    cfg_path = bench_dir / "configs" / "resolved_config.yaml"
    if not cfg_path.exists():
        return None
    try:
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit benchmark configs for label_horizon and label_goal_min.")
    parser.add_argument("--bench_dirs", nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--output", dest="output_path", type=Path, default=None)
    parser.add_argument("--expected_horizon", type=int, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_path
    if output_path is None:
        out_dir = Path("runs") / f"sanity_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "audit_summary.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    overall_ok = True
    for raw_dir in args.bench_dirs:
        bench_dir = Path(raw_dir)
        item: Dict[str, Any] = {
            "bench_dir": str(bench_dir),
            "ok": True,
            "errors": [],
            "warnings": [],
        }
        cfg = _load_config(bench_dir)
        if cfg is None:
            item["ok"] = False
            item["errors"].append("missing_or_invalid_config")
            results.append(item)
            overall_ok = False
            continue

        exp_name = cfg.get("exp_name")
        label_goal_min = cfg.get("label_goal_min")
        label_horizon = cfg.get("label_horizon")
        item.update(
            {
                "exp_name": exp_name,
                "label_goal_min": label_goal_min,
                "label_horizon": label_horizon,
                "sample_strategy": cfg.get("sample_strategy"),
                "sample_seed": cfg.get("sample_seed"),
                "split_seed": cfg.get("split_seed"),
                "seeds": cfg.get("seeds"),
                "limit_rows": cfg.get("limit_rows"),
                "use_edgar": cfg.get("use_edgar"),
            }
        )

        expected_goal = _parse_goal(str(exp_name)) or _parse_goal(bench_dir.name)
        expected_horizon_name = _parse_horizon(str(exp_name)) or _parse_horizon(bench_dir.name)
        expected_horizon = args.expected_horizon if args.expected_horizon is not None else expected_horizon_name
        item["expected_goal_min"] = expected_goal
        item["expected_horizon"] = expected_horizon
        if label_horizon is None:
            item["ok"] = False
            item["errors"].append("missing_label_horizon")
        else:
            try:
                int(label_horizon)
            except (TypeError, ValueError):
                item["ok"] = False
                item["errors"].append(f"label_horizon_invalid:{label_horizon}")

        if label_goal_min is None:
            item["ok"] = False
            item["errors"].append("missing_label_goal_min")
        else:
            try:
                goal_val = float(label_goal_min)
                if goal_val not in {50.0, 500.0}:
                    item["ok"] = False
                    item["errors"].append(f"label_goal_min_not_in_grid:{label_goal_min}")
                if expected_goal is not None and abs(goal_val - float(expected_goal)) > 1e-6:
                    item["ok"] = False
                    item["errors"].append("label_goal_min_mismatch_name")
            except (TypeError, ValueError):
                item["ok"] = False
                item["errors"].append(f"label_goal_min_invalid:{label_goal_min}")

        if expected_horizon is not None and label_horizon is not None:
            try:
                if int(label_horizon) != int(expected_horizon):
                    item["ok"] = False
                    item["errors"].append(
                        f"label_horizon_mismatch expected={expected_horizon} got={label_horizon}"
                    )
            except (TypeError, ValueError):
                item["ok"] = False
                item["errors"].append("label_horizon_invalid_compare")

        results.append(item)
        if not item["ok"]:
            overall_ok = False

    summary = {
        "timestamp": timestamp,
        "overall_ok": overall_ok,
        "run_count": len(results),
        "runs": results,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
