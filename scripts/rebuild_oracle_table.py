#!/usr/bin/env python3
"""
Rebuild ORACLE_TABLE_V735 from ground-truth benchmark data.

Reads all metrics.json files from the Phase 7 results directory, finds the
actual champion model for each of the 48 evaluation conditions, and prints
a Python dict suitable for pasting into nf_adaptive_champion.py.

Usage:
    python3 scripts/rebuild_oracle_table.py
    python3 scripts/rebuild_oracle_table.py --output-format code
    python3 scripts/rebuild_oracle_table.py --metric mae
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Rebuild oracle table from benchmark data")
    parser.add_argument("--runs-dir", default="runs/benchmarks/block3_20260203_225620_phase7")
    parser.add_argument("--metric", default="rmse", choices=["rmse", "mae"])
    parser.add_argument("--output-format", default="code", choices=["code", "json", "csv"])
    parser.add_argument("--top-k", type=int, default=3, help="Number of top models per condition")
    parser.add_argument("--exclude-autofit", action="store_true",
                        help="Exclude AutoFit models from oracle (standalone baselines only)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} does not exist")
        return

    # Collect all records
    records = []
    for mf in sorted(runs_dir.rglob("metrics.json")):
        try:
            data = json.loads(mf.read_text())
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)
        except Exception:
            pass

    # Standardize field names
    for r in records:
        if "target" not in r and "target_col" in r:
            r["target"] = r["target_col"]
        if "ablation" not in r and "ablation_name" in r:
            r["ablation"] = r["ablation_name"]

    # Build per-condition best metric (deduped: keep lowest metric per model per condition)
    cond_metrics = defaultdict(dict)  # (target, horizon, ablation) -> {model: metric}

    for r in records:
        t = r.get("target", "")
        h = r.get("horizon")
        a = r.get("ablation", "")
        m = r.get("model", r.get("model_name", ""))
        metric_val = r.get(args.metric)

        if not all([t, h, a, m]) or metric_val is None:
            continue

        # Optionally exclude AutoFit models
        if args.exclude_autofit and m.startswith("AutoFit"):
            continue

        cond = (t, int(h), a)
        val = float(metric_val)

        if m not in cond_metrics[cond] or val < cond_metrics[cond][m]:
            cond_metrics[cond][m] = val

    # Build oracle table
    oracle = {}
    for cond in sorted(cond_metrics.keys()):
        models_sorted = sorted(cond_metrics[cond].items(), key=lambda x: x[1])
        top_k = models_sorted[:args.top_k]
        oracle[cond] = top_k

    # Output
    if args.output_format == "code":
        print(f"# Auto-generated oracle table from {len(records)} records")
        print(f"# Metric: {args.metric}, Top-K: {args.top_k}")
        print(f"# Conditions: {len(oracle)}")
        print(f"# Exclude AutoFit: {args.exclude_autofit}")
        print()

        # V735-style: exact per-condition, single best
        print("ORACLE_TABLE_V735: Dict[Tuple[str, int, str], str] = {")
        for cond in sorted(oracle.keys()):
            top = oracle[cond]
            target, horizon, ablation = cond
            champion = top[0][0]
            metric_val = top[0][1]
            print(f'    ("{target}", {horizon:>2}, "{ablation}"): '
                  f'"{champion}",  # {args.metric}={metric_val:.6f}')
        print("}")
        print()

        # V734-style: coarse (target_type, horizon, ablation_class), top-3 with ranks
        print("\n# ---- V734-style coarse oracle (for reference) ----")
        print("ORACLE_TABLE_V734: Dict[Tuple[str, int, str], List[Tuple[str, float]]] = {")

        # Group by coarse keys
        coarse = defaultdict(lambda: defaultdict(list))
        for cond, top in oracle.items():
            target, horizon, ablation = cond
            # Target type detection
            if target == "is_funded":
                target_type = "binary"
            elif target == "investors_count":
                target_type = "count"
            elif target == "funding_raised_usd":
                target_type = "heavy_tail"
            else:
                target_type = "general"
            abl_cls = "exogenous" if ablation in ("core_edgar", "full") else "temporal"
            coarse_key = (target_type, horizon, abl_cls)
            for model_name, metric_val in top:
                coarse[coarse_key][model_name].append(metric_val)

        # Average across conditions within each coarse key
        for coarse_key in sorted(coarse.keys()):
            model_avgs = {}
            for model_name, vals in coarse[coarse_key].items():
                model_avgs[model_name] = sum(vals) / len(vals)
            top3 = sorted(model_avgs.items(), key=lambda x: x[1])[:3]

            # Convert to avg rank across conditions
            tt, h, ac = coarse_key
            entries = ", ".join(f'("{m}", {v:.2f})' for m, v in top3)
            print(f'    ("{tt}", {h:>2}, "{ac}"): [{entries}],')

        print("}")

    elif args.output_format == "json":
        out = {}
        for cond, top in oracle.items():
            key_str = f"{cond[0]}|{cond[1]}|{cond[2]}"
            out[key_str] = [{"model": m, args.metric: v} for m, v in top]
        print(json.dumps(out, indent=2))

    elif args.output_format == "csv":
        print(f"target,horizon,ablation,rank,model,{args.metric}")
        for cond in sorted(oracle.keys()):
            for rank, (model, val) in enumerate(oracle[cond], 1):
                print(f"{cond[0]},{cond[1]},{cond[2]},{rank},{model},{val:.6f}")

    # Summary stats
    import sys
    print(f"\n# Summary: {len(oracle)} conditions, "
          f"{len(set(m for top in oracle.values() for m, _ in top))} unique champion models",
          file=sys.stderr)


if __name__ == "__main__":
    main()
