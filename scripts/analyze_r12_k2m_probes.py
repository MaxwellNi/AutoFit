"""Round-12 K2+M probe analyser.

Walks the `runs/audits/r12cpu_*` sidecars and the matching
`runs/benchmarks/r12cpu_*/metrics.json` files, prints a concise table:
  * per-probe: baseline/k2m, horizon, lane_residual_blend,
    lane_source_scale_silently_dead, lane_ss_fallback_active, MAE.
  * baseline-vs-k2m delta at matched horizon.
  * horizon-divergence: does MAE differ between h=7 and h=30 under k2m?

This is the Level-0 "did trunk really learn horizon structure" probe
payload described in §0u of .local_mandatory_preexec.md.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  WARN could not read {path}: {e}", file=sys.stderr)
        return None


def _collect() -> list[dict]:
    rows = []
    for audit_dir in sorted(glob.glob(str(REPO / "runs" / "audits" / "r12cpu_*"))):
        name = Path(audit_dir).name  # r12cpu_<acct>_<tag>_h<h>_<ts>
        parts = name.split("_")
        if len(parts) < 5:
            continue
        _, acct, tag, hstr = parts[0], parts[1], parts[2], parts[3]
        horizon = int(hstr.lstrip("h"))
        sidecars = sorted(Path(audit_dir).glob("*.json"))
        if not sidecars:
            continue
        sc = _load(sidecars[-1])
        if not sc:
            continue
        ri = sc.get("routing_info") or {}
        rows.append(
            {
                "audit_dir": audit_dir,
                "acct": acct,
                "tag": tag,
                "horizon": horizon,
                "lane_residual_blend": ri.get("lane_residual_blend"),
                "lane_source_scale_silently_dead": ri.get(
                    "lane_source_scale_silently_dead"
                ),
                "lane_ss_fallback_active": ri.get("lane_ss_fallback_active"),
                "lane_calibration_rows": ri.get("lane_calibration_rows"),
                "lane_anchor_cal_mae": ri.get("lane_anchor_calibration_mae"),
                "lane_guarded_cal_mae": ri.get("lane_guarded_calibration_mae"),
                "lane_source_scaling_enabled": ri.get("lane_source_scaling_enabled"),
            }
        )
    # Attach MAE from matching benchmark output_dir (prefix match, tag+h).
    for r in rows:
        prefix = f"r12cpu_{r['acct']}_{r['tag']}_h{r['horizon']}_"
        bench_dirs = sorted((REPO / "runs" / "benchmarks").glob(f"{prefix}*"))
        matches = [str(d / "metrics.json") for d in bench_dirs if (d / "metrics.json").exists()]
        if matches:
            m = _load(Path(matches[-1]))
            if m is not None:
                if isinstance(m, list):
                    recs = m
                elif isinstance(m, dict) and "records" in m:
                    recs = m["records"]
                else:
                    recs = [m]
                for rec in recs:
                    mae = (rec.get("metrics") or {}).get("mae")
                    if mae is None:
                        mae = rec.get("mae")
                    if mae is not None:
                        r["mae"] = float(mae)
                        break
    return rows


def main() -> int:
    rows = _collect()
    if not rows:
        print("No r12cpu_* audits found yet.")
        return 1
    hdr = [
        "acct", "tag", "h",
        "blend", "silent", "fallback", "mae",
    ]
    widths = [6, 9, 3, 6, 7, 8, 10]
    print("  ".join(f"{h:<{w}}" for h, w in zip(hdr, widths)))
    for r in rows:
        print("  ".join([
            f"{r['acct']:<6}",
            f"{r['tag']:<9}",
            f"{r['horizon']:<3}",
            f"{str(r.get('lane_residual_blend','?'))[:6]:<6}",
            f"{str(r.get('lane_source_scale_silently_dead','?'))[:7]:<7}",
            f"{str(r.get('lane_ss_fallback_active','?'))[:8]:<8}",
            f"{r.get('mae','?')}",
        ]))

    # Pairwise delta.
    by_key = {}
    for r in rows:
        by_key[(r["acct"], r["tag"], r["horizon"])] = r
    print("\n--- baseline-vs-k2m MAE delta (matched acct+h) ---")
    for acct in sorted({r["acct"] for r in rows}):
        for h in sorted({r["horizon"] for r in rows}):
            b = by_key.get((acct, "baseline", h))
            k = by_key.get((acct, "k2m", h))
            if b and k and "mae" in b and "mae" in k:
                d = k["mae"] - b["mae"]
                pct = 100 * d / b["mae"]
                print(f"  {acct:<6} h={h:<3} baseline={b['mae']:.4f}  k2m={k['mae']:.4f}  Δ={d:+.4f} ({pct:+.2f}%)")

    print("\n--- horizon divergence: does K2+M make h=7 differ from h=30? ---")
    for acct in sorted({r["acct"] for r in rows}):
        for tag in ["baseline", "k2m"]:
            r7 = by_key.get((acct, tag, 7))
            r30 = by_key.get((acct, tag, 30))
            if r7 and r30 and "mae" in r7 and "mae" in r30:
                d = r30["mae"] - r7["mae"]
                print(f"  {acct:<6} {tag:<9} MAE(h=30) − MAE(h=7) = {d:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
