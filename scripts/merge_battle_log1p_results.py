#!/usr/bin/env python3
"""Merge split Battle1 + Battle3 results (core_only + core_edgar halves).

Usage:
    python3 scripts/merge_battle_log1p_results.py
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def merge_battle1() -> None:
    co_paths = [
        REPO / "runs/edssm_battle1_log1p_co/battle1_revenge_results.json",
    ]
    ce_paths = [
        REPO / "runs/edssm_battle1_log1p_ce/battle1_revenge_results.json",
        REPO / "runs/edssm_battle1_log1p_ce_hop/battle1_revenge_results.json",
    ]

    co = next((d for p in co_paths if (d := _load(p)) is not None), None)
    ce = next((d for p in ce_paths if (d := _load(p)) is not None), None)

    if co is None and ce is None:
        print("[Battle1] Neither half finished yet.")
        return
    if co is None:
        print("[Battle1] core_only not finished yet.")
    if ce is None:
        print("[Battle1] core_edgar not finished yet.")

    cells: list[dict] = []
    n_pass = n_fail = n_skip = 0
    for part in filter(None, [co, ce]):
        for c in part.get("cells", []):
            cells.append(c)
            s = c.get("status")
            if s == "skipped":
                n_skip += 1
            elif c.get("verdict", {}).get("PASS"):
                n_pass += 1
            else:
                n_fail += 1

    merged = {
        "battle": "1_trunk_ablation_revenge_log1p_MERGED",
        "total": len(cells),
        "pass": n_pass,
        "fail": n_fail,
        "skip": n_skip,
        "pass_rate": f"{n_pass}/{n_pass + n_fail}" if (n_pass + n_fail) > 0 else "N/A",
        "cells": cells,
    }
    out = REPO / "runs/edssm_battle1_log1p_merged.json"
    with open(out, "w") as f:
        json.dump(merged, f, indent=2, default=str)

    print(f"[Battle1] Merged → {out}")
    print(f"  {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP  ({merged['pass_rate']})")
    for c in cells:
        v = c.get("verdict", {})
        sym = "✓" if v.get("PASS") else "✗"
        pct = v.get("edssm_vs_orig_pct")
        pct_s = f"{pct:+.1f}%" if pct is not None else "N/A"
        m = c.get("edssm", {}).get("metrics", {})
        pmean = m.get("pred_mean")
        pmean_s = f"pred_mean={pmean:.2f}" if pmean is not None else ""
        print(f"  {sym} {c['cell']:55s} {pct_s:>10s}  {pmean_s}")


def merge_battle3() -> None:
    co_paths = [
        REPO / "runs/edssm_battle3_log1p_co/battle3_audit_results.json",
    ]
    ce_paths = [
        REPO / "runs/edssm_battle3_log1p_ce/battle3_audit_results.json",
        REPO / "runs/edssm_battle3_log1p_ce_hop/battle3_audit_results.json",
    ]

    co = next((d for p in co_paths if (d := _load(p)) is not None), None)
    ce = next((d for p in ce_paths if (d := _load(p)) is not None), None)

    if co is None and ce is None:
        print("[Battle3] Neither half finished yet.")
        return

    cells: list[dict] = []
    n_pass = n_fail = 0
    for part in filter(None, [co, ce]):
        for c in part.get("cells", []):
            cells.append(c)
            if c.get("verdict", {}).get("PASS"):
                n_pass += 1
            else:
                n_fail += 1

    merged = {
        "battle": "3_ghost_audit_log1p_MERGED",
        "total": len(cells),
        "pass": n_pass,
        "fail": n_fail,
        "pass_rate": f"{n_pass}/{n_pass + n_fail}" if (n_pass + n_fail) > 0 else "N/A",
        "cells": cells,
    }
    out = REPO / "runs/edssm_battle3_log1p_merged.json"
    with open(out, "w") as f:
        json.dump(merged, f, indent=2, default=str)

    print(f"[Battle3] Merged → {out}")
    print(f"  {n_pass} PASS / {n_fail} FAIL  ({merged['pass_rate']})")
    for c in cells:
        v = c.get("verdict", {})
        sym = "✓" if v.get("PASS") else "✗"
        eds = c.get("edssm", {})
        dn = c.get("deepnpts", {})
        print(
            f"  {sym} {c.get('cell','?'):45s}  "
            f"ED-SSM ghost={eds.get('ghost_rate','?'):.4f}  "
            f"DeepNPTS ghost={dn.get('ghost_rate','?'):.4f}  "
            f"pred_investors_mean={eds.get('pred_investors_mean','?')}"
        )


if __name__ == "__main__":
    merge_battle1()
    print()
    merge_battle3()
