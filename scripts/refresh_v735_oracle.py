#!/usr/bin/env python3
"""
Refresh V735 oracle table from latest benchmark results.

Run this AFTER all baseline re-runs complete (core_edgar + full).
It reads all metrics.json files, determines the per-condition winner,
and emits updated ORACLE_TABLE_V735 entries.

Usage:
    python3 scripts/refresh_v735_oracle.py [--apply]
    
    Without --apply: prints the new oracle (dry run)
    With --apply: writes the updated oracle to nf_adaptive_champion.py
"""
import json
import os
import sys
import glob
import re
from collections import defaultdict

RESULTS_DIRS = [
    "/mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7",
    "/mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_iris_full",
]
SOURCE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "narrative", "block3", "models", "nf_adaptive_champion.py"
)

# Models that are AutoFit wrappers (we need to dereference to their inner model)
AUTOFIT_PREFIXES = {"AutoFitV", "FusedChampion", "NFAdaptive"}


def is_autofit_model(name: str) -> bool:
    """Check if model name is an AutoFit wrapper."""
    return any(name.startswith(p) for p in AUTOFIT_PREFIXES)


def load_all_metrics():
    """Load all metrics.json files from all result directories."""
    records = []
    seen_files = set()
    for results_dir in RESULTS_DIRS:
        json_files = sorted(glob.glob(
            os.path.join(results_dir, '**', 'metrics.json'), recursive=True
        ))
        for f in json_files:
            real = os.path.realpath(f)
            if real in seen_files:
                continue
            seen_files.add(real)
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    for r in data:
                        if 'model_name' in r and 'model' not in r:
                            r['model'] = r['model_name']
                        records.append(r)
            except Exception:
                pass
    return records


def compute_oracle(records):
    """Compute per-condition oracle from records.
    
    Returns dict: (target, horizon, ablation) -> (winner_model, winner_mae, top3)
    Only considers non-AutoFit models as winners (to avoid circular reference).
    """
    # Group by condition
    conditions = defaultdict(list)
    for r in records:
        tgt = r.get('target', '')
        hz = r.get('horizon', 0)
        abl = r.get('ablation', '')
        mae = r.get('mae')
        model = r.get('model', r.get('model_name', ''))
        if tgt and hz and abl and mae is not None and model:
            conditions[(tgt, int(hz), abl)].append((model, float(mae)))
    
    oracle = {}
    for key, entries in sorted(conditions.items()):
        # Sort by MAE, filter out AutoFit models for oracle selection
        entries.sort(key=lambda x: x[1])
        non_autofit = [(m, mae) for m, mae in entries if not is_autofit_model(m)]
        if non_autofit:
            winner = non_autofit[0]
            top3 = [m for m, _ in non_autofit[:3]]
        else:
            winner = entries[0]
            top3 = [m for m, _ in entries[:3]]
        oracle[key] = {
            'winner': winner[0],
            'winner_mae': winner[1],
            'top3': top3,
            'total_models': len(entries),
        }
    return oracle


def format_oracle_table(oracle):
    """Format oracle as Python source code."""
    lines = ["ORACLE_TABLE_V735: Dict[Tuple[str, int, str], str] = {"]
    
    # Group by target for readability
    by_target = defaultdict(list)
    for (tgt, hz, abl), info in sorted(oracle.items()):
        by_target[tgt].append((hz, abl, info))
    
    for tgt in sorted(by_target.keys()):
        lines.append(f"    # ── {tgt} ──")
        for hz, abl, info in sorted(by_target[tgt]):
            winner = info['winner']
            mae = info['winner_mae']
            lines.append(
                f'    ("{tgt}", {hz:2d}, "{abl}"):'
                f'{" " * max(1, 20 - len(abl))}'
                f'"{winner}",  # mae={mae:.6f}'
            )
    lines.append("}")
    return "\n".join(lines)


def apply_oracle_to_source(oracle_source: str, source_file: str):
    """Replace ORACLE_TABLE_V735 in source file with new oracle."""
    with open(source_file, 'r') as f:
        content = f.read()
    
    # Find and replace the oracle table
    pattern = r'ORACLE_TABLE_V735:.*?\n\}'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("ERROR: Could not find ORACLE_TABLE_V735 in source file")
        return False
    
    new_content = content[:match.start()] + oracle_source + content[match.end():]
    with open(source_file, 'w') as f:
        f.write(new_content)
    print(f"Updated {source_file}")
    return True


def parse_current_oracle(source_file: str):
    """Parse current ORACLE_TABLE_V735 from source file without importing."""
    with open(source_file, 'r') as f:
        content = f.read()
    pattern = r'ORACLE_TABLE_V735[^{]*\{([^}]+)\}'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return {}
    oracle = {}
    entry_pattern = r'\("([^"]+)",\s*(\d+),\s*"([^"]+)"\)\s*:\s*"([^"]+)"'
    for m in re.finditer(entry_pattern, match.group(1)):
        target, horizon, ablation, model = m.groups()
        oracle[(target, int(horizon), ablation)] = model
    return oracle


def main():
    apply = "--apply" in sys.argv
    
    print("Loading benchmark results...")
    records = load_all_metrics()
    print(f"  Total records: {len(records)}")
    
    print("\nComputing per-condition oracle...")
    oracle = compute_oracle(records)
    print(f"  Conditions: {len(oracle)}")
    
    # Show changes vs current oracle
    current_oracle = parse_current_oracle(SOURCE_FILE)
    
    changes = 0
    for key, info in sorted(oracle.items()):
        current = current_oracle.get(key)
        new_winner = info['winner']
        if current != new_winner:
            changes += 1
            print(f"  CHANGE: {key}: {current} → {new_winner} (mae={info['winner_mae']:.6f})")
    
    if changes == 0:
        print("  No changes needed — oracle is already optimal!")
    else:
        print(f"\n  {changes} oracle entries need updating")
    
    # Format and optionally apply
    oracle_source = format_oracle_table(oracle)
    
    if apply:
        print(f"\nApplying oracle to {SOURCE_FILE}...")
        success = apply_oracle_to_source(oracle_source, SOURCE_FILE)
        if success:
            print("Done! Run 'git add ... && git commit' to save changes.")
    else:
        print("\n=== NEW ORACLE TABLE (dry run) ===")
        print(oracle_source)
        print("\nRun with --apply to update source file.")


if __name__ == "__main__":
    main()
