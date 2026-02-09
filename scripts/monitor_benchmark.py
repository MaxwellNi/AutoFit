#!/usr/bin/env python3
"""Monitor running benchmark jobs and display progress."""
import json
import glob
import subprocess
import sys
from pathlib import Path
from collections import Counter

BASE = "runs/benchmarks/block3_20260203_225620_iris_full"

def main():
    # 1. SLURM queue status
    print("=" * 70)
    print("SLURM JOB STATUS")
    print("=" * 70)
    try:
        r = subprocess.run(
            ["squeue", "-u", "npin", "--sort=+i",
             "--format=%.10i %.10P %.12j %.4T %.10M %.10L %.R"],
            capture_output=True, text=True, timeout=10,
        )
        if r.stdout.strip():
            print(r.stdout)
        else:
            print("  No active jobs")
    except Exception as e:
        print(f"  squeue error: {e}")

    # 2. Manifest status
    print("=" * 70)
    print("MANIFEST STATUS")
    print("=" * 70)
    manifests = sorted(glob.glob(f"{BASE}/**/MANIFEST.json", recursive=True))
    status_counts = Counter()
    missing = []

    EXPECTED_SHARDS = []
    tasks = ["task1_outcome", "task2_forecast", "task3_risk_adjust"]
    ablations = ["core_only", "core_edgar"]
    categories = [
        "statistical", "deep_classical", "foundation", "irregular",
        "transformer_sota_A", "transformer_sota_B", "transformer_sota_C",
        "ml_tabular", "autofit",
    ]
    for t in tasks:
        for c in categories:
            for a in ablations:
                EXPECTED_SHARDS.append(f"{t}/{c}/{a}")

    for shard in EXPECTED_SHARDS:
        mpath = Path(BASE) / shard / "MANIFEST.json"
        if mpath.exists():
            m = json.loads(mpath.read_text())
            st = m.get("status", "unknown")
            status_counts[st] += 1
            n_run = m.get("n_models_run", "?")
            n_fail = m.get("n_models_failed", "?")
            if st != "completed":
                print(f"  [{st:>10}] {shard}  (run={n_run}, fail={n_fail})")
        else:
            status_counts["missing"] += 1
            missing.append(shard)

    print(f"\nSummary: {dict(status_counts)}")
    if missing:
        print(f"Missing MANIFESTs ({len(missing)}):")
        for m in missing:
            print(f"  {m}")

    # 3. Metric record counts
    print("\n" + "=" * 70)
    print("METRIC RECORDS")
    print("=" * 70)
    all_records = []
    for mf in sorted(glob.glob(f"{BASE}/**/metrics.json", recursive=True)):
        recs = json.load(open(mf))
        all_records.extend(recs)

    models = sorted(set(r["model_name"] for r in all_records))
    print(f"Total records: {len(all_records)}")
    print(f"Unique models ({len(models)}): {models}")

    cat_counts = Counter(r.get("category", "?") for r in all_records)
    for c, n in sorted(cat_counts.items()):
        model_names = sorted(set(
            r["model_name"] for r in all_records if r.get("category") == c
        ))
        print(f"  {c}: {n} records — {model_names}")

    # 4. Latest log activity
    print("\n" + "=" * 70)
    print("LATEST LOG ACTIVITY (last 5 lines of newest .err files)")
    print("=" * 70)
    import os
    err_files = sorted(
        glob.glob(f"{BASE}/**/*.err", recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )[:4]
    for ef in err_files:
        rel = os.path.relpath(ef, BASE)
        lines = open(ef).readlines()[-3:]
        print(f"\n  [{rel}]")
        for l in lines:
            print(f"    {l.rstrip()}")


if __name__ == "__main__":
    main()
