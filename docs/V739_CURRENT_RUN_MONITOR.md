# V739 Current Run Monitor and Landing Checklist

> Last verified: 2026-03-15
> Scope: current AutoFit V739 execution reality on the canonical clean benchmark line.

## Current Verified Reality — V739 FULLY LANDED ✅

1. **V739 is COMPLETE: 112/112 conditions landed** in canonical `runs/benchmarks/block3_phase9_fair/`.
2. All 12 V739 jobs completed successfully:
   - 8 on l40s partition (npin, iris-snt QOS): t1_co, t1_ce, t2_co, t2_ce, t3_co, t3_ce, t1_ct, t2_ct
   - 4 on gpu partition (cfisch): t3_ct, t1_fu, t2_fu, t3_fu
3. V739 metrics.json files verified under:
   - `runs/benchmarks/block3_phase9_fair/task1_outcome/autofit/core_only/metrics.json` (etc.)
   - 12 files total: 3 tasks × 4 ablations

## V739 Quality Audit

| Metric | Value |
| --- | --- |
| Total records | 112 |
| Ablation coverage | core_only=28, core_edgar=28, core_text=28, full=28 |
| Task coverage | task1=48, task2=32, task3=32 |
| Horizons | [1, 7, 14, 30] |
| Targets (task1) | funding_raised_usd, investors_count, is_funded |
| Targets (task2/3) | funding_raised_usd, investors_count |
| NaN/Inf in MAE | 0 |
| NaN/Inf in RMSE | 0 |
| Fairness pass | 112/112 (100%) |
| Fallback fraction | 0.0 (no fallback used) |
| MAE range | [0.034914, 388219.51] |
| RMSE range | [0.152572, 1742470.54] |

## V739 Benchmark Ranking

**Per-condition mean rank (56 universal conditions — fairer metric):**
- **#13 out of 80 complete models** (mean rank = 14.38)
- Top 17% of all complete models
- Won 3 champion conditions (1 per task)

**Context:** V739 uses validation-based model selection (8 candidates, harness val_raw, temporal split with 7-day embargo). No oracle test-set leakage. A #13/80 ranking for a validation-based meta-learner among 80 diverse models is a strong result.

**Benchmark top-5 by mean rank (56 universal conditions):**
1. NHITS (4.21)
2. PatchTST (4.36)
3. NBEATS (4.77)
4. NBEATSx (5.84)
5. ChronosBolt (7.11)

**Dominant champion model:** NBEATS — 24/56 conditions won (43%)
**Per-task champion distribution:** NBEATS(8), NHITS(5+), KAN(5), DeepNPTS(4), GRU(3), V739(3), PatchTST(2), Chronos(2+)

## V739 Ablation Note

V739 uses ablation names `{core_only, core_edgar, core_text, full}` while standard models use `{core_only, core_edgar, core_only_seed2, core_edgar_seed2}`.

**Pre-Phase 12 (historical):** `select_dtypes(include=[np.number])` dropped raw text strings → V739 core_text ≡ core_only, V739 full ≡ core_edgar. Comparison was fair since both had 2 unique feature sets × 2 replications.

**Phase 12 (current):** PCA text embeddings (float32 columns `text_emb_0`..`text_emb_63`) now survive `select_dtypes(include=[np.number])`. Text ablation IS FUNCTIONAL. Phase 12 reruns will produce genuine core_text/full results for all models.

## Monitoring Commands (Post-Landing)

### Verify V739 metrics files
```bash
find runs/benchmarks/block3_phase9_fair -path '*/autofit/*/metrics.json' | sort
```

### Count V739 records
```bash
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 -c "
from pathlib import Path; import json
root = Path('/work/projects/eint/repo_root/runs/benchmarks/block3_phase9_fair')
total = sum(len(json.loads(mf.read_text())) for mf in root.rglob('metrics.json') if 'autofit' in str(mf))
print(f'V739 total records: {total}')
"
```

## Landing History

| Date | Event | Job IDs |
| --- | --- | --- |
| 2026-03-15 | V739 ALL 112 conditions landed | l40s: 5250853-5250868; gpu: 5250869-5250872 |
3. confirm `AutoFitV739` rows appear in `all_results.csv`
4. confirm `docs/benchmarks/phase9_current_snapshot.md` is rebuilt
5. only then update any status document with V739 empirical claims
