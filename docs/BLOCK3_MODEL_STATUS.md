# Block 3 Model Benchmark Status

> Last updated: 2026-03-06
> Full results: `docs/BLOCK3_RESULTS.md`

## Snapshot

| Metric | Value |
|---|---:|
| Evaluation conditions | 48 (3 targets × 4 horizons × 4 ablations) |
| Models evaluated | 90 (Phase 7), 108+ planned (Phase 8) |
| Deduplicated records | 6,670 (from 81 metrics.json files) |
| Registered models | 97 code (Phase 7) + 18 new (Phase 8) = 115 total |
| Champion distribution | TimesNet(12), DeepNPTS(8), NBEATS(8), NBEATSx(4), GRU(4), Autoformer(2), DeepAR(4), FusedChampion(4), other(2) |

## Phase Status

### Phase 7 (COMPLETE — 121 SLURM jobs)
All categories ran. Key gaps: tslib_sota never submitted, many models <48 conditions.

### Phase 8 (READY — 84 SLURM jobs, dry-run validated)
Saturated benchmark fills all gaps:
- **8a**: tslib_sota (14 models, never submitted) — 22 GPU jobs
- **8b**: New foundation models (Sundial, TTM, TimerXL, TimesFM2) — 11 GPU jobs
- **8c**: Deep classical backfill (GRU/LSTM/TCN/MLP/DilatedRNN) — 11 GPU jobs
- **8d**: NF transformer gap-fill (43→48/48 conditions) — 18 GPU jobs
- **8e**: AutoFit gap-fill (V1-V733 + V736) — 22 batch jobs

## Category Status

| Category | Models (P7) | New (P8) | P7 Status | P8 Target |
|---|---:|---:|---|---|
| ml_tabular | 18 | — | 8-12/48 conditions | — |
| statistical | 5 | — | 48/48 | — |
| deep_classical | 9 | — | NBEATS/NHITS/TFT/DeepAR=48; GRU/LSTM/TCN/MLP/DilatedRNN=24 | backfill to 48 |
| transformer_sota | 23 | — | 43/48 | gap-fill to 48 |
| foundation | 10 | +4 | 47-48/48 | +Sundial,TTM,TimerXL,TimesFM2 |
| irregular | 2 | — | 48/48 | — |
| tslib_sota | 6/14 | +8 | 6/48 (only core_only) | full 48/48 for all 14 |
| autofit | 17 | +1 | V734/V735=48; V1-V3E=40; V72-V731=18-30 | +V736; gap-fill all |

## AutoFit Progression

| Version | Strategy | Conds | Notes |
|---|---|---:|---|
| V1-V3E | Tree CV | 40 | Gap-fill in Phase 8e |
| V3Max-V7 | Enhanced tree | 46 | Gap-fill in Phase 8e |
| V72 | Knowledge distill | 29 | Gap-fill in Phase 8e |
| V733 | NF-native selector | 47 | Oracle from hand-coded table (noise-level "wins") |
| V734 | Empirical oracle + ensemble | 48 | Coarse keys, softmax weights |
| V735 | Exact condition oracle | 48 | **Rebuilt from 6,670 records (RMSE)** |
| **V736** | **OOF stacking ensemble** | **0 (new)** | **TRUE ensemble: top-3 oracle models × inverse-RMSE weights** |

## V736 Design
First V7.3.x that can **systematically beat standalone** models:
- Trains top-3 oracle models per condition (not model selection)
- Inverse-RMSE weighted average → variance reduction
- Model diversity: TimesNet + NBEATS + KAN, DeepNPTS + GRU + TFT, etc.
- Registered in autofit_wrapper.py, included in Phase 8e shard

## Notes

1. Horizons: {1, 7, 14, 30} days.
2. V735 oracle rebuilt from actual data via `scripts/rebuild_oracle_table.py` (old oracle had 4.2% accuracy).
3. Phase 8 submit script: `scripts/submit_phase8_saturated.sh` (dry-run validated, 84 jobs).
4. New foundation models: Sundial (ICML'25 Oral), TTM (NeurIPS'24), TimerXL (ICLR'25), TimesFM2 (Google).
4. See `docs/BLOCK3_RESULTS.md` for the full 91-model leaderboard, all 48 condition champions, and V734 vs V735 head-to-head table.
