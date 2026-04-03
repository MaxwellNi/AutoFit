# V740 Patch3 SLURM Continuation (Updated 2026-04-03)

This note records the current patch3 local artifacts and the honest SLURM continuation path.

As of 2026-04-03, active April V740 local benchmark artifacts live under:

- `runs/benchmarks/v740_localclear_20260401/`
- `runs/benchmarks/v740_localclear_20260402/`
- `runs/benchmarks/v740_localclear_20260403/`

`docs/references/` keeps summaries and continuation notes only.
Earlier tracked smoke and minibench artifacts remain historical reference material.

## Preserved Local Patch3 Artifacts

- H1 patch3 probe summary: `docs/references/V740_SHARED112_BINARY_PROBE_PATCH3_20260401.md`
- H1 patch3 probe JSON root: `runs/benchmarks/v740_localclear_20260401/v740_shared112_binary_probe_patch3_20260401/`
- Partial full binary patch3 JSON root from the interrupted local run: `runs/benchmarks/v740_localclear_20260401/v740_shared112_binary_patch3_16_20260401/`

Current partial patch3 full-loop state before SLURM resume:

- `19` JSON artifacts already landed under `v740_shared112_binary_patch3_16_20260401/`
- `--skip-existing` in `scripts/run_v740_shared112_champion_loop.py` resumes at per-JSON granularity
- one interrupted cell already has only the `v740_alpha` side written, so resume should fill only the missing incumbent side and remaining cases

## Current V740 Local-Only SLURM Surface

### Completed jobs

- `5303499 v740_112_bin`: completed, output root `runs/benchmarks/v740_localclear_20260401/v740_shared112_binary_loop_20260401/`
- `5303500 v740_bin_ab`: completed, output root `runs/benchmarks/v740_localclear_20260401/v740_alpha_component_ablation_binary_20260401/`
- `5303501 v740_112_inv`: completed, output root `runs/benchmarks/v740_localclear_20260401/v740_shared112_investors_loop_20260401/`
- `5303613 v740_bin_p3r`: completed, patch3 binary resume job for `runs/benchmarks/v740_localclear_20260401/v740_shared112_binary_patch3_16_20260401/`
- `5303702 v740_112_fnd`: completed, funding loop for `runs/benchmarks/v740_localclear_20260401/v740_shared112_funding_loop_20260401/`
- `5303844 v740_fnd_g1`: completed, focused post-patch funding and binary gate for `runs/benchmarks/v740_localclear_20260401/v740_funding_patch1_gate_20260401/`
- `5304057 v740_fnd_g2`: completed, controlled funding mechanism split gate for `runs/benchmarks/v740_localclear_20260401/v740_funding_patchsplit_gate_20260402/`
- `5304068 v740_fnd_g3`: completed, no-log funding sub-split gate for `runs/benchmarks/v740_localclear_20260401/v740_funding_nolog_subsplit_gate_20260402/`
- `5304260 v740_fnd_g4`: completed, widened best-branch funding duel for `runs/benchmarks/v740_localclear_20260401/v740_funding_bestbranch_duel_20260402/`

### Pending jobs

- `5304393 v740_repr_pa`: pending, output roots
  - `runs/benchmarks/v740_localclear_20260402/v740_shared112_binary_postaudit_20260402/`
  - `runs/benchmarks/v740_localclear_20260402/v740_shared112_investors_postaudit_20260402/`
- `5305468 v740_112_inv`: pending, output root `runs/benchmarks/v740_localclear_20260402/v740_shared112_investors_routed_loop_20260402/`
- `5305469 v740_112_bin`: pending, output root `runs/benchmarks/v740_localclear_20260402/v740_shared112_binary_routed_guard_20260402/`
- `5305472 v740_112_invh1`: pending, output root `runs/benchmarks/v740_localclear_20260403/v740_shared112_investors_routed_h1_probe_20260403/`
- `5305473 v740_112_binh1`: pending, output root `runs/benchmarks/v740_localclear_20260403/v740_shared112_binary_routed_h1_probe_20260403/`

The completed `v740_112_bin` result from 2026-04-01 is a separate non-routed binary-loop landing and must not overwrite the patch3-resume path.
Patch3 continuation should keep using the dedicated interrupted-run root below.

## Current Shared112 Progress

- `is_funded`: `16/16` complete in `docs/references/V740_SHARED112_BINARY_PATCH3_16_20260401.md`
- `investors_count`: `48/48` complete in `docs/references/V740_SHARED112_INVESTORS_LOOP_20260401.md`
- `funding_raised_usd`: `48/48` complete in `docs/references/V740_SHARED112_FUNDING_LOOP_20260401.md`
- aggregate shared112 progress after the non-routed April loop wave: `112/112` complete (`100%`)
- aggregate shared112 outcome after all three non-routed lanes landed: `15` wins / `2` ties / `95` losses
- target split after all three non-routed lanes landed:
  - `is_funded`: `7/2/7`
  - `funding_raised_usd`: `8/0/40`
  - `investors_count`: `0/0/48`
- current active April V740 local validation jobs: `5`, all still `PENDING` on `gpu`

## Funding Gate Summary

### Post-patch1 gate

- `5303844` showed the first funding patch did **not** pass the honest gate.
- `log1p` funding stayed catastrophic and even regressed binary guard behavior.

### Controlled split gate

- `5304057` isolated `log1p` as the dominant funding damage source.
- `scale_anchor_no_log` emerged as the best current funding combination.

### No-log sub-split gate

- `5304068` showed source scaling alone is near-inert.
- the dominant rescue mechanism is the continuous anchor
- anchor `0.85` is materially stronger than anchor `0.35`
- scaling only matters once paired with the strong anchor
- best current primary branch: `scale_anchor_no_log_a085`
- best current secondary comparator: `anchor_only_no_log_a085`

### Wider best-branch duel

- `5304260` completed successfully in `00:44:10`
- result surface: `runs/benchmarks/v740_localclear_20260401/v740_funding_bestbranch_duel_20260402/`
- result summary:
  - `anchor_only_no_log_a085`: `20 wins / 28 losses`
  - `scale_anchor_no_log_a085`: `20 wins / 28 losses`
  - `full` funding remained `0 wins / 12 losses`
- honest read:
  - the current funding line now has a clear no-log strong-anchor ceiling
  - small regime tweaks are no longer the main blocker

## Routed Target-Specialization Continuation (2026-04-03)

### Formal routed full loops

- script: `.slurm_scripts/v740_local/v740_shared112_investors_loop_gpu.sh`
  - job: `5305468 v740_112_inv`
  - output root: `runs/benchmarks/v740_localclear_20260402/v740_shared112_investors_routed_loop_20260402/`
  - summary target: `docs/references/V740_SHARED112_INVESTORS_ROUTED_LOOP_20260402.md`
  - current scheduler ETA: `2026-04-12T11:20:00`
- script: `.slurm_scripts/v740_local/v740_shared112_binary_loop_gpu.sh`
  - job: `5305469 v740_112_bin`
  - output root: `runs/benchmarks/v740_localclear_20260402/v740_shared112_binary_routed_guard_20260402/`
  - summary target: `docs/references/V740_SHARED112_BINARY_ROUTED_GUARD_20260402.md`
  - current scheduler ETA: `2026-04-12T12:10:00`

### Faster routed h1 probes

- script: `.slurm_scripts/v740_local/v740_shared112_investors_h1_routed_probe_gpu.sh`
  - job: `5305472 v740_112_invh1`
  - output root: `runs/benchmarks/v740_localclear_20260403/v740_shared112_investors_routed_h1_probe_20260403/`
  - summary target: `docs/references/V740_SHARED112_INVESTORS_ROUTED_H1_PROBE_20260403.md`
  - current scheduler ETA: `2026-04-12T15:30:00`
- script: `.slurm_scripts/v740_local/v740_shared112_binary_h1_routed_probe_gpu.sh`
  - job: `5305473 v740_112_binh1`
  - output root: `runs/benchmarks/v740_localclear_20260403/v740_shared112_binary_routed_h1_probe_20260403/`
  - summary target: `docs/references/V740_SHARED112_BINARY_ROUTED_H1_PROBE_20260403.md`
  - current scheduler ETA: `2026-04-12T15:40:00`

### Honest current state

- no routed summary markdown has landed yet
- no routed JSON outputs have landed yet
- the routed path is therefore still an execution plan, not executed evidence

## Intended Resume Path

- resume script: `.slurm_scripts/v740_local/v740_shared112_binary_patch3_resume_gpu.sh`
- output root: `runs/benchmarks/v740_localclear_20260401/v740_shared112_binary_patch3_16_20260401/`
- summary target: `docs/references/V740_SHARED112_BINARY_PATCH3_16_20260401.md`
- surface manifest: `runs/benchmarks/v740_localclear_20260401/v740_shared112_surface_20260401.json`

This preserves the already-finished local patch3 work and keeps any future V740 local benchmark work under isolated `runs/benchmarks` roots instead of mixing raw artifacts into `docs/references`.
