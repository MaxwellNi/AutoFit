# V740 Shared112 Failure Structure (2026-04-02)

This note consolidates what the current shared112 local surface actually says about V740, and records the controlled follow-up path now landed in code.

It is a local-only engineering diagnosis note. It does not alter the canonical Phase 9 benchmark line.

## Bottom Line

- shared112 is already fully covered: `112/112`
- aggregate honest local outcome is still `15 wins / 2 ties / 95 losses`
- the failures are not homogeneous
- binary, funding, and investors are three different problems and should not be patched as if they were one

## Binary Lane

Binary is not an "everything is broken" lane.

- strongest known guard cell remains `task1_outcome / full / is_funded / h30`
- prior honest local state there was a true win structure: V740 `0.1076` vs incumbent `0.2730`
- `full` is the best binary ablation family
- `core_edgar` also has live binary signal
- `core_only` and `core_text` are materially weaker

Operational consequence:

- any funding-side intervention must preserve the binary guard cell
- a funding patch that degrades the binary guard is disqualified even if it helps one funding slice

## Funding Lane

Funding is not uniformly dead, but the viable zone is narrow.

- funding aggregate is `8 wins / 40 losses`
- the only clearly live region is `core_edgar`, mainly shorter-horizon slices
- `full` is the catastrophic region
- the largest honest local failures are all `full` funding cells with incumbent MAE near `463` and V740 still in the `1.4e5` to `1.9e5` range

Representative failures:

- `task2_forecast / full / funding_raised_usd / h7`: `158726.3889` vs `463.6938`
- `task2_forecast / full / funding_raised_usd / h30`: `158458.3889` vs `463.6847`
- `task2_forecast / core_edgar / funding_raised_usd / h30`: `148158.9125` vs `463.6847`

Patch1 conclusion:

- patch1 bundled three mechanisms together: log domain, source scaling, and anchor
- honest post-patch gate failed
- `full / funding / h7` and `core_only / funding / h30` exploded further
- binary guard regressed from win state to slight loss state

Operational consequence:

- funding must now be tested as a controlled mechanism split, not by another bundled patch

## Investors Lane

Investors is a separate unsolved route, not a side effect of funding tuning.

- investors aggregate is `0 wins / 0 ties / 48 losses`
- many cells show constant predictions or near-constant predictions
- many incumbents are effectively at `0.000x` local MAE while V740 remains in `10s` to `200s`
- this is consistent with target geometry / output-head / scale calibration failure, not a small regularization miss

Operational consequence:

- investors should not be treated as a downstream beneficiary of funding fixes
- investors needs its own inductive-bias path later

## Engineering Changes Landed On 2026-04-02

The code now exposes funding mechanisms as independent controls.

In `src/narrative/block3/models/v740_alpha.py`:

- `enable_funding_log_domain`
- `enable_funding_source_scaling`
- `enable_funding_anchor`
- `funding_anchor_strength`

The local harnesses now expose the same flags:

- `scripts/run_v740_shared112_champion_loop.py`
- `scripts/run_v740_alpha_smoke_slice.py`
- `scripts/run_v740_alpha_minibenchmark.py`

Each V740 local artifact can now self-describe its regime with:

- `regime_info`
- effective `funding_log_domain`
- effective `funding_source_scaling`
- effective `funding_anchor_enabled`
- effective `funding_anchor_strength`

Artifact hygiene was also tightened:

- local JSON writers now stamp the actual `json_path`
- skip-existing reloads rewrite stale `json_path` metadata to the real path before reuse

## Controlled Split Gate

New resumable gate script:

- `.slurm_scripts/v740_local/v740_funding_patchsplit_gate_gpu.sh`

It tests the same focused gate surface across six controlled variants:

- `baseline_off`: log off, scaling off, anchor off
- `log_only`: log on, scaling off, anchor off
- `log_anchor`: log on, scaling off, anchor on
- `log_scale`: log on, scaling on, anchor off
- `patch1_full`: log on, scaling on, anchor on
- `scale_anchor_no_log`: log off, scaling on, anchor on

Each variant keeps the same five cells:

- `task2_forecast / full / funding_raised_usd / h7`
- `task2_forecast / full / funding_raised_usd / h30`
- `task2_forecast / core_edgar / funding_raised_usd / h30`
- `task2_forecast / core_only / funding_raised_usd / h30`
- `task1_outcome / full / is_funded / h30` binary guard

This is the minimum honest split needed to answer four questions cleanly:

- is `log1p` itself the dominant damage source?
- does source scaling help only when paired with log domain?
- does anchor help only in log domain, or does it damage funding regardless?
- can any funding improvement survive the binary guard?

## No-Log Sub-Split Result

The follow-up no-log gate `5304068 v740_fnd_g3` has now completed and sharpens the attribution further.

Key result:

- `log1p` is no longer the live question; it is already ruled out as the dominant damage source
- source scaling alone is near-inert on the tested funding cells
- the main rescue mechanism is the continuous anchor
- strong anchor `0.85` is materially better than weak anchor `0.35`
- source scaling only becomes meaningfully useful when paired with the strong anchor, mainly on the hardest `full / h7` funding cell

Operational consequence:

- the two only widening-worthy funding branches are now:
	- `anchor_only_no_log_a085`
	- `scale_anchor_no_log_a085`
- future funding-side widening should compare those two branches on the broader 48-cell funding surface instead of reopening log variants or low-anchor variants
- binary guard remains non-discriminative for these funding controls because the effective funding regime is inactive on binary targets

## EDGAR and Text Root-Cause Audit

A first direct source audit now exists at:

- `docs/references/V740_EDGAR_TEXT_ROOTCAUSE_AUDIT_20260402.md`

The most important consequence is that two easy fallback explanations are now substantially weaker.

- `core_edgar` is not mainly failing because local exact-day join semantics differ from benchmark-style backward as-of join; on the frozen daily surface, the two joins are effectively identical for matched CIK rows
- the stronger EDGAR-side limitation is source reach: only about `25.6%` of core rows carry a usable `cik`, so EDGAR can only influence a minority of the panel at all
- text is not mainly failing because embeddings are absent; the active `gte-Qwen2-1.5B` PCA artifact has row parity with the core panel and sampled exact-join coverage is complete
- the stronger text-side explanation is representation failure: heavy repetition, sparse informative fields, `1536 -> 64` PCA compression, and dense daily consumption of what behaves more like a sparse semantic event source

Operational consequence:

- the next V740 implementation wave should emphasize source-native EDGAR/text event memory and target-aware usage, not another round of join debugging or unsupported claims that the current text artifact never landed

## Current Working Rule

Use the micromamba `insider` interpreter as the authoritative local validation runtime for these scripts.

- system `python3` on this host is too old for repo syntax
- ad-hoc `python3.11` is acceptable for quick syntax checks
- meaningful local validation should prefer the `insider` environment already used by the SLURM V740 local jobs