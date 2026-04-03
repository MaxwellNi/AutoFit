# AutoFit Retirement Ledger

> Current active AutoFit baseline: `AutoFitV739` only.
> Effective current-surface cleanup landed: 2026-04-03.
> Current-surface impact of the cleanup: `23` archived AutoFit-family models / aliases and `460` rows removed from the active registry and current public leaderboard outputs.

This ledger records why older AutoFit-family lines were retired or invalidated and what lessons must be preserved from those failed iterations.

## Current Rule

- The active registry exports `AutoFitV739` only.
- `runs/benchmarks/block3_phase9_fair/` may still contain archived AutoFit-family raw artifacts for auditability.
- Archived lines must not re-enter `registry.py`, `all_results.csv`, `docs/BLOCK3_RESULTS.md`, or `docs/benchmarks/phase9_current_snapshot.*` as active current-surface entries.

## Retirement / Invalidation Ledger

| Line | Status | Why it left the current environment | Lessons retained |
|---|---|---|---|
| `AutoFitV1-V3Max` | Archived historical | Early stacking / ensemble exploration line, superseded by later clean baselines | More ensemble search does not compensate for weak current-surface governance or weak candidate quality |
| `AutoFitV4-V7` | Archived historical | Added complexity and heuristics without becoming a durable benchmark-clean frontier | Complexity inflation must be justified by stable wins under the canonical benchmark, not by local improvements |
| `AutoFitV71-V733` | Archived historical | Lane-adaptive / champion-transfer / oracle-adjacent research line, superseded by the clean V739 line | Routing ideas must stay benchmark-clean and must not blur the boundary between historical research lines and current baselines |
| `FusedChampion` (`V7.3.2`) | Archived historical | Historical fused prototype retired after root-cause audit | Champion orchestration should stay simple until the full failure surface is closed and audited |
| `NFAdaptiveChampion` (pre-V739 alias) | Archived historical | Historical pre-clean alias superseded by the named V739 implementation | Aliases must not remain in the active registry once a clean successor line exists |
| `AutoFitV734-V738` | **INVALID** | Oracle test-set leakage | Never use oracle or test-set knowledge in routing, champion selection, or leaderboard claims |

## Hard Lessons From The Cleanup

1. Documentation alone is not enough. The policy has to be enforced in the active registry and in every public artifact generator.
2. Prefix-only filtering is not enough. `FusedChampion` and `NFAdaptiveChampion` were historical AutoFit-family aliases that also had to be purged from the current surface.
3. Raw audit preservation and current-surface cleanliness are different responsibilities. Historical lines can remain in frozen raw artifacts, but they must be excluded from active execution and active comparisons.
4. Leakage failures must stay visible as lessons, not as runnable baselines. `V734-V738` remain documented here precisely so they cannot quietly return as current candidates.

## Operational Consequences

- Keep the active `autofit` category restricted to `AutoFitV739`.
- Treat archived AutoFit-family lines as audit-only material.
- Record future AutoFit retirement reasons here instead of letting archived lines persist in current status tables.