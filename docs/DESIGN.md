# Block 3 Design (Current)

## 1. Benchmark Design

1. Canonical benchmark root: `runs/benchmarks/block3_phase9_fair/`
2. Condition lattice: `104` conditions
3. Canonical data access: `FreezePointer` → `docs/audits/FULL_SCALE_POINTER.yaml`
4. Freeze assets remain read-only.

## 2. Two Reporting Layers

### Raw materialization layer
Use this layer to answer: what has physically landed?

Current verified values:
- metrics files: `78`
- raw records: `8664`
- raw models: `90`
- raw complete models: `77`

### Filtered leaderboard layer
Use this layer to answer: what remains fair and comparable after exclusions?

Current verified values:
- filtered records: `6672`
- filtered models: `69`
- filtered complete models: `59`
- filtered AutoFit models: none yet

## 3. Current Ablation Semantics

1. The current physical Phase 9 results still reflect the seed-replication reinterpretation.
2. That means the benchmark currently represents:
   - `core_only`
   - `core_only_seed2`
   - `core_edgar`
   - `core_edgar_seed2`
3. The real text-enabled reruns must still be executed on top of the newly available text embedding artifacts.

## 4. Current AutoFit Design Baseline

1. V739 is the only valid current AutoFit baseline.
2. V739 uses validation-based selection with harness-supplied `val_raw`.
3. V734-V738 are historical only and must not be reused for implementation or ranking.

## 5. Current Design Boundary

1. The project is still completing the clean Phase 9 line.
2. Therefore, the current design goal is not “invent the next version first.”
3. The current design goal is to finish a clean benchmark surface that future versions can trust.
