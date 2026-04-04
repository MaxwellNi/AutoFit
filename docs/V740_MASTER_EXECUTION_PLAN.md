# V740 Master Execution Plan

> Date: 2026-04-04
> Status: active operational plan
> Scope: the concrete execution path from the current V740 local research line to
> 1) fair internal benchmark leadership,
> 2) external 2025-2026 time-series generalization leadership,
> 3) a paper package that is strong enough to justify aiming at NeurIPS-level review.
>
> Read together with:
> - `AGENTS.md`
> - `.local_mandatory_preexec.md`
> - `docs/CURRENT_SOURCE_OF_TRUTH.md`
> - `docs/BLOCK3_MODEL_STATUS.md`
> - `docs/BLOCK3_FULL_SOTA_BENCHMARK.md`
> - `docs/references/BLOCK3_CHAMPION_COMPONENT_ANALYSIS.md`
> - `docs/references/V740_DESIGN_SPECIFICATION.md`
> - `docs/references/V740_SHARED112_FAILURE_STRUCTURE_20260402.md`
> - `docs/references/V740_REPR_POSTAUDIT_GATE_20260402.md`
> - `docs/references/V740_EDGAR_TEXT_ROOTCAUSE_AUDIT_20260402.md`
> - `docs/references/V740_INVESTORS_OCCURRENCE_GATE_20260404.md`

## 0. Goal Definitions

This plan tracks three different end-states. They must not be conflated.

### Goal A: Internal fair benchmark champion

This means V740 is not merely locally promising. It must become a credible
top-line winner on the current Block 3 fair benchmark under the same freeze,
the same fairness filter, the same target definitions, and the same leakage
constraints as every other model.

The minimum acceptable interpretation is:

- no hidden runtime ensemble or post-hoc oracle selection
- no benchmark-specific leakage
- no fairness regressions
- no constant-prediction failure lane
- competitive behavior across all three target families:
  - `is_funded`
  - `funding_raised_usd`
  - `investors_count`

### Goal B: External 2025-2026 generalization champion suite

This means V740 is not only a Block 3 solution. It must also win or lead on a
fair, reproduced external suite derived from the public datasets actually used
by mainstream 2025-2026 NeurIPS / ICML / ICLR time-series papers.

The minimum acceptable interpretation is:

- paper-by-paper dataset registry exists
- legal / reproducible data access exists
- splits, metrics, and covariates are aligned fairly
- baseline reruns are audited and not copied from papers blindly
- V740 wins are measured on a controlled shared surface, not cherry-picked

### Goal C: NeurIPS-level oral-grade paper package

This is stricter than "good results".

The package must include:

- a real algorithmic contribution, not only benchmark tuning
- a clean fairness and leakage story
- strong internal benchmark evidence
- strong external generalization evidence
- targeted ablations that isolate why V740 works
- runtime / cost analysis against `V739` and key competitors
- negative results and failure boundaries documented honestly

## 1. Current Honest Status (2026-04-04)

| Workstream | Current state | Honest reading |
| --- | --- | --- |
| Canonical benchmark truth | stable | the truth pack is now strong and should not be reopened casually |
| Valid AutoFit baseline | `AutoFitV739` only, `132/160` | scientifically valid but operationally stalled |
| V740 code surface | real code, not placeholder | target-aware routing, source-native EDGAR/text memory, and count-side regime controls are implemented |
| shared112 non-routed local execution | `112/112` complete | execution coverage is done, but quality is still far from champion level |
| binary lane | post-audit `10/0/6`, routed h1 `2/0/2` | real competitiveness exists, but it is not dominant |
| funding lane | best widened branch `20/28` | partially rescuable, still not a championship lane |
| investors lane | full post-audit `0/0/48`, h1 `0/0/12` | still a structural failure lane |
| investors count repair | softplus + learned occurrence infrastructure landed | the only local evidence that survived is task2-only usefulness; this is not yet benchmark proof |
| formal routed evidence | binary h1 landed, other routed jobs pending | routed code path is real, full routed verdict is still blocked by queue |
| external 2025-2026 suite | not started | there is currently no external generalization benchmark package |
| paper package | not started | there is no oral-grade evidence bundle yet |

Two facts matter most right now:

1. the current V740 line is **not** yet close to an honest claim of universal
   benchmark leadership
2. the current hardest blocker is still `investors_count`, not EDGAR join
   semantics, not missing text embeddings, and not another funding log-domain
   tweak

## 2. Progress Dashboard

These percentages are execution-readiness estimates, not measured truths.
A single overall percentage would be misleading, so the dashboard stays split.

| Workstream | Progress | Bar | Honest reading |
| --- | ---: | --- | --- |
| truth pack / auditability | `95%` | `[##########]` | this is already strong enough for disciplined execution |
| V740 implementation surface | `75%` | `[########..]` | the code scaffolding is real and broad, but not yet quality-complete |
| local diagnostic coverage | `90%` | `[#########.]` | non-routed shared112 and post-audit evidence are already wide |
| binary competitiveness | `60%` | `[######....]` | real wins exist, but the lane is not consistently dominant |
| funding competitiveness | `35%` | `[####......]` | partial rescue exists, but not enough for championship claims |
| investors competitiveness | `8%` | `[#.........]` | structural failure is still the right default reading |
| formal routed evidence | `12%` | `[#.........]` | only the first binary h1 slice has landed |
| internal benchmark champion readiness | `18%` | `[##........]` | too early to call V740 a true challenger |
| external benchmark readiness | `2%` | `[..........]` | registry, harness, and reruns are not built yet |
| oral-grade paper readiness | `5%` | `[#.........]` | the scientific package does not exist yet |

## 3. Current Live Queue Reality

At the time this plan was refreshed, the most relevant V740 jobs were:

- `5305468 v740_112_inv` -> pending, current start estimate `2026-04-07T03:20:00`
- `5305469 v740_112_bin` -> pending, current start estimate `2026-04-07T06:00:00`
- `5305472 v740_112_invh1` -> pending, current start estimate `2026-04-07T12:20:00`
- `5306887 v740_inv_gate` -> pending, current start estimate `2026-04-07T12:30:00`
- `5306906 v740_inv_g2` -> pending, current start estimate `2026-04-07T12:30:00`
- `5306956 v740_inv_t2occ` -> pending, current start estimate `2026-04-07T12:30:00`

So the immediate blocker is still not "missing code". It is the combination of:

- queue delay on the remaining routed / investors jobs
- unresolved investors target geometry
- and the lack of a landed full routed verdict

## 4. Honest Time Estimates

These are planning ranges, not promises.

| Milestone | Optimistic | Base case | Conservative | Why |
| --- | --- | --- | --- | --- |
| settle the current V740 branch verdict | `1-2 weeks` | `2-4 weeks` | `>4 weeks` | depends on pending routed and focused investors jobs |
| turn investors into a non-dead lane | `3-6 weeks` | `6-10 weeks` | `>10 weeks` | likely requires at least one more architectural iteration |
| reach a real shared112 local contender package | `6-10 weeks` | `10-16 weeks` | `>16 weeks` | binary must hold, funding must not regress, investors must stop being zero-win |
| reach a credible internal benchmark champion candidate | `10-18 weeks` | `4-8 months` | `>8 months` | requires full routed evidence and wider fair-surface dominance |
| build external benchmark suite v1 | `12-20 weeks` | `5-8 months` | `>8 months` | dataset registry, adapters, reruns, and baseline reproduction all need to exist |
| reach an oral-grade paper package | `20-32 weeks` | `8-14 months` | `>14 months` | requires both results and the scientific explanation package |

### Practical reading

If the target is truly:

- internal fair benchmark absolute champion,
- external 2025-2026 suite champion,
- and oral-grade paper quality,

then the honest base-case estimate from the current state is **closer to many
months than many weeks**.

If the investors lane does not stop being structurally broken soon, the right
response is not infinite patching. The right response is to branch the next
architecture generation (`V741+`) intentionally.

## 5. Critical Path Ranking

### Real critical path

1. land and read the currently pending routed / investors jobs
2. resolve the investors target-geometry failure
3. prevent route collapse while preserving binary wins
4. reintegrate funding without reopening the failed log-domain branch
5. move from partial local competitiveness to full routed shared112 quality
6. then build the external generalization suite

### Things that are no longer the critical path

1. exact-day vs as-of EDGAR join debugging
2. claims that text embeddings are missing
3. another broad funding log-domain patch cycle
4. generic "one more representation tweak everywhere" without target-specific evidence

## 6. Phase Plan

## Phase 0: Queue Readout And Truth Synchronization

### Objective

Finish the current evidence wave honestly before inventing a new narrative.

### Concrete tasks

1. read out `5305468`, `5305469`, `5305472`, `5306887`, `5306906`, and `5306956`
2. summarize only executed results into docs
3. decide whether the task2-only occurrence gate survives beyond local smokes
4. update the V740 status notes with executed evidence only

### Exit criteria

1. every currently queued V740 job is either landed, failed, or explicitly replaced
2. the task2-only occurrence gate has a clear keep / revise / discard verdict
3. there is no ambiguity about what the routed line has actually proven

## Phase 1: Investors Rescue

### Objective

Convert `investors_count` from a structural failure lane into a real competitive lane.

### Concrete tasks

1. keep `softplus` as the global positive count transform
2. keep occurrence/intensity separation as the active count-side direction
3. test anti-collapse routing controls on investors specifically
4. test count-specific objective variants that penalize false activity without killing active windows
5. test source-conditioned routing on investors instead of one global gate recipe
6. keep all investors experiments benchmark-like and paired against no-gate and route-off controls
7. use short, resumable h1 probes first; widen only after a real signal appears

### Hard stop rules

1. if investors h1 stays below `4/12` wins after `2-3` serious architecture-quality iterations, stop polishing the current V740 line and open `V741+`
2. if full investors local stays below `12/48`, do not claim the lane is "basically solved"
3. if a count fix regresses binary or funding badly, it is not a valid integrated fix

### Exit criteria

1. no constant-prediction failure on the audited investors surface
2. routed investors h1 reaches at least a non-trivial win signal
3. full investors local is no longer a zero-win lane

## Phase 2: Integrated Shared112 Contender Package

### Objective

Assemble one target-aware V740 package that is no longer locally broken by lane.

### Concrete tasks

1. preserve the binary win structure on `full` and `core_edgar`
2. keep funding on the no-log strong-anchor family only
3. integrate the best surviving investors repair from Phase 1
4. run a full routed shared112 local package
5. audit route entropy, dominant-expert collapse, fairness, constant prediction, and per-target failure pockets

### Exit criteria

1. binary remains a real positive lane
2. funding does not collapse back to patch1-style catastrophic behavior
3. investors is no longer zero-win
4. aggregate shared112 local outcome is materially better than the current `15/2/95`

## Phase 3: Internal Benchmark Champion Attempt

### Objective

Turn V740 from a local-only contender into a real candidate for current fair benchmark leadership.

### Concrete tasks

1. widen from the routed shared112 surface to the full comparable surfaces
2. compare against `V739` and the current clean frontier, not only one incumbent cell at a time
3. track both mean-rank and win-count behavior
4. measure runtime and cost against `V739`
5. keep promotion criteria explicit and stable

### Promotion criteria

1. top-tier mean rank on the `112`-cell non-seed shared surface
2. top-tier mean rank on the strict `160`-cell comparable surface
3. no target family remains catastrophic
4. full auditability of routing / target regime behavior

### Exit criteria

1. V740 becomes a real internal benchmark leader rather than a promising local-only line
2. the remaining debate shifts from viability to margin and robustness

## Phase 4: External 2025-2026 Benchmark Suite Build-Out

### Objective

Create the external suite first, then optimize on it honestly.

### Concrete tasks

1. build a paper-by-paper registry for mainstream 2025-2026 NeurIPS / ICML / ICLR time-series papers
2. record, for each paper:
   - dataset name
   - task family
   - official split if available
   - horizon(s)
   - metric(s)
   - covariates / modalities
   - code / license / access path
3. partition the registry into:
   - directly reproducible
   - reproducible with adaptation
   - not legally or technically reproducible yet
4. define one external fairness contract:
   - same train / validation / test logic when possible
   - same budget class across models
   - no hidden feature advantage for V740
5. implement dataset adapters and a unified benchmark harness
6. reproduce the strongest available baselines before claiming any V740 generalization win

### Exit criteria

1. a stable external-suite registry exists
2. the external harness is reproducible
3. the best baseline reruns are audited rather than copied from papers

## Phase 5: External Champion Push And Paper Package

### Objective

Only after the suite exists should V740 try to dominate it.

### Concrete tasks

1. run V740 on the external suite with a fixed documented recipe
2. compare against the reproduced baseline set on a shared evaluation surface
3. collect category-level results, not only one grand average
4. run ablations for:
   - target routing
   - source-native EDGAR/text memories
   - investors occurrence/intensity path
   - funding anchor regime
   - runtime / cost tradeoff versus `V739`
5. write the negative-result section honestly
6. freeze the final paper recipe and rerun for reproducibility

### Exit criteria

1. V740 is demonstrably strong both internally and externally
2. the mechanism story is isolated by ablations
3. the paper package is robust enough to survive hostile review

## 7. Immediate Task Inventory From Today

### Immediate

1. read out all currently pending V740 jobs
2. decide the fate of the task2-only investors occurrence gate
3. keep the investors line on the shortest feedback loop possible
4. update docs after every landed result wave

### Near-term

1. turn investors h1 into a non-zero-win routed lane
2. widen the best investors repair beyond h1 only if it keeps winning
3. preserve binary wins while reintegrating the repaired investors line
4. keep funding on the no-log strong-anchor family only

### Medium-term

1. rerun full routed shared112
2. audit route collapse and per-target behavior
3. establish a full comparable internal benchmark challenger package
4. decide whether V740 remains the right vehicle or whether `V741+` is required

### Longer-term

1. build the external 2025-2026 dataset registry
2. implement the external benchmark harness
3. reproduce the strongest public baselines fairly
4. run the external-suite champion push
5. build the oral-grade paper package

## 8. Weekly Operating Rules

1. do not count queued jobs as results
2. do not reopen falsified root-cause stories unless new evidence appears
3. every landed V740 wave must update at least one status doc
4. every new mechanism must name its protected guard surface before it is widened
5. if a lane stays structurally broken after multiple high-quality iterations, branch the architecture instead of endlessly patching

## 9. Bottom Line

V740 is currently in the transition between:

- "broad local-only audited prototype"
- and "real benchmark contender"

It is **not** yet in the phase where an honest person should promise:

- internal absolute championship,
- external suite dominance,
- or oral-grade paper certainty.

The right present reading is:

- truth discipline is strong,
- code surface is strong,
- binary is alive,
- funding is partially alive,
- investors is the hard blocker,
- routed evidence is still too partial,
- and the external generalization program has not started.

That means the project is still far from done, but the next concrete steps are
now much clearer than they were before the April evidence wave.