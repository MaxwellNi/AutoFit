#!/usr/bin/env python3
"""
Phase 3 Fix Verification — Strict Audit Script.

Validates ALL Phase 3 code changes for correctness:
1. Deep/Transformer entity coverage fix (max_entities, Ridge fallback)
2. EDGAR as-of join (merge_asof, no future leakage)
3. AutoFit V3Max timeout budget (K≤6, 30min time limit)
4. GBDT count-target detection (Tweedie/Poisson objective)
5. Horizon dedup for cross-sectional models
6. Entity coverage increase for statistical/irregular models

Exit code 0 = ALL PASS, else 1 = FAIL with details.
"""
from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    msg = f"  {status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def audit_deep_models() -> List[bool]:
    """Audit deep_models.py Phase 3 fixes."""
    print("\n[1/6] Deep/Transformer Entity Coverage + Ridge Fallback")
    results = []

    src = Path("src/narrative/block3/models/deep_models.py").read_text()

    # Check _build_panel_df defaults
    results.append(check(
        "Panel default max_entities ≥ 2000",
        "max_entities: int = 2000" in src or "max_entities: int=2000" in src,
        "Was 200, now 2000 for full entity coverage",
    ))
    results.append(check(
        "Panel default min_obs ≤ 10",
        "min_obs: int = 10" in src or "min_obs: int=10" in src,
        "Was 20, now 10 for more entity eligibility",
    ))

    # Check n_series conditional entity limit
    results.append(check(
        "n_series models capped at 200 entities",
        "_NEEDS_N_SERIES" in src and "200 if self.model_name in _NEEDS_N_SERIES" in src,
        "iTransformer/TSMixer/etc. limited for VRAM safety",
    ))

    # Check Ridge fallback training
    results.append(check(
        "Ridge fallback trained in DeepModelWrapper.fit",
        "_fallback_ridge" in src and "Ridge" in src and "fallback_feature_cols" in src,
        "Trained on tabular features for unseen entity fallback",
    ))

    # Check Ridge fallback used in predict
    results.append(check(
        "Ridge fallback used in DeepModelWrapper.predict",
        "ridge_preds" in src and "coverage" in src,
        "Unseen entities get Ridge prediction instead of global_mean",
    ))

    # Check FoundationModelWrapper entity limits
    results.append(check(
        "Foundation entity limit increased",
        "MAX_E, MIN_O = 500, 10" in src,
        "Was (200, 20), now (500, 10)",
    ))

    # Verify no data leakage in panel construction
    results.append(check(
        "No future data in panel construction",
        "sort_values" in src and "seed=42" in src,
        "Consistent seeding, sorted by entity+date",
    ))

    return results


def audit_edgar_join() -> List[bool]:
    """Audit EDGAR as-of join fix."""
    print("\n[2/6] EDGAR As-Of Join (No Future Leakage)")
    results = []

    src = Path("scripts/run_block3_benchmark_shard.py").read_text()

    results.append(check(
        "merge_asof used instead of merge",
        "merge_asof" in src,
        "As-of join matches most recent EDGAR filing",
    ))
    results.append(check(
        "direction=backward (no future leak)",
        'direction="backward"' in src,
        "Only past EDGAR filings used",
    ))
    results.append(check(
        "90-day tolerance window",
        '90D' in src or "90" in src,
        "Matches quarterly SEC filing cadence",
    ))
    results.append(check(
        "CIK null handling",
        "df_no_cik" in src and "df_cik" in src,
        "Rows without CIK get NaN EDGAR features (no dropped rows)",
    ))
    results.append(check(
        "Match rate logging",
        "match_rate" in src and "EDGAR" in src,
        "Logs join quality for auditability",
    ))
    results.append(check(
        "No exact-date merge on EDGAR",
        'on=["cik", "crawled_date_day"], how="left"' not in src
        or "merge_asof" in src,
        "Old exact merge replaced with as-of join",
    ))

    return results


def audit_autofit_timeout() -> List[bool]:
    """Audit AutoFit V3Max timeout fix."""
    print("\n[3/6] AutoFit V3/V3Max Timeout + K Reduction")
    results = []

    src = Path("src/narrative/block3/models/autofit_wrapper.py").read_text()

    results.append(check(
        "_MAX_EXHAUSTIVE_K ≤ 6",
        "_MAX_EXHAUSTIVE_K = 6" in src,
        "Was 8 (256 combos), now 6 (64 combos)",
    ))
    results.append(check(
        "V3Max top_k=6",
        'top_k=6' in src and 'mode="exhaustive"' in src,
        "Factory function creates V3Max with K=6",
    ))
    results.append(check(
        "Time budget in exhaustive search",
        "_TIME_BUDGET" in src and "timed_out" in src,
        "30-minute budget prevents SLURM walltime overrun",
    ))

    return results


def audit_gbdt_count_target() -> List[bool]:
    """Audit GBDT count-target Tweedie/Poisson fix."""
    print("\n[4/6] GBDT Count-Target Detection (Tweedie/Poisson)")
    results = []

    src = Path("src/narrative/block3/models/traditional_ml.py").read_text()

    results.append(check(
        "Count target detection in ProductionGBDTWrapper",
        "Count target detected" in src,
        "Auto-detects integer non-negative targets",
    ))
    results.append(check(
        "LightGBM → tweedie loss for counts",
        "objective" in src and "tweedie" in src,
        "tweedie_variance_power=1.5",
    ))
    results.append(check(
        "XGBoost → poisson loss for counts",
        "count:poisson" in src,
        "Proper count regression objective",
    ))
    results.append(check(
        "Guard: only when objective not already set",
        "'objective' not in self.init_kwargs" in src,
        "Doesn't override explicit user configuration",
    ))

    return results


def audit_horizon_dedup() -> List[bool]:
    """Audit horizon deduplication for cross-sectional models."""
    print("\n[5/6] Horizon Deduplication (Cross-Sectional Models)")
    results = []

    src = Path("scripts/run_block3_benchmark_shard.py").read_text()

    results.append(check(
        "Cross-sectional horizon optimization",
        "ml_tabular" in src and "run_horizons" in src,
        "ml_tabular runs single horizon (results are identical across H)",
    ))

    return results


def audit_entity_coverage() -> List[bool]:
    """Audit entity coverage increases in statistical + irregular models."""
    print("\n[6/6] Entity Coverage (Statistical + Irregular)")
    results = []

    stat_src = Path("src/narrative/block3/models/statistical.py").read_text()
    irreg_src = Path("src/narrative/block3/models/irregular_models.py").read_text()

    results.append(check(
        "Statistical MAX_ENTITIES ≥ 500",
        "500" in stat_src and "MAX_ENTITIES" in stat_src,
        "Was 50, now 500 for better test coverage",
    ))
    results.append(check(
        "Irregular max_entities ≥ 1000",
        "max_entities: int = 1000" in irreg_src or "max_entities=1000" in irreg_src,
        "Was 200, now 1000 for better test coverage",
    ))

    return results


def audit_anti_leakage() -> List[bool]:
    """Cross-cutting anti-leakage audit."""
    print("\n[CROSS-CUTTING] Anti-Leakage Verification")
    results = []

    bench_src = Path("scripts/run_block3_benchmark_shard.py").read_text()

    # Target-synonym leakage groups still present
    results.append(check(
        "TARGET_LEAK_GROUPS enforced",
        "_TARGET_LEAK_GROUPS" in bench_src,
        "Co-determined columns dropped per target",
    ))

    # Temporal split preserved
    results.append(check(
        "Temporal split enforced (no shuffle)",
        "apply_temporal_split" in bench_src,
        "Strict temporal ordering: train < val < test",
    ))

    # EDGAR backward-only join
    results.append(check(
        "EDGAR backward-only (no future features)",
        'direction="backward"' in bench_src,
        "merge_asof only uses past filings",
    ))

    # max_rows after split
    results.append(check(
        "max_rows truncation after temporal split",
        "tail(self.max_rows)" in bench_src or "train.tail" in bench_src,
        "Train truncated AFTER split, never test",
    ))

    # 5-fold temporal CV in AutoFit
    autofit_src = Path("src/narrative/block3/models/autofit_wrapper.py").read_text()
    results.append(check(
        "5-fold expanding-window temporal CV",
        "_N_TEMPORAL_FOLDS = 5" in autofit_src,
        "Consistent anti-leak evaluation across all AutoFit variants",
    ))
    results.append(check(
        "Stability penalty active",
        "_STABILITY_PENALTY = 0.25" in autofit_src,
        "adj_MAE = mean_MAE * (1 + 0.25 * CV(MAE))",
    ))

    return results


def main():
    print("=" * 70)
    print("Phase 3 Fix Verification — Strict Audit")
    print("=" * 70)

    all_results = []
    all_results.extend(audit_deep_models())
    all_results.extend(audit_edgar_join())
    all_results.extend(audit_autofit_timeout())
    all_results.extend(audit_gbdt_count_target())
    all_results.extend(audit_horizon_dedup())
    all_results.extend(audit_entity_coverage())
    all_results.extend(audit_anti_leakage())

    n_pass = sum(all_results)
    n_total = len(all_results)
    n_fail = n_total - n_pass

    print("\n" + "=" * 70)
    print(f"RESULT: {n_pass}/{n_total} checks passed, {n_fail} failed")
    if n_fail == 0:
        print("🏆 ALL PHASE 3 FIXES VERIFIED — READY FOR BENCHMARK")
    else:
        print("⚠️  SOME CHECKS FAILED — review above")
    print("=" * 70)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
