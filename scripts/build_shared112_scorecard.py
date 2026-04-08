#!/usr/bin/env python3
"""Build a serious-branch scorecard draft from shared112 local compare artifacts."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent

TASK_ORDER = {"task1_outcome": 0, "task2_forecast": 1, "task3_risk_adjust": 2}
TARGET_ORDER = {"is_funded": 0, "funding_raised_usd": 1, "investors_count": 2}
ABLATION_ORDER = {"core_only": 0, "core_edgar": 1, "core_text": 2, "full": 3}

TASK_DISPLAY = {
    "task1_outcome": "task1_outcome",
    "task2_forecast": "task2_forecast",
    "task3_risk_adjust": "task3_risk_adjust",
}
TARGET_DISPLAY = {
    "is_funded": "is_funded",
    "funding_raised_usd": "funding_raised_usd",
    "investors_count": "investors_count",
}

HARDEST_FAMILY_DEFS = [
    {
        "label": "all tasks / investors_count / h1",
        "predicate": lambda case: case["target"] == "investors_count" and int(case["horizon"]) == 1,
    },
    {
        "label": "all tasks / investors_count / h7-h30",
        "predicate": lambda case: case["target"] == "investors_count" and int(case["horizon"]) in {7, 14, 30},
    },
    {
        "label": "all tasks / funding_raised_usd / full / h14-h30",
        "predicate": lambda case: case["target"] == "funding_raised_usd" and case["ablation"] == "full" and int(case["horizon"]) in {14, 30},
    },
    {
        "label": "task1_outcome / is_funded / all horizons",
        "predicate": lambda case: case["task"] == "task1_outcome" and case["target"] == "is_funded",
    },
]

CALIBRATION_BRIER_DELTA = 0.02
CALIBRATION_LOGLOSS_DELTA = 0.10
CALIBRATION_ECE_DELTA = 0.05
CALIBRATION_RATIO_FLOOR = 1.25
DISPERSION_DEVIATION_DELTA = 0.75
RESIDUAL_RATIO_DELTA = 0.75


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--surface-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--candidate-label", default="")
    parser.add_argument("--branch-name", default="")
    parser.add_argument("--branch-type", default="cross_task")
    parser.add_argument("--report-date", default=str(date.today()))
    parser.add_argument("--reference-truth", default="docs/CURRENT_SOURCE_OF_TRUTH.md")
    parser.add_argument(
        "--execution-spec",
        default="docs/references/single_model_true_champion/SINGLE_MODEL_TRUE_CHAMPION_EXECUTION_LOOP_ZH_20260407.md",
    )
    parser.add_argument("--tie-tolerance-pct", type=float, default=0.5)
    parser.add_argument("--catastrophic-ratio", type=float, default=10.0)
    parser.add_argument("--hypothesis", action="append", default=[])
    parser.add_argument("--rejects", action="append", default=[])
    parser.add_argument("--implementation-change", action="append", default=[])
    parser.add_argument("--expected-improve", action="append", default=[])
    parser.add_argument("--guard-rail", action="append", default=[])
    parser.add_argument("--collateral-risk", action="append", default=[])
    parser.add_argument("--mergeable", action="append", default=[])
    parser.add_argument("--rollback", action="append", default=[])
    parser.add_argument("--next-step", action="append", default=[])
    parser.add_argument("--one-line-conclusion", default="")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_surface(surface_json: Optional[Path], results_dir: Optional[Path]) -> Dict[str, Any]:
    candidates: List[Path] = []
    if surface_json is not None:
        candidates.append(surface_json)
    if results_dir is not None:
        candidates.extend([
            results_dir / "surface.json",
            results_dir / "v740_shared112_surface_20260401.json",
        ])
    for path in candidates:
        if path.exists():
            data = _load_json(path)
            if isinstance(data, dict) and isinstance(data.get("cells"), list):
                return data
    raise FileNotFoundError("Unable to locate a shared112 surface manifest. Provide --surface-json or a results dir with surface.json.")


def _load_result_rows(results_dir: Optional[Path]) -> List[Dict[str, Any]]:
    if results_dir is None or not results_dir.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        data = _load_json(path)
        if isinstance(data, dict) and "case_name" in data and "model_label" in data:
            rows.append(data)
    return rows


def _pick_primary_candidate_label(results: List[Dict[str, Any]]) -> str:
    labels: List[str] = []
    seen = set()
    for row in results:
        label = str(row.get("model_label", "")).strip()
        if not label or label.startswith("incumbent__") or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    for preferred in (
        "single_model_mainline",
        "mainline_alpha",
        "v745_evidence_residual",
        "v744_guarded_phase",
        "v743_factorized",
        "v742_unified",
        "v741_lite",
        "v740_alpha",
        "v739",
    ):
        if preferred in seen:
            return preferred
    return labels[0] if labels else "candidate"


def _case_sort_key(case: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        TASK_ORDER.get(str(case["task"]), 99),
        TARGET_ORDER.get(str(case["target"]), 99),
        int(case["horizon"]),
        ABLATION_ORDER.get(str(case["ablation"]), 99),
    )


def _fmt_metric(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}%"


def _fmt_signed_metric(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:+,.2f}"
    return f"{value:+.4f}"


def _fmt_multiple(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


def _median(values: Iterable[float]) -> Optional[float]:
    data = list(values)
    return statistics.median(data) if data else None


def _mean(values: Iterable[float]) -> Optional[float]:
    data = list(values)
    return statistics.mean(data) if data else None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(part: int, total: int) -> str:
    if total <= 0:
        return f"{part}/{total}"
    return f"{part}/{total} ({part / total * 100.0:.1f}%)"


def _short_case(case: Dict[str, Any]) -> str:
    task = case["task"].replace("task", "t")
    target = case["target"].replace("funding_raised_usd", "funding").replace("investors_count", "investors")
    return f"{task}/{case['ablation']}/{target}/h{int(case['horizon'])}"


def _classify_case(candidate: Optional[Dict[str, Any]], incumbent: Optional[Dict[str, Any]], tie_tol_pct: float) -> str:
    if not candidate or not incumbent:
        return "incomplete"
    if candidate.get("error") or incumbent.get("error"):
        return "error"
    candidate_mae = candidate.get("metrics", {}).get("mae")
    incumbent_mae = incumbent.get("metrics", {}).get("mae")
    if candidate_mae is None or incumbent_mae is None:
        return "error"
    denom = max(abs(float(incumbent_mae)), 1e-9)
    rel = abs(float(candidate_mae) - float(incumbent_mae)) / denom * 100.0
    if rel <= tie_tol_pct:
        return "tie"
    return "win" if float(candidate_mae) < float(incumbent_mae) else "loss"


def _severity(count: int) -> str:
    if count <= 0:
        return "-"
    if count >= 5:
        return "high"
    if count >= 2:
        return "medium"
    return "low"


def _group_rows_by_case(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row.get("case_name", ""))][str(row.get("model_label", ""))] = row
    return grouped


def _get_row_metric(row: Optional[Dict[str, Any]], metric_name: str) -> Optional[float]:
    if not row:
        return None
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    return _safe_float(metrics.get(metric_name))


def _get_row_value(row: Optional[Dict[str, Any]], field_name: str) -> Optional[float]:
    if not row:
        return None
    return _safe_float(row.get(field_name))


def _is_severe_binary_calibration_case(
    candidate: Optional[Dict[str, Any]],
    incumbent: Optional[Dict[str, Any]],
    target: str,
) -> bool:
    if target != "is_funded" or candidate is None or incumbent is None:
        return False
    if candidate.get("error") or incumbent.get("error"):
        return True
    if bool(candidate.get("constant_prediction")):
        return True

    checks = 0
    regressions = 0
    for metric_name, abs_delta in (
        ("brier", CALIBRATION_BRIER_DELTA),
        ("logloss", CALIBRATION_LOGLOSS_DELTA),
        ("ece", CALIBRATION_ECE_DELTA),
    ):
        candidate_value = _get_row_metric(candidate, metric_name)
        incumbent_value = _get_row_metric(incumbent, metric_name)
        if candidate_value is None or incumbent_value is None:
            continue
        checks += 1
        if (
            candidate_value > incumbent_value + abs_delta
            and candidate_value > max(incumbent_value, 1e-9) * CALIBRATION_RATIO_FLOOR
        ):
            regressions += 1

    candidate_prob_std = _get_row_value(candidate, "binary_prob_std")
    incumbent_prob_std = _get_row_value(incumbent, "binary_prob_std")
    if candidate_prob_std is not None and incumbent_prob_std is not None:
        checks += 1
        if candidate_prob_std < 0.01 and incumbent_prob_std >= 0.05:
            regressions += 1

    return bool(regressions) if checks else False


def _is_severe_dispersion_drift_case(
    candidate: Optional[Dict[str, Any]],
    incumbent: Optional[Dict[str, Any]],
    target: str,
) -> bool:
    if target == "is_funded" or candidate is None or incumbent is None:
        return False
    candidate_ratio = _get_row_value(candidate, "prediction_to_truth_std_ratio")
    incumbent_ratio = _get_row_value(incumbent, "prediction_to_truth_std_ratio")
    if candidate_ratio is not None and incumbent_ratio is not None:
        candidate_dev = abs(candidate_ratio - 1.0)
        incumbent_dev = abs(incumbent_ratio - 1.0)
        if candidate_ratio < 0.25 <= incumbent_ratio:
            return True
        if candidate_ratio > 4.0 and incumbent_ratio < 2.5:
            return True
        if candidate_dev > incumbent_dev + DISPERSION_DEVIATION_DELTA and candidate_dev > 1.0:
            return True

    candidate_residual_ratio = _get_row_value(candidate, "residual_to_truth_std_ratio")
    incumbent_residual_ratio = _get_row_value(incumbent, "residual_to_truth_std_ratio")
    if candidate_residual_ratio is not None and incumbent_residual_ratio is not None:
        if (
            candidate_residual_ratio > incumbent_residual_ratio + RESIDUAL_RATIO_DELTA
            and candidate_residual_ratio > 2.0
        ):
            return True

    return False


def _summarize_metric_pairs(pairs: List[Tuple[float, float]]) -> str:
    if not pairs:
        return "-"
    deltas = [candidate - incumbent for candidate, incumbent in pairs]
    ratios = [candidate / max(abs(incumbent), 1e-9) for candidate, incumbent in pairs]
    return f"comp={len(pairs)}; med_delta={_fmt_signed_metric(_median(deltas))}; med_ratio={_fmt_multiple(_median(ratios))}"


def _summarize_dispersion_pairs(
    pairs: List[Tuple[float, float]],
    use_distance_to_one: bool,
) -> str:
    if not pairs:
        return "-"
    if use_distance_to_one:
        deltas = [abs(candidate - 1.0) - abs(incumbent - 1.0) for candidate, incumbent in pairs]
    else:
        deltas = [candidate - incumbent for candidate, incumbent in pairs]
    candidate_values = [candidate for candidate, _ in pairs]
    return f"comp={len(pairs)}; med_delta={_fmt_signed_metric(_median(deltas))}; cand_med={_fmt_multiple(_median(candidate_values))}"


def _build_cases(
    manifest: Dict[str, Any],
    rows: List[Dict[str, Any]],
    candidate_label: str,
    tie_tol_pct: float,
    catastrophic_ratio: float,
) -> List[Dict[str, Any]]:
    by_case = _group_rows_by_case(rows)
    cases: List[Dict[str, Any]] = []
    for cell in sorted(manifest.get("cells", []), key=_case_sort_key):
        case_rows = by_case.get(cell["name"], {})
        candidate = case_rows.get(candidate_label)
        incumbent = case_rows.get(f"incumbent__{cell['incumbent_model']}")
        if incumbent is None:
            incumbent = next((row for label, row in case_rows.items() if label.startswith("incumbent__")), None)
        outcome = _classify_case(candidate, incumbent, tie_tol_pct)
        candidate_mae = candidate.get("metrics", {}).get("mae") if candidate else None
        incumbent_mae = incumbent.get("metrics", {}).get("mae") if incumbent else None
        gap_pct = None
        if candidate_mae is not None and incumbent_mae is not None:
            gap_pct = (float(candidate_mae) / max(abs(float(incumbent_mae)), 1e-9) - 1.0) * 100.0
        blow_up = bool(
            gap_pct is not None and candidate_mae is not None and incumbent_mae is not None and float(candidate_mae) >= max(float(incumbent_mae), 1e-9) * catastrophic_ratio
        )
        constant_prediction = bool(candidate.get("constant_prediction")) if candidate else False
        error_text = None
        if candidate and candidate.get("error"):
            error_text = str(candidate.get("error"))
        elif incumbent and incumbent.get("error"):
            error_text = str(incumbent.get("error"))
        cases.append(
            {
                "name": cell["name"],
                "task": cell["task"],
                "ablation": cell["ablation"],
                "target": cell["target"],
                "horizon": int(cell["horizon"]),
                "incumbent_model": cell["incumbent_model"],
                "incumbent_benchmark_mae": float(cell["incumbent_benchmark_mae"]),
                "candidate": candidate,
                "incumbent": incumbent,
                "candidate_mae": float(candidate_mae) if candidate_mae is not None else None,
                "incumbent_local_mae": float(incumbent_mae) if incumbent_mae is not None else None,
                "gap_pct": float(gap_pct) if gap_pct is not None else None,
                "outcome": outcome,
                "constant_prediction": constant_prediction,
                "binary_calibration_issue": _is_severe_binary_calibration_case(candidate, incumbent, str(cell["target"])),
                "dispersion_drift": _is_severe_dispersion_drift_case(candidate, incumbent, str(cell["target"])),
                "error": error_text,
                "blow_up": blow_up,
                "selected_model": candidate.get("selected_model") if candidate else None,
                "resolved_model_name": candidate.get("resolved_model_name") if candidate else None,
            }
        )
    return cases


def _summarize_cases(cases: List[Dict[str, Any]], predicate: Callable[[Dict[str, Any]], bool]) -> Dict[str, Any]:
    selected = [case for case in cases if predicate(case)]
    comparable = [case for case in selected if case["candidate_mae"] is not None and case["incumbent_local_mae"] is not None and not case["error"]]
    wins = sum(1 for case in selected if case["outcome"] == "win")
    ties = sum(1 for case in selected if case["outcome"] == "tie")
    losses = sum(1 for case in selected if case["outcome"] == "loss")
    incomplete = sum(1 for case in selected if case["outcome"] == "incomplete")
    errors = sum(1 for case in selected if case["outcome"] == "error")
    catastrophic = [
        case
        for case in selected
        if case["blow_up"]
        or case["constant_prediction"]
        or case["error"]
        or case.get("binary_calibration_issue")
        or case.get("dispersion_drift")
    ]
    gaps = [float(case["gap_pct"]) for case in comparable if case["gap_pct"] is not None]
    return {
        "cells": selected,
        "total": len(selected),
        "comparable": len(comparable),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "non_loss": wins + ties,
        "incomplete": incomplete,
        "errors": errors,
        "catastrophic": len(catastrophic),
        "median_gap_pct": _median(gaps),
        "mean_gap_pct": _mean(gaps),
        "best_cases": sorted(comparable, key=lambda case: (case["gap_pct"], case["name"]))[:3],
        "worst_cases": sorted(comparable, key=lambda case: (-case["gap_pct"], case["name"]))[:3],
        "catastrophic_cases": catastrophic[:5],
    }


def _judgement(summary: Dict[str, Any]) -> str:
    if summary["comparable"] == 0:
        return "待补"
    if summary["catastrophic"] > 0 and summary["losses"] >= summary["non_loss"]:
        return "退化"
    if summary["non_loss"] == summary["comparable"] and (summary["median_gap_pct"] or 0.0) <= 0:
        return "前进"
    if summary["non_loss"] >= summary["losses"]:
        return "混合"
    return "退化"


def _describe_case_list(cases: List[Dict[str, Any]], positive: bool) -> str:
    if not cases:
        return "无"
    filtered = []
    for case in cases:
        gap = case.get("gap_pct")
        if gap is None:
            continue
        if positive and gap <= 0:
            filtered.append(case)
        if not positive and gap > 0:
            filtered.append(case)
    if not filtered:
        return "无清晰自动改善" if positive else "未见自动退化"
    chosen = filtered
    return "；".join(f"{_short_case(case)} ({_fmt_pct(case.get('gap_pct'))})" for case in chosen[:2])


def _hardest_family_rows(cases: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for definition in HARDEST_FAMILY_DEFS:
        summary = _summarize_cases(cases, definition["predicate"])
        champion_counts = Counter(case["incumbent_model"] for case in summary["cells"])
        champions = ", ".join(f"{model}x{count}" for model, count in champion_counts.most_common(3)) or "-"
        verdict = _judgement(summary)
        if verdict == "前进":
            improvement_type = "geometry-level gain"
        elif verdict == "混合":
            improvement_type = "local gain but guard still mixed"
        elif verdict == "待补":
            improvement_type = "pending"
        else:
            improvement_type = "no clear gain"
        residual_bits: List[str] = []
        if summary["catastrophic"]:
            residual_bits.append(f"catastrophic={summary['catastrophic']}")
        if summary["incomplete"]:
            residual_bits.append(f"incomplete={summary['incomplete']}")
        residual_risk = ", ".join(residual_bits) if residual_bits else "none"
        rows.append(
            {
                "label": definition["label"],
                "baseline": f"champions={champions}; bench_med={_fmt_metric(_median(case['incumbent_benchmark_mae'] for case in summary['cells']))}",
                "current": f"comp={summary['comparable']}/{summary['total']}; W/T/L={summary['wins']}/{summary['ties']}/{summary['losses']}; med_gap={_fmt_pct(summary['median_gap_pct'])}",
                "improvement_type": improvement_type,
                "residual_risk": residual_risk,
                "verdict": verdict,
            }
        )
    return rows


def _source_regime_rows(cases: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for ablation in sorted(ABLATION_ORDER, key=lambda key: ABLATION_ORDER[key]):
        lane_summaries = {}
        lane_medians = {}
        for target in sorted(TARGET_ORDER, key=lambda key: TARGET_ORDER[key]):
            summary = _summarize_cases(cases, lambda case, ablation=ablation, target=target: case["ablation"] == ablation and case["target"] == target)
            lane_summaries[target] = f"W/T/L={summary['wins']}/{summary['ties']}/{summary['losses']}; med={_fmt_pct(summary['median_gap_pct'])}; comp={summary['comparable']}/{summary['total']}"
            lane_medians[target] = float(summary["median_gap_pct"]) if summary["median_gap_pct"] is not None else 0.0
        worst_target = max(lane_medians, key=lane_medians.get)
        overall = _summarize_cases(cases, lambda case, ablation=ablation: case["ablation"] == ablation)
        rows.append(
            {
                "ablation": ablation,
                "binary": lane_summaries["is_funded"],
                "funding": lane_summaries["funding_raised_usd"],
                "investors": lane_summaries["investors_count"],
                "major_problem": f"{worst_target} worst median gap { _fmt_pct(lane_medians[worst_target]) }",
                "verdict": _judgement(overall),
            }
        )
    return rows


def _binary_calibration_rows(cases: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for ablation in sorted(ABLATION_ORDER, key=lambda key: ABLATION_ORDER[key]):
        selected = [case for case in cases if case["ablation"] == ablation and case["target"] == "is_funded"]
        brier_pairs: List[Tuple[float, float]] = []
        logloss_pairs: List[Tuple[float, float]] = []
        ece_pairs: List[Tuple[float, float]] = []
        for case in selected:
            candidate = case.get("candidate")
            incumbent = case.get("incumbent")
            candidate_brier = _get_row_metric(candidate, "brier")
            incumbent_brier = _get_row_metric(incumbent, "brier")
            if candidate_brier is not None and incumbent_brier is not None:
                brier_pairs.append((candidate_brier, incumbent_brier))
            candidate_logloss = _get_row_metric(candidate, "logloss")
            incumbent_logloss = _get_row_metric(incumbent, "logloss")
            if candidate_logloss is not None and incumbent_logloss is not None:
                logloss_pairs.append((candidate_logloss, incumbent_logloss))
            candidate_ece = _get_row_metric(candidate, "ece")
            incumbent_ece = _get_row_metric(incumbent, "ece")
            if candidate_ece is not None and incumbent_ece is not None:
                ece_pairs.append((candidate_ece, incumbent_ece))
        severe_cases = [case for case in selected if case.get("binary_calibration_issue")]
        comparable = max(len(brier_pairs), len(logloss_pairs), len(ece_pairs))
        if comparable == 0:
            verdict = "待补"
        else:
            brier_delta = _median([candidate - incumbent for candidate, incumbent in brier_pairs])
            logloss_delta = _median([candidate - incumbent for candidate, incumbent in logloss_pairs])
            ece_delta = _median([candidate - incumbent for candidate, incumbent in ece_pairs])
            nonpositive = [delta for delta in (brier_delta, logloss_delta, ece_delta) if delta is not None and delta <= 0.0]
            positive = [delta for delta in (brier_delta, logloss_delta, ece_delta) if delta is not None and delta > 0.0]
            if severe_cases or len(positive) >= 2:
                verdict = "drift"
            elif len(nonpositive) >= 2:
                verdict = "stable"
            else:
                verdict = "mixed"
        rows.append(
            {
                "ablation": ablation,
                "coverage": _ratio(comparable, len(selected)),
                "brier": _summarize_metric_pairs(brier_pairs),
                "logloss": _summarize_metric_pairs(logloss_pairs),
                "ece": _summarize_metric_pairs(ece_pairs),
                "severe_cases": "；".join(_short_case(case) for case in severe_cases[:2]) or "-",
                "verdict": verdict,
            }
        )
    return rows


def _dispersion_rows(cases: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for ablation in sorted(ABLATION_ORDER, key=lambda key: ABLATION_ORDER[key]):
        for target in ("funding_raised_usd", "investors_count"):
            selected = [case for case in cases if case["ablation"] == ablation and case["target"] == target]
            prediction_pairs: List[Tuple[float, float]] = []
            residual_pairs: List[Tuple[float, float]] = []
            for case in selected:
                candidate = case.get("candidate")
                incumbent = case.get("incumbent")
                candidate_ratio = _get_row_value(candidate, "prediction_to_truth_std_ratio")
                incumbent_ratio = _get_row_value(incumbent, "prediction_to_truth_std_ratio")
                if candidate_ratio is not None and incumbent_ratio is not None:
                    prediction_pairs.append((candidate_ratio, incumbent_ratio))
                candidate_residual = _get_row_value(candidate, "residual_to_truth_std_ratio")
                incumbent_residual = _get_row_value(incumbent, "residual_to_truth_std_ratio")
                if candidate_residual is not None and incumbent_residual is not None:
                    residual_pairs.append((candidate_residual, incumbent_residual))
            severe_cases = [case for case in selected if case.get("dispersion_drift")]
            comparable = max(len(prediction_pairs), len(residual_pairs))
            if comparable == 0:
                verdict = "待补"
            else:
                pred_delta = _median([
                    abs(candidate - 1.0) - abs(incumbent - 1.0)
                    for candidate, incumbent in prediction_pairs
                ])
                residual_delta = _median([
                    candidate - incumbent
                    for candidate, incumbent in residual_pairs
                ])
                stable_pred = pred_delta is not None and pred_delta <= 0.0
                stable_residual = residual_delta is not None and residual_delta <= 0.0
                if severe_cases or (pred_delta is not None and pred_delta > 0.0 and residual_delta is not None and residual_delta > 0.0):
                    verdict = "drift"
                elif stable_pred and stable_residual:
                    verdict = "stable"
                else:
                    verdict = "mixed"
            rows.append(
                {
                    "ablation": ablation,
                    "target": target,
                    "coverage": _ratio(comparable, len(selected)),
                    "prediction_dispersion": _summarize_dispersion_pairs(prediction_pairs, use_distance_to_one=True),
                    "residual_scale": _summarize_dispersion_pairs(residual_pairs, use_distance_to_one=False),
                    "severe_cases": "；".join(_short_case(case) for case in severe_cases[:2]) or "-",
                    "verdict": verdict,
                }
            )
    return rows


def _catastrophic_rows(cases: List[Dict[str, Any]], candidate_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    blow_up_cases = [case for case in cases if case["blow_up"]]
    binary_cases = [case for case in cases if case.get("binary_calibration_issue")]
    source_rich_cases = [
        case
        for case in cases
        if case["ablation"] in {"core_edgar", "core_text", "full"}
        and (case["blow_up"] or case["constant_prediction"] or case["error"] or case.get("dispersion_drift"))
    ]
    constant_cases = [case for case in cases if case["constant_prediction"]]
    selected_models = {str(row.get("selected_model")) for row in candidate_rows if row.get("selected_model")}
    resolved_models = {str(row.get("resolved_model_name")) for row in candidate_rows if row.get("resolved_model_name")}
    hidden_multi_model = bool(selected_models) or len(resolved_models) > 1
    return [
        {
            "signal": "orders-of-magnitude blow-up",
            "present": "是" if blow_up_cases else "否",
            "cells": "；".join(_short_case(case) for case in blow_up_cases[:3]) or "-",
            "severity": _severity(len(blow_up_cases)),
            "note": "自动规则：candidate local MAE >= incumbent local MAE x catastrophic_ratio",
        },
        {
            "signal": "binary calibration collapse",
            "present": "是" if binary_cases else "否",
            "cells": "；".join(_short_case(case) for case in binary_cases[:3]) or "-",
            "severity": _severity(len(binary_cases)),
            "note": "优先使用 shared112 JSON 内的 Brier / logloss / ECE / prob-std；缺失时才退回 constant/error proxy",
        },
        {
            "signal": "source-rich severe drift",
            "present": "是" if source_rich_cases else "否",
            "cells": "；".join(_short_case(case) for case in source_rich_cases[:3]) or "-",
            "severity": _severity(len(source_rich_cases)),
            "note": "监控 core_edgar/core_text/full 上的 severe local failures 与 dispersion drift",
        },
        {
            "signal": "constant prediction return",
            "present": "是" if constant_cases else "否",
            "cells": "；".join(_short_case(case) for case in constant_cases[:3]) or "-",
            "severity": _severity(len(constant_cases)),
            "note": "直接读取 shared112 local compare JSON 的 constant_prediction 标志",
        },
        {
            "signal": "hidden multi-model dependency",
            "present": "是" if hidden_multi_model else "否",
            "cells": ", ".join(sorted(selected_models or resolved_models)) or "-",
            "severity": _severity(1 if hidden_multi_model else 0),
            "note": "自动规则：出现 selected_model 或 candidate resolved model 不止一种",
        },
    ]


def _coverage_conclusion(total_summary: Dict[str, Any]) -> str:
    if total_summary["comparable"] == 0:
        return "当前还没有可比 local compare 单元；只能生成 skeleton scorecard。"
    return (
        f"shared112 可比覆盖为 { _ratio(total_summary['comparable'], total_summary['total']) }，"
        f"另有 incomplete={total_summary['incomplete']}、error={total_summary['errors']}；"
        "本记分卡仍然是同 slice 的 research-only 读法。"
    )


def _hardest_conclusion(hardest_rows: List[Dict[str, str]]) -> str:
    verdicts = Counter(row["verdict"] for row in hardest_rows)
    return "；".join(f"{label}={count}" for label, count in verdicts.items()) if verdicts else "-"


def _balance_conclusion(task_rows: List[Dict[str, Any]]) -> str:
    return "；".join(f"{row['task']}={row['judgement']}" for row in task_rows)


def _architecture_conclusion(catastrophic_rows: List[Dict[str, str]]) -> str:
    multi_model = next(row for row in catastrophic_rows if row["signal"] == "hidden multi-model dependency")
    if multi_model["present"] == "是":
        return f"自动审计提示存在潜在 runtime 多模型依赖：{multi_model['cells']}。"
    return "自动审计未见显式 runtime 多模型依赖；仍应结合代码 review 继续复核。"


def _overall_verdict(total_summary: Dict[str, Any], hardest_rows: List[Dict[str, str]]) -> Tuple[str, str]:
    comparable_ratio = total_summary["comparable"] / total_summary["total"] if total_summary["total"] else 0.0
    hardest_forward = sum(1 for row in hardest_rows if row["verdict"] in {"前进", "混合"})
    if total_summary["comparable"] == 0:
        return "neutral", "还没有形成足够可比单元，只能保留中性裁决。"
    if total_summary["catastrophic"] > 0 and total_summary["losses"] >= total_summary["non_loss"]:
        return "reject", "出现 catastrophic signal 且 losses 不低于 non-loss。"
    if comparable_ratio < 0.5:
        if total_summary["non_loss"] > total_summary["losses"] and hardest_forward > 0 and total_summary["catastrophic"] == 0:
            return "partial forward", "局部 family 有进展，但覆盖仍不足以宣布主线前进。"
        return "neutral", "局部证据不足，先保守维持中性裁决。"
    if comparable_ratio < 1.0:
        if total_summary["non_loss"] > total_summary["losses"] and hardest_forward >= 2 and total_summary["catastrophic"] == 0:
            return "partial forward", "已有跨 family 进展，但仍不是 full-surface 结论。"
        if total_summary["losses"] > total_summary["non_loss"]:
            return "reject", "局部覆盖已足够显示 regression 主导。"
        return "neutral", "仍需补完覆盖与 hardest family 证据。"
    if total_summary["non_loss"] >= total_summary["losses"] * 2 and hardest_forward >= 3 and total_summary["catastrophic"] == 0:
        return "mainline forward", "shared112 全面上 non-loss 占优，hardest family 也大体改善。"
    if total_summary["non_loss"] > total_summary["losses"] and hardest_forward >= 2 and total_summary["catastrophic"] == 0:
        return "partial forward", "全覆盖下有真实改善，但 hardest family 还没有全部收口。"
    if total_summary["losses"] > total_summary["non_loss"] or total_summary["catastrophic"] > 0:
        return "reject", "全覆盖下 regression 或 catastrophic 仍然主导。"
    return "neutral", "结果混合，尚不能宣告前进或拒绝。"


def _auto_lines(values: List[str], fallback: str) -> List[str]:
    return values if values else [fallback]


def _bullet_lines(values: List[str]) -> List[str]:
    return [f"- {value}" for value in values]


def main() -> int:
    args = _parse_args()
    manifest = _load_surface(args.surface_json, args.results_dir)
    result_rows = _load_result_rows(args.results_dir)
    candidate_label = args.candidate_label or _pick_primary_candidate_label(result_rows)
    branch_name = args.branch_name or candidate_label

    cases = _build_cases(manifest, result_rows, candidate_label, args.tie_tolerance_pct, args.catastrophic_ratio)
    candidate_rows = [row for row in result_rows if str(row.get("model_label", "")) == candidate_label]
    total_summary = _summarize_cases(cases, lambda case: True)

    task_rows = []
    for task in sorted(TASK_ORDER, key=lambda key: TASK_ORDER[key]):
        summary = _summarize_cases(cases, lambda case, task=task: case["task"] == task)
        task_rows.append(
            {
                "task": task,
                "summary": summary,
                "improve": _describe_case_list(summary["best_cases"], positive=True),
                "regress": _describe_case_list(summary["worst_cases"], positive=False),
                "judgement": _judgement(summary),
            }
        )

    investors_hard = _summarize_cases(cases, lambda case: case["target"] == "investors_count")
    funding_hard = _summarize_cases(cases, lambda case: case["target"] == "funding_raised_usd" and case["ablation"] == "full" and int(case["horizon"]) in {14, 30})
    binary_hard = _summarize_cases(cases, lambda case: case["task"] == "task1_outcome" and case["target"] == "is_funded")

    target_rows = []
    for target in sorted(TARGET_ORDER, key=lambda key: TARGET_ORDER[key]):
        summary = _summarize_cases(cases, lambda case, target=target: case["target"] == target)
        if target == "investors_count":
            hard_text = _judgement(investors_hard)
        elif target == "funding_raised_usd":
            hard_text = _judgement(funding_hard)
        else:
            hard_text = _judgement(binary_hard)
        guard_hurt = "是" if summary["catastrophic"] > 0 else "否"
        target_rows.append(
            {
                "target": target,
                "summary": summary,
                "hard": hard_text,
                "guard_hurt": guard_hurt,
                "judgement": _judgement(summary),
            }
        )

    hardest_rows = _hardest_family_rows(cases)
    source_rows = _source_regime_rows(cases)
    binary_calibration_rows = _binary_calibration_rows(cases)
    dispersion_rows = _dispersion_rows(cases)
    catastrophic_rows = _catastrophic_rows(cases, candidate_rows)
    verdict, verdict_reason = _overall_verdict(total_summary, hardest_rows)

    best_auto = _describe_case_list(total_summary["best_cases"], positive=True)
    worst_auto = _describe_case_list(total_summary["worst_cases"], positive=False)
    default_expected_improve = (
        "AUTO: 当前没有自动识别到 non-loss family；下一步应先把注意力放在 regression 修复上。"
        if best_auto == "无清晰自动改善"
        else f"AUTO: 先重点看 {best_auto} 是否代表真实 family 改善。"
    )
    default_mergeable = (
        "AUTO: 当前没有自动识别到可直接并入主线的 non-loss family。"
        if best_auto == "无清晰自动改善"
        else f"AUTO: 优先保留 {best_auto} 对应的结构增益。"
    )
    default_collateral = (
        "AUTO: 当前还没有自动识别到明确退化。"
        if worst_auto == "未见自动退化"
        else f"AUTO: 当前最值得警惕的是 {worst_auto} 暴露出的 family regression。"
    )
    default_rollback = (
        "AUTO: 当前没有自动识别到必须立即隔离的退化路径。"
        if worst_auto == "未见自动退化"
        else f"AUTO: 先隔离 {worst_auto} 暴露出来的退化路径。"
    )

    mergeable = args.mergeable or [
        default_mergeable
    ]
    rollback = args.rollback or [
        default_rollback
    ]
    next_steps = args.next_step or [
        "AUTO: 只允许继续打 hardest family 与 catastrophic signal 直接相关的 lane-private 补丁。"
    ]
    one_line = args.one_line_conclusion or f"{branch_name} 当前裁决为 {verdict}；原因是 {verdict_reason}"

    lines: List[str] = [
        f"# {branch_name} 单模型真冠军记分卡",
        "",
        f"> 日期：{args.report_date}",
        f"> 分支：`{branch_name}`",
        f"> 类型：`{args.branch_type}`",
        f"> 参考真相包：`{args.reference_truth}`",
        f"> 参考执行规范：`{args.execution_spec}`",
        "> 生成方式：`scripts/build_shared112_scorecard.py`",
        "> 说明：这是 shared112 local compare 的自动草稿；local 数字只用于同 slice 比较，不覆盖 canonical benchmark truth",
        "",
        "## 0. 本轮结构假设",
        "",
    ]
    lines.extend(_bullet_lines(_auto_lines(args.hypothesis, "AUTO: 需要手工补充本轮核心结构假设。")))
    lines.extend(_bullet_lines(_auto_lines(args.rejects, "AUTO: 需要手工补充本轮不再接受的旧解释。")))
    lines.extend(_bullet_lines(_auto_lines(args.implementation_change, "AUTO: 需要手工补充本轮最重要的实现变化。")))

    lines.extend(["", "## 1. 预期影响范围", "", "### 1.1 预计首先改善的 family", ""])
    lines.extend(_bullet_lines(_auto_lines(args.expected_improve, default_expected_improve)))
    lines.extend(["", "### 1.2 本轮 guard rails", ""])
    lines.extend(_bullet_lines(_auto_lines(args.guard_rail, "AUTO: 至少确保 binary lane、investors hardest cells、source-rich family 不新增 catastrophic signal。")))
    lines.extend(["", "### 1.3 本轮最可能的 collateral damage", ""])
    lines.extend(_bullet_lines(_auto_lines(args.collateral_risk, default_collateral)))

    lines.extend([
        "",
        "## 2. shared112 总览",
        "",
        "| 维度 | wins | ties | losses | non-loss | median gap | mean gap | catastrophic count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| total | {total_summary['wins']} | {total_summary['ties']} | {total_summary['losses']} | {total_summary['non_loss']} | {_fmt_pct(total_summary['median_gap_pct'])} | {_fmt_pct(total_summary['mean_gap_pct'])} | {total_summary['catastrophic']} |",
        "",
        f"- 可比覆盖：`{_ratio(total_summary['comparable'], total_summary['total'])}`",
        f"- incomplete：`{total_summary['incomplete']}`",
        f"- errors：`{total_summary['errors']}`",
        f"- primary candidate：`{candidate_label}`",
    ])

    lines.extend(["", "## 3. 按任务家族记分", "", "| task family | cells | wins | ties | losses | non-loss | 主要改善 | 主要退化 | 判定 |", "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"])
    for row in task_rows:
        summary = row["summary"]
        lines.append(
            f"| `{TASK_DISPLAY[row['task']]}` | `{summary['total']}` | {summary['wins']} | {summary['ties']} | {summary['losses']} | {summary['non_loss']} | {row['improve']} | {row['regress']} | {row['judgement']} |"
        )

    lines.extend(["", "## 4. 按 target lane 记分", "", "| target lane | cells | wins | ties | losses | non-loss | hardest family 是否改善 | guard 是否受损 | 判定 |", "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"])
    for row in target_rows:
        summary = row["summary"]
        lines.append(
            f"| `{TARGET_DISPLAY[row['target']]}` | `{summary['total']}` | {summary['wins']} | {summary['ties']} | {summary['losses']} | {summary['non_loss']} | {row['hard']} | {row['guard_hurt']} | {row['judgement']} |"
        )

    lines.extend(["", "## 5. Hardest Family 记分", "", "| hardest family | baseline read | current read | improvement type | residual risk | verdict |", "| --- | --- | --- | --- | --- | --- |"])
    for row in hardest_rows:
        lines.append(
            f"| `{row['label']}` | {row['baseline']} | {row['current']} | {row['improvement_type']} | {row['residual_risk']} | {row['verdict']} |"
        )

    lines.extend(["", "## 6. Source-Regime 稳定性", "", "| source regime | binary lane | funding lane | investors lane | 主要问题 | 判定 |", "| --- | --- | --- | --- | --- | --- |"])
    for row in source_rows:
        lines.append(
            f"| `{row['ablation']}` | {row['binary']} | {row['funding']} | {row['investors']} | {row['major_problem']} | {row['verdict']} |"
        )

    lines.extend([
        "",
        "## 7. Calibration / Uncertainty Audit",
        "",
        "### 7.1 Binary calibration",
        "",
        "| source regime | coverage | Brier | LogLoss | ECE | severe cases | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in binary_calibration_rows:
        lines.append(
            f"| `{row['ablation']}` | `{row['coverage']}` | {row['brier']} | {row['logloss']} | {row['ece']} | {row['severe_cases']} | {row['verdict']} |"
        )

    lines.extend([
        "",
        "### 7.2 Regression dispersion",
        "",
        "| source regime | target | coverage | prediction dispersion | residual scale | severe cases | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in dispersion_rows:
        lines.append(
            f"| `{row['ablation']}` | `{row['target']}` | `{row['coverage']}` | {row['prediction_dispersion']} | {row['residual_scale']} | {row['severe_cases']} | {row['verdict']} |"
        )

    lines.extend(["", "## 8. Catastrophic Signals", "", "| 信号 | 是否出现 | 具体 family / cell | 严重性 | 备注 |", "| --- | --- | --- | --- | --- |"])
    for row in catastrophic_rows:
        lines.append(
            f"| {row['signal']} | {row['present']} | {row['cells']} | {row['severity']} | {row['note']} |"
        )

    lines.extend([
        "",
        "## 9. Whole-Model Champion Contract",
        "",
        "### 9.1 覆盖面",
        "",
        f"- `S_coverage` 结论：{_coverage_conclusion(total_summary)}",
        "",
        "### 9.2 hardest cells",
        "",
        f"- `S_hard-cells` 结论：{_hardest_conclusion(hardest_rows)}",
        "",
        "### 9.3 跨任务平衡",
        "",
        f"- `S_balance` 结论：{_balance_conclusion(task_rows)}",
        "",
        "### 9.4 架构纯度",
        "",
        f"- `P_architecture` 结论：{_architecture_conclusion(catastrophic_rows)}",
        "",
        "### 9.5 综合判定",
        "",
        f"- 本轮是：`{verdict}`",
        f"- 原因：{verdict_reason}",
    ])

    lines.extend(["", "## 10. Merge / Reject Decision", "", "### 10.1 可以并入主线的部分", ""])
    lines.extend(_bullet_lines(mergeable))
    lines.extend(["", "### 10.2 必须回滚或隔离的部分", ""])
    lines.extend(_bullet_lines(rollback))
    lines.extend(["", "### 10.3 下一轮只允许做的事", ""])
    lines.extend(_bullet_lines(next_steps))
    lines.extend(["", "## 11. 一句话结论", "", f"> {one_line}"])

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "branch_name": branch_name,
                    "candidate_label": candidate_label,
                    "total_summary": total_summary,
                    "task_rows": task_rows,
                    "target_rows": target_rows,
                    "hardest_rows": hardest_rows,
                    "source_rows": source_rows,
                    "binary_calibration_rows": binary_calibration_rows,
                    "dispersion_rows": dispersion_rows,
                    "catastrophic_rows": catastrophic_rows,
                    "verdict": verdict,
                    "verdict_reason": verdict_reason,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(f"[shared112-scorecard] wrote markdown to {args.output_md}")
    if args.output_json is not None:
        print(f"[shared112-scorecard] wrote json to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())