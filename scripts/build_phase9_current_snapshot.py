#!/usr/bin/env python3
"""Build a current Phase 9 / V739 fact snapshot from live artifacts."""

import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH = ROOT / "runs" / "benchmarks" / "block3_phase9_fair"
DEFAULT_FILTERED = DEFAULT_BENCH / "all_results.csv"
DEFAULT_TEXT_DIR = ROOT / "runs" / "text_embeddings"
DEFAULT_JSON = ROOT / "docs" / "benchmarks" / "phase9_current_snapshot.json"
DEFAULT_MD = ROOT / "docs" / "benchmarks" / "phase9_current_snapshot.md"
FULL_CONDITION_COUNT = 160


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _condition_key(row: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    return (
        row.get("task"),
        row.get("ablation"),
        row.get("target"),
        row.get("horizon"),
    )


def _scan_raw_metrics(bench_dir: Path) -> Dict[str, Any]:
    metrics_files = sorted(bench_dir.rglob("metrics.json"))
    rows: List[Dict[str, Any]] = []
    model_conditions: Dict[str, set] = defaultdict(set)
    category_counter: Counter[str] = Counter()
    model_counter: Counter[str] = Counter()

    for mf in metrics_files:
        payload = _read_json(mf, default=[])
        if isinstance(payload, dict):
            payload = [payload]
        for row in payload:
            if not isinstance(row, dict):
                continue
            row["_source"] = str(mf.relative_to(bench_dir))
            rows.append(row)
            model = str(row.get("model_name", "UNKNOWN"))
            category = str(row.get("category", "unknown"))
            model_counter[model] += 1
            category_counter[category] += 1
            model_conditions[model].add(_condition_key(row))

    complete_models = sorted(k for k, v in model_conditions.items() if len(v) >= FULL_CONDITION_COUNT)
    partial_models = sorted(k for k, v in model_conditions.items() if 0 < len(v) < FULL_CONDITION_COUNT)
    retired_autofit_models = sorted(
        model for model in model_conditions
        if model.startswith("AutoFitV") and model != "AutoFitV739"
    )
    partial_detail = [
        {"model_name": model, "conditions": len(model_conditions[model]), "records": model_counter[model]}
        for model in sorted(partial_models, key=lambda m: (-len(model_conditions[m]), m))
    ]

    v739_rows = [r for r in rows if str(r.get("model_name")) == "AutoFitV739"]
    v739_conditions = sorted({_condition_key(r) for r in v739_rows})

    return {
        "metrics_files": len(metrics_files),
        "raw_records": len(rows),
        "raw_models": len(model_conditions),
        "raw_retired_autofit_models": len(retired_autofit_models),
        "raw_nonretired_models": len(model_conditions) - len(retired_autofit_models),
        "raw_complete_models": len(complete_models),
        "raw_partial_models": len(partial_models),
        "raw_complete_model_names": complete_models,
        "raw_partial_detail": partial_detail,
        "raw_category_records": dict(sorted(category_counter.items())),
        "v739_records": len(v739_rows),
        "v739_conditions_landed": len(v739_conditions),
        "v739_condition_keys": [
            {
                "task": key[0],
                "ablation": key[1],
                "target": key[2],
                "horizon": key[3],
            }
            for key in v739_conditions
        ],
    }


def _scan_filtered(filtered_csv: Path) -> Dict[str, Any]:
    rows = _read_csv(filtered_csv)
    model_conditions: Dict[str, set] = defaultdict(set)
    category_counter: Counter[str] = Counter()

    for row in rows:
        model = str(row.get("model_name", "UNKNOWN"))
        category = str(row.get("category", "unknown"))
        model_conditions[model].add(_condition_key(row))
        category_counter[category] += 1

    retired_autofit_models = sorted(
        model for model in model_conditions
        if model.startswith("AutoFitV") and model != "AutoFitV739"
    )
    complete = sum(1 for v in model_conditions.values() if len(v) >= FULL_CONDITION_COUNT)
    partial = sum(1 for v in model_conditions.values() if 0 < len(v) < FULL_CONDITION_COUNT)
    return {
        "filtered_records": len(rows),
        "filtered_models": len(model_conditions),
        "filtered_retired_autofit_models": len(retired_autofit_models),
        "filtered_nonretired_models": len(model_conditions) - len(retired_autofit_models),
        "filtered_complete_models": complete,
        "filtered_partial_models": partial,
        "filtered_categories": dict(sorted(category_counter.items())),
        "filtered_autofit_models": sorted(
            model for model, conds in model_conditions.items() if any(
                row.get("category") == "autofit" and row.get("model_name") == model for row in rows
            )
        ),
    }


def _run_squeue() -> Dict[str, Any]:
    fmt = "%.18j|%.8u|%.10T|%.10P|%.18q|%.25R"
    try:
        out = subprocess.check_output(
            ["bash", "-lc", f"squeue -u npin,cfisch -h -o '{fmt}'"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
    except Exception:
        return {
            "jobs_total": 0,
            "state_counts": {},
            "user_state_counts": {},
            "partition_state_counts": {},
            "qos_state_counts": {},
            "pending_reasons": {},
            "v739_jobs": 0,
            "v739_state_counts": {},
            "text_jobs": 0,
            "jobs": [],
        }

    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        name, user, state, partition, qos, reason = [x.strip() for x in line.split("|", 5)]
        rows.append(
            {
                "job_name": name,
                "user": user,
                "state": state,
                "partition": partition,
                "qos": qos,
                "reason": reason,
            }
        )

    state_counts = Counter(r["state"] for r in rows)
    user_state_counts = Counter((r["user"], r["state"]) for r in rows)
    partition_state_counts = Counter((r["partition"], r["state"]) for r in rows)
    qos_state_counts = Counter((r["qos"], r["state"]) for r in rows)
    pending_reasons = Counter(r["reason"] for r in rows if r["state"] == "PENDING")
    v739_rows = []
    for r in rows:
        job = r["job_name"].lower()
        if job.startswith("af739_") or job.startswith("v739_") or job == "autofitv739":
            v739_rows.append(r)
    text_rows = [r for r in rows if "text" in r["job_name"].lower() or "embed" in r["job_name"].lower()]

    return {
        "jobs_total": len(rows),
        "state_counts": dict(sorted(state_counts.items())),
        "user_state_counts": {
            f"{user}|{state}": count for (user, state), count in sorted(user_state_counts.items())
        },
        "partition_state_counts": {
            f"{partition}|{state}": count
            for (partition, state), count in sorted(partition_state_counts.items())
        },
        "qos_state_counts": {
            f"{qos}|{state}": count for (qos, state), count in sorted(qos_state_counts.items())
        },
        "pending_reasons": dict(sorted(pending_reasons.items(), key=lambda kv: (-kv[1], kv[0]))),
        "v739_jobs": len(v739_rows),
        "v739_state_counts": dict(sorted(Counter(r["state"] for r in v739_rows).items())),
        "text_jobs": len(text_rows),
        "jobs": rows,
    }


def _scan_text_embeddings(text_dir: Path) -> Dict[str, Any]:
    meta = _read_json(text_dir / "embedding_metadata.json", default={}) or {}
    parquet_path = text_dir / "text_embeddings.parquet"
    pca_path = text_dir / "pca_model.pkl"
    files = sorted(p.name for p in text_dir.iterdir()) if text_dir.exists() else []
    return {
        "directory_exists": text_dir.exists(),
        "files_present": files,
        "artifacts_complete": parquet_path.exists() and pca_path.exists() and bool(meta),
        "metadata": meta,
        "parquet_path": str(parquet_path.relative_to(ROOT)) if parquet_path.exists() else None,
        "pca_model_path": str(pca_path.relative_to(ROOT)) if pca_path.exists() else None,
    }


def _extract_output_dir(script_path: Path) -> str:
    try:
        text = script_path.read_text(encoding="utf-8")
    except Exception:
        return ""
    match = re.search(r"--output-dir\s+(\S+)", text)
    return match.group(1) if match else ""


def _scan_v739_submission_surface() -> Dict[str, Any]:
    families = {
        "phase10_fast_npin": sorted((ROOT / ".slurm_scripts" / "phase10_fast" / "v739").glob("*.sh")),
        "phase10_fast_cfisch": sorted((ROOT / ".slurm_scripts" / "phase10_fast" / "v739_cf").glob("*.sh")),
        "phase12_l40s_cfisch": sorted((ROOT / ".slurm_scripts" / "phase12" / "v739_l40s").glob("*.sh")),
        "phase12_rerun_autofit": sorted((ROOT / ".slurm_scripts" / "phase12" / "rerun").glob("*af39*.sh")),
    }

    detail: Dict[str, List[Dict[str, Any]]] = {}
    canonical = 0
    legacy_phase10 = 0
    for family, paths in families.items():
        rows = []
        for path in paths:
            output_dir = _extract_output_dir(path)
            target_class = "other"
            if output_dir.startswith("runs/benchmarks/block3_phase9_fair"):
                target_class = "canonical_phase9_fair"
                canonical += 1
            elif output_dir.startswith("runs/benchmarks/block3_phase10"):
                target_class = "legacy_phase10"
                legacy_phase10 += 1
            rows.append(
                {
                    "script": str(path.relative_to(ROOT)),
                    "target_class": target_class,
                    "output_dir": output_dir or None,
                }
            )
        detail[family] = rows

    return {
        "families": detail,
        "canonical_phase9_fair_scripts": canonical,
        "legacy_phase10_scripts": legacy_phase10,
    }


def _render_table(rows: Iterable[Dict[str, Any]], cols: List[str]) -> str:
    rows = list(rows)
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join([header, sep] + body)


def _write_outputs(snapshot: Dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    raw = snapshot["raw"]
    filtered = snapshot["filtered"]
    slurm = snapshot["slurm"]
    text = snapshot["text_embeddings"]
    v739_surface = snapshot["v739_submission_surface"]

    md_lines = [
        "# Phase 9 / V739 Current Snapshot",
        "",
        f"> Generated: {snapshot['generated_at_utc']}",
        f"> Canonical benchmark: `{snapshot['benchmark_dir']}`",
        "",
        "## Verified Current Facts",
        "",
        _render_table(
            [
                {"metric": "metrics_files", "value": raw["metrics_files"], "evidence": "raw metrics scan"},
                {"metric": "raw_records", "value": raw["raw_records"], "evidence": "raw metrics scan"},
                {"metric": "raw_models", "value": raw["raw_models"], "evidence": "raw metrics scan"},
                {"metric": "raw_nonretired_models", "value": raw["raw_nonretired_models"], "evidence": "raw metrics scan minus retired AutoFit legacy lines"},
                {"metric": "raw_retired_autofit_models", "value": raw["raw_retired_autofit_models"], "evidence": "raw metrics scan"},
                {"metric": "raw_complete_models", "value": raw["raw_complete_models"], "evidence": "raw metrics scan"},
                {"metric": "raw_partial_models", "value": raw["raw_partial_models"], "evidence": "raw metrics scan"},
                {"metric": "filtered_records", "value": filtered["filtered_records"], "evidence": "`all_results.csv`"},
                {"metric": "filtered_models", "value": filtered["filtered_models"], "evidence": "`all_results.csv` (includes retired AutoFit legacy lines if they pass filters)"},
                {"metric": "filtered_nonretired_models", "value": filtered["filtered_nonretired_models"], "evidence": "`all_results.csv` minus retired AutoFit legacy lines"},
                {"metric": "filtered_retired_autofit_models", "value": filtered["filtered_retired_autofit_models"], "evidence": "`all_results.csv`"},
                {"metric": "filtered_complete_models", "value": filtered["filtered_complete_models"], "evidence": "`all_results.csv`"},
                {"metric": "filtered_partial_models", "value": filtered["filtered_partial_models"], "evidence": "`all_results.csv`"},
                {"metric": "v739_conditions_landed", "value": raw["v739_conditions_landed"], "evidence": "raw metrics scan"},
                {"metric": "v739_jobs_live", "value": slurm["v739_jobs"], "evidence": "`squeue -u npin,cfisch`"},
                {"metric": "v739_canonical_phase9_scripts", "value": v739_surface["canonical_phase9_fair_scripts"], "evidence": "V739 script scan"},
                {"metric": "v739_legacy_phase10_scripts", "value": v739_surface["legacy_phase10_scripts"], "evidence": "V739 script scan"},
                {"metric": "text_embeddings_artifacts_complete", "value": text["artifacts_complete"], "evidence": "`runs/text_embeddings/`"},
            ],
            ["metric", "value", "evidence"],
        ),
        "",
        "## Live Queue Snapshot",
        "",
        _render_table(
            [
                {"metric": "jobs_total", "value": slurm["jobs_total"]},
                {"metric": "running", "value": slurm["state_counts"].get("RUNNING", 0)},
                {"metric": "pending", "value": slurm["state_counts"].get("PENDING", 0)},
                {"metric": "npin_pending", "value": slurm["user_state_counts"].get("npin|PENDING", 0)},
                {"metric": "cfisch_pending", "value": slurm["user_state_counts"].get("cfisch|PENDING", 0)},
                {"metric": "v739_pending", "value": slurm["v739_state_counts"].get("PENDING", 0)},
                {"metric": "v739_running", "value": slurm["v739_state_counts"].get("RUNNING", 0)},
            ],
            ["metric", "value"],
        ),
        "",
        "### Pending Reasons",
        "",
        _render_table(
            [{"reason": k, "count": v} for k, v in slurm["pending_reasons"].items()],
            ["reason", "count"],
        ),
        "",
        "## Text Embedding Artifacts",
        "",
        _render_table(
            [
                {"field": "directory_exists", "value": text["directory_exists"]},
                {"field": "artifacts_complete", "value": text["artifacts_complete"]},
                {"field": "parquet_path", "value": text["parquet_path"]},
                {"field": "pca_model_path", "value": text["pca_model_path"]},
                {"field": "n_total_rows", "value": text["metadata"].get("n_total_rows")},
                {"field": "n_unique_texts", "value": text["metadata"].get("n_unique_texts")},
                {"field": "n_entities", "value": text["metadata"].get("n_entities")},
                {"field": "pca_dim", "value": text["metadata"].get("pca_dim")},
            ],
            ["field", "value"],
        ),
        "",
        "## V739 Submission Surface",
        "",
        _render_table(
            [
                {"metric": "canonical_phase9_fair_scripts", "value": v739_surface["canonical_phase9_fair_scripts"]},
                {"metric": "legacy_phase10_scripts", "value": v739_surface["legacy_phase10_scripts"]},
            ],
            ["metric", "value"],
        ),
        "",
        "### V739 Script Families",
        "",
    ]

    for family, rows in v739_surface["families"].items():
        md_lines.extend(
            [
                f"#### {family}",
                "",
                _render_table(rows, ["script", "target_class", "output_dir"]),
                "",
            ]
        )

    md_lines.extend(
        [
            "## Partial Raw Models",
            "",
            _render_table(raw["raw_partial_detail"], ["model_name", "conditions", "records"]),
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a current Phase 9 / V739 fact snapshot.")
    parser.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--filtered-csv", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--text-dir", type=Path, default=DEFAULT_TEXT_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    raw = _scan_raw_metrics(args.bench_dir)
    filtered = _scan_filtered(args.filtered_csv)
    slurm = _run_squeue()
    text_embeddings = _scan_text_embeddings(args.text_dir)
    v739_submission_surface = _scan_v739_submission_surface()

    snapshot = {
        "generated_at_utc": _utc_now(),
        "benchmark_dir": str(args.bench_dir.relative_to(ROOT)),
        "filtered_csv": str(args.filtered_csv.relative_to(ROOT)),
        "text_dir": str(args.text_dir.relative_to(ROOT)),
        "raw": raw,
        "filtered": filtered,
        "slurm": slurm,
        "text_embeddings": text_embeddings,
        "v739_submission_surface": v739_submission_surface,
    }

    _write_outputs(snapshot, args.json_out, args.md_out)
    print(json.dumps({
        "json_out": str(args.json_out.relative_to(ROOT)),
        "md_out": str(args.md_out.relative_to(ROOT)),
        "raw_records": raw["raw_records"],
        "filtered_records": filtered["filtered_records"],
        "v739_conditions_landed": raw["v739_conditions_landed"],
        "v739_jobs_live": slurm["v739_jobs"],
        "v739_canonical_phase9_scripts": v739_submission_surface["canonical_phase9_fair_scripts"],
        "v739_legacy_phase10_scripts": v739_submission_surface["legacy_phase10_scripts"],
        "text_embeddings_artifacts_complete": text_embeddings["artifacts_complete"],
    }, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
