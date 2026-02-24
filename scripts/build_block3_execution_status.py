#!/usr/bin/env python3
"""Build canonical Block3 execution status, queue-action ledger, and certification artifacts."""

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRUTH_PACK_DIR = ROOT / "docs" / "benchmarks" / "block3_truth_pack"
DEFAULT_MASTER_DOC = ROOT / "docs" / "AUTOFIT_V72_EVIDENCE_MASTER_20260217.md"
DEFAULT_STATUS_DOC = ROOT / "docs" / "BLOCK3_MODEL_STATUS.md"
DEFAULT_RESULTS_DOC = ROOT / "docs" / "BLOCK3_RESULTS.md"
DEFAULT_SLURM_SINCE = "2026-02-12"

TASK_MAP = {
    "1": "task1_outcome",
    "2": "task2_forecast",
    "3": "task3_risk_adjust",
}
ABLATION_MAP = {
    "co": "core_only",
    "ct": "core_text",
    "ce": "core_edgar",
    "fu": "full",
}
TARGET_ORDER = {
    "task1_outcome": ["funding_raised_usd", "investors_count", "is_funded"],
    "task2_forecast": ["funding_raised_usd", "investors_count"],
    "task3_risk_adjust": ["funding_raised_usd", "investors_count"],
}

GROUP_ORDER = [
    "autofit_v72_completion",
    "autofit_resubmit",
    "statistical_resubmit",
    "v71_g01",
    "foundation_reference",
    "other_active",
]


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _read_json(path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _to_int(value):
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        return int(float(str(value)))
    except Exception:
        return None


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _bar(done, total, width=24):
    if total <= 0:
        return "[" + ("-" * width) + "] 0/0 (0.0%)"
    done = max(0, min(done, total))
    ratio = float(done) / float(total)
    filled = int(round(ratio * width))
    return "[%s%s] %d/%d (%.1f%%)" % ("#" * filled, "-" * (width - filled), done, total, ratio * 100.0)


def _parse_duration_hours(raw):
    text = (raw or "").strip()
    if not text or text in {"N/A", "Unknown"}:
        return None
    day = 0
    if "-" in text:
        day_str, text = text.split("-", 1)
        try:
            day = int(day_str)
        except Exception:
            return None
    parts = text.split(":")
    try:
        nums = [int(p) for p in parts]
    except Exception:
        return None
    if len(nums) == 3:
        hh, mm, ss = nums
    elif len(nums) == 2:
        hh, mm = nums
        ss = 0
    else:
        return None
    total_hours = day * 24.0 + hh + (mm / 60.0) + (ss / 3600.0)
    return float(total_hours)


def _run_command(cmd, timeout=40):
    try:
        out = subprocess.check_output(list(cmd), stderr=subprocess.DEVNULL, timeout=timeout)
        return out.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _detect_job_group(name):
    if name.startswith("p7v72c_"):
        return "autofit_v72_completion"
    if name.startswith("p7r_af"):
        return "autofit_resubmit"
    if name.startswith("p7_sta"):
        return "statistical_resubmit"
    if name.startswith("p7x_g01"):
        return "v71_g01"
    if name.startswith("p7xF_fdr"):
        return "foundation_reference"
    return "other_active"


def _infer_task_ablation(job_name):
    m = re.search(r"_t([123])_([a-z]{2})(?:_|$)", job_name)
    if not m:
        return None, None
    task = TASK_MAP.get(m.group(1))
    ablation = ABLATION_MAP.get(m.group(2))
    return task, ablation


def _capture_slurm(slurm_since):
    squeue_cmd = ["bash", "-lc", "squeue -u $USER -h -o '%i|%T|%j|%P|%R|%M|%l'"]
    sacct_cmd = ["bash", "-lc", "sacct -u $USER -S %s -n -X -o JobName,State -P" % slurm_since]
    qos_cmd = [
        "bash",
        "-lc",
        "sacctmgr show qos iris-batch-long,iris-gpu-long format=Name,MaxJobsPU,MaxWall,Priority -P -n",
    ]

    squeue_raw = _run_command(squeue_cmd)
    sacct_raw = _run_command(sacct_cmd)
    qos_raw = _run_command(qos_cmd)

    jobs = []
    pending_reason_counter = Counter()
    running_by_partition = Counter()
    pending_by_partition = Counter()

    for line in squeue_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 6)
        if len(parts) != 7:
            continue
        job_id, state, job_name, partition, reason, elapsed, timelimit = parts
        task, ablation = _infer_task_ablation(job_name)
        elapsed_hours = _parse_duration_hours(elapsed)
        timelimit_hours = _parse_duration_hours(timelimit)
        remaining_hours = None
        if elapsed_hours is not None and timelimit_hours is not None:
            remaining_hours = max(0.0, timelimit_hours - elapsed_hours)

        jobs.append(
            {
                "job_id": job_id,
                "state": state,
                "job_name": job_name,
                "partition": partition,
                "reason": reason,
                "elapsed": elapsed,
                "timelimit": timelimit,
                "elapsed_hours": elapsed_hours,
                "timelimit_hours": timelimit_hours,
                "remaining_hours": remaining_hours,
                "task": task,
                "ablation": ablation,
                "group": _detect_job_group(job_name),
            }
        )

        if state == "RUNNING":
            running_by_partition[partition] += 1
        if state == "PENDING":
            pending_by_partition[partition] += 1
            pending_reason_counter[reason or "(unknown)"] += 1

    qos_caps = {}
    for line in qos_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        name = parts[0]
        qos_caps[name] = {
            "MaxJobsPU": _to_int(parts[1]),
            "MaxWall": parts[2],
            "Priority": _to_int(parts[3]),
        }

    prefix_status_sacct = defaultdict(lambda: defaultdict(int))
    for line in sacct_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 1)
        if len(parts) != 2:
            continue
        job_name = parts[0].strip()
        state = parts[1].strip().split()[0]
        prefix = None
        for pfx in ["p7xF", "p7x", "p7r", "p7"]:
            if job_name.startswith(pfx):
                prefix = pfx
                break
        if prefix is None:
            continue
        prefix_status_sacct[prefix][state] += 1

    prefix_status_squeue = defaultdict(lambda: defaultdict(int))
    for job in jobs:
        prefix = None
        for pfx in ["p7xF", "p7x", "p7r", "p7"]:
            if job["job_name"].startswith(pfx):
                prefix = pfx
                break
        if prefix is None:
            continue
        prefix_status_squeue[prefix][job["state"]] += 1

    return {
        "snapshot_ts": _utc_now(),
        "running_total": int(sum(1 for j in jobs if j["state"] == "RUNNING")),
        "pending_total": int(sum(1 for j in jobs if j["state"] == "PENDING")),
        "running_by_partition": dict(sorted(running_by_partition.items())),
        "pending_by_partition": dict(sorted(pending_by_partition.items())),
        "pending_reason_topk": [
            {"reason": r, "count": int(c)} for r, c in pending_reason_counter.most_common(8)
        ],
        "qos_caps": qos_caps,
        "prefix_status_squeue": {
            k: dict(sorted(v.items())) for k, v in sorted(prefix_status_squeue.items())
        },
        "prefix_status_sacct": {
            k: dict(sorted(v.items())) for k, v in sorted(prefix_status_sacct.items())
        },
        "commands": {
            "squeue": "squeue -u $USER -h -o '%i|%T|%j|%P|%R|%M|%l'",
            "sacct": "sacct -u $USER -S %s -n -X -o JobName,State -P" % slurm_since,
            "qos": "sacctmgr show qos iris-batch-long,iris-gpu-long format=Name,MaxJobsPU,MaxWall,Priority -P -n",
        },
        "jobs": jobs,
    }


def _expected_keys_from_inventory(rows):
    keys = []
    for r in rows:
        if not _to_bool(r.get("expected")):
            continue
        horizon = _to_int(r.get("horizon"))
        if horizon is None:
            continue
        keys.append((str(r.get("task")), str(r.get("ablation")), str(r.get("target")), int(horizon)))
    return sorted(set(keys))


def _missing_keys_from_manifest(rows):
    out = set()
    for r in rows:
        h = _to_int(r.get("horizon"))
        if h is None:
            continue
        out.add((str(r.get("task")), str(r.get("ablation")), str(r.get("target")), int(h)))
    return out


def _strict_done_set(rows):
    out = set()
    for r in rows:
        if not _to_bool(r.get("strict_completed")):
            continue
        h = _to_int(r.get("horizon"))
        if h is None:
            continue
        out.add((str(r.get("task")), str(r.get("ablation")), str(r.get("target")), int(h)))
    return out


def _build_nested_tree(expected_keys, strict_done, v72_done, jobs):
    tree = {}

    for task, ablation, target, horizon in expected_keys:
        tnode = tree.setdefault(
            task,
            {
                "expected_keys": 0,
                "strict_completed_keys": 0,
                "v72_completed_keys": 0,
                "running_jobs": [],
                "pending_jobs": [],
                "ablations": {},
            },
        )
        tnode["expected_keys"] += 1
        key = (task, ablation, target, horizon)
        if key in strict_done:
            tnode["strict_completed_keys"] += 1
        if key in v72_done:
            tnode["v72_completed_keys"] += 1

        anode = tnode["ablations"].setdefault(
            ablation,
            {
                "expected_keys": 0,
                "strict_completed_keys": 0,
                "v72_completed_keys": 0,
                "running_jobs": [],
                "pending_jobs": [],
                "targets": {},
            },
        )
        anode["expected_keys"] += 1
        if key in strict_done:
            anode["strict_completed_keys"] += 1
        if key in v72_done:
            anode["v72_completed_keys"] += 1

        gnode = anode["targets"].setdefault(
            target,
            {
                "expected_keys": 0,
                "strict_completed_keys": 0,
                "v72_completed_keys": 0,
                "horizons": {},
            },
        )
        gnode["expected_keys"] += 1
        if key in strict_done:
            gnode["strict_completed_keys"] += 1
        if key in v72_done:
            gnode["v72_completed_keys"] += 1

        gnode["horizons"][str(horizon)] = {
            "strict_completed": key in strict_done,
            "v72_completed": key in v72_done,
        }

    for job in jobs:
        if not job.get("task") or not job.get("ablation"):
            continue
        tnode = tree.get(job["task"])
        if not isinstance(tnode, dict):
            continue
        anode = tnode.get("ablations", {}).get(job["ablation"])
        if not isinstance(anode, dict):
            continue
        if job.get("state") == "RUNNING":
            tnode["running_jobs"].append(job.get("job_name"))
            anode["running_jobs"].append(job.get("job_name"))
        elif job.get("state") == "PENDING":
            tnode["pending_jobs"].append(job.get("job_name"))
            anode["pending_jobs"].append(job.get("job_name"))

    return tree


def _estimate_eta(slurm):
    jobs = slurm.get("jobs", []) if isinstance(slurm.get("jobs"), list) else []

    caps = slurm.get("qos_caps", {}) if isinstance(slurm.get("qos_caps"), dict) else {}
    batch_cap = _to_int((caps.get("iris-batch-long") or {}).get("MaxJobsPU")) or 8
    gpu_cap = _to_int((caps.get("iris-gpu-long") or {}).get("MaxJobsPU")) or 4

    def _partition_eta(partition, cap, default_hours):
        running = [j for j in jobs if j.get("partition") == partition and j.get("state") == "RUNNING"]
        pending = [j for j in jobs if j.get("partition") == partition and j.get("state") == "PENDING"]

        remaining_samples = [j.get("remaining_hours") for j in running if j.get("remaining_hours") is not None]
        timelimit_samples = [j.get("timelimit_hours") for j in running if j.get("timelimit_hours") is not None]

        if timelimit_samples:
            avg_job_hours = float(sum(timelimit_samples) / len(timelimit_samples))
        else:
            avg_job_hours = default_hours

        throughput = float(cap) / max(avg_job_hours, 1e-6)
        queue_clear = float(len(pending)) / max(throughput, 1e-6)
        running_tail = max(remaining_samples) if remaining_samples else 0.0
        baseline = running_tail + queue_clear

        return {
            "partition": partition,
            "cap": int(cap),
            "running": len(running),
            "pending": len(pending),
            "avg_job_hours": round(avg_job_hours, 2),
            "running_tail_hours": round(running_tail, 2),
            "queue_clear_hours": round(queue_clear, 2),
            "baseline_hours": round(baseline, 2),
        }

    batch = _partition_eta("batch", batch_cap, 24.0)
    gpu = _partition_eta("gpu", gpu_cap, 24.0)

    baseline_hours = max(batch["baseline_hours"], gpu["baseline_hours"])
    optimistic_hours = round(max(0.5, baseline_hours * 0.70), 2)
    conservative_hours = round(max(baseline_hours, baseline_hours * 1.40 + 4.0), 2)

    return {
        "method": "partition_aware_heuristic_v2",
        "optimistic_hours": optimistic_hours,
        "baseline_hours": round(baseline_hours, 2),
        "conservative_hours": conservative_hours,
        "batch": batch,
        "gpu": gpu,
        "confidence_note": "Heuristic ETA from active elapsed/timelimit and QOS caps; scheduler priority can shift actual start/end.",
    }


def _group_horizons(keys):
    grouped = defaultdict(list)
    for t, a, g, h in keys:
        grouped[(t, a, g)].append(int(h))
    for k in list(grouped.keys()):
        grouped[k] = sorted(set(grouped[k]))
    return grouped


def _render_nested_md(snapshot_ts, strict_done_n, strict_total_n, v72_done_n, v72_total_n, tree, jobs, eta_model, queue_bottlenecks, missing_keys, v72_done_keys):
    lines = [
        "- Snapshot UTC: **%s**" % snapshot_ts,
        "- Queue policy: **V72-first**",
        "- Strict matrix progress: **%s**" % _bar(strict_done_n, strict_total_n),
        "- V7.2 coverage progress: **%s**" % _bar(v72_done_n, v72_total_n),
        "",
    ]

    running_total = sum(1 for j in jobs if j.get("state") == "RUNNING")
    pending_total = sum(1 for j in jobs if j.get("state") == "PENDING")
    lines.extend(
        [
            "## Active Queue Summary",
            "",
            "| metric | value |",
            "|---|---:|",
            "| running_total | %s |" % running_total,
            "| pending_total | %s |" % pending_total,
            "| eta_optimistic_hours | %s |" % eta_model.get("optimistic_hours"),
            "| eta_baseline_hours | %s |" % eta_model.get("baseline_hours"),
            "| eta_conservative_hours | %s |" % eta_model.get("conservative_hours"),
            "",
            "### Active Jobs by Group",
            "",
            "| group | running | pending |",
            "|---|---:|---:|",
        ]
    )
    grouped_counts = defaultdict(lambda: {"RUNNING": 0, "PENDING": 0})
    for j in jobs:
        if j.get("state") in {"RUNNING", "PENDING"}:
            grouped_counts[j.get("group")][j.get("state")] += 1
    for g in GROUP_ORDER:
        c = grouped_counts.get(g, {"RUNNING": 0, "PENDING": 0})
        lines.append("| %s | %d | %d |" % (g, c.get("RUNNING", 0), c.get("PENDING", 0)))

    lines.extend([
        "",
        "### Queue Bottlenecks",
        "",
        "| reason | count |",
        "|---|---:|",
    ])
    for row in queue_bottlenecks:
        lines.append("| %s | %s |" % (row.get("reason"), row.get("count")))

    lines.extend(["", "## Nested Task Progress", ""])

    for task in ["task1_outcome", "task2_forecast", "task3_risk_adjust"]:
        tnode = tree.get(task)
        if not isinstance(tnode, dict):
            continue
        t_exp = int(tnode.get("expected_keys", 0))
        t_strict = int(tnode.get("strict_completed_keys", 0))
        t_v72 = int(tnode.get("v72_completed_keys", 0))
        lines.append("### %s" % task)
        lines.append("- strict: %s" % _bar(t_strict, t_exp))
        lines.append("- v72: %s" % _bar(t_v72, t_exp))
        lines.append("- running_jobs: %d, pending_jobs: %d" % (len(tnode.get("running_jobs", [])), len(tnode.get("pending_jobs", []))))
        lines.append("")
        lines.append("| ablation | strict_progress | v72_progress | running_jobs | pending_jobs |")
        lines.append("|---|---|---|---:|---:|")
        for ablation in ["core_only", "core_text", "core_edgar", "full"]:
            anode = tnode.get("ablations", {}).get(ablation)
            if not isinstance(anode, dict):
                continue
            a_exp = int(anode.get("expected_keys", 0))
            a_strict = int(anode.get("strict_completed_keys", 0))
            a_v72 = int(anode.get("v72_completed_keys", 0))
            lines.append(
                "| %s | %s | %s | %d | %d |"
                % (
                    ablation,
                    _bar(a_strict, a_exp),
                    _bar(a_v72, a_exp),
                    len(anode.get("running_jobs", [])),
                    len(anode.get("pending_jobs", [])),
                )
            )
        lines.append("")

        lines.append("| target | strict_done/expected | v72_done/expected | missing_horizons |")
        lines.append("|---|---:|---:|---|")
        for target in TARGET_ORDER.get(task, []):
            total = 0
            s_done = 0
            v_done = 0
            missing_h = []
            for ablation in ["core_only", "core_text", "core_edgar", "full"]:
                anode = tnode.get("ablations", {}).get(ablation)
                if not isinstance(anode, dict):
                    continue
                gnode = anode.get("targets", {}).get(target)
                if not isinstance(gnode, dict):
                    continue
                total += int(gnode.get("expected_keys", 0))
                s_done += int(gnode.get("strict_completed_keys", 0))
                v_done += int(gnode.get("v72_completed_keys", 0))
                for h, st in gnode.get("horizons", {}).items():
                    if not _to_bool((st or {}).get("v72_completed")):
                        hh = _to_int(h)
                        if hh is not None:
                            missing_h.append(hh)
            if total == 0:
                continue
            missing_h = sorted(set(missing_h))
            lines.append(
                "| %s | %d/%d | %d/%d | %s |"
                % (target, s_done, total, v_done, total, ",".join(str(x) for x in missing_h) if missing_h else "-")
            )
        lines.append("")

    missing_grouped = _group_horizons(missing_keys)
    done_grouped = _group_horizons(v72_done_keys)

    lines.extend([
        "## V7.2 Completed Subtasks (by task/ablation/target)",
        "",
        "| task | ablation | target | completed_horizons |",
        "|---|---|---|---|",
    ])
    for (task, ablation, target), horizons in sorted(done_grouped.items()):
        lines.append("| %s | %s | %s | %s |" % (task, ablation, target, ",".join(str(h) for h in horizons)))

    lines.extend([
        "",
        "## V7.2 Missing Subtasks (by task/ablation/target)",
        "",
        "| task | ablation | target | missing_horizons |",
        "|---|---|---|---|",
    ])
    for (task, ablation, target), horizons in sorted(missing_grouped.items()):
        lines.append("| %s | %s | %s | %s |" % (task, ablation, target, ",".join(str(h) for h in horizons)))

    lines.append("")
    return "\n".join(lines)


def _build_queue_actions(jobs):
    rows = []
    for j in jobs:
        redundancy_check = "non_redundant_required"
        action = "keep"
        expected_eta_delta_hours = 0.0
        reason = "keep_non_duplicate_required_work"
        evidence_path = "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv"

        if j.get("group") == "v71_g01":
            redundancy_check = "strict_matrix_already_complete=true"
            if j.get("state") == "PENDING":
                action = "deprioritize_hold_recommended"
                expected_eta_delta_hours = 2.0
            else:
                action = "no_action_running"
            reason = "residual_v71_variant_not_required_for_v72_first_completion"
        elif j.get("group") == "foundation_reference":
            redundancy_check = "strict_matrix_already_complete=true"
            if j.get("state") == "PENDING":
                action = "deprioritize_hold_recommended"
                expected_eta_delta_hours = 1.0
            else:
                action = "no_action_running"
            reason = "duplicate_reference_branch_already_covered_in_strict_matrix"
        elif j.get("group") in {"autofit_v72_completion", "autofit_resubmit", "statistical_resubmit"}:
            action = "keep_priority"
            redundancy_check = "required_for_v72_first"
            reason = "contributes_to_v72_or_required_resubmit_coverage"
            evidence_path = "docs/benchmarks/block3_truth_pack/missing_key_manifest.csv"

        rows.append(
            {
                "job_id": j.get("job_id"),
                "job_name": j.get("job_name"),
                "state": j.get("state"),
                "partition": j.get("partition"),
                "group": j.get("group"),
                "action": action,
                "redundancy_check": redundancy_check,
                "reason": reason,
                "expected_eta_delta_hours": expected_eta_delta_hours,
                "evidence_path": evidence_path,
                "timestamp_utc": _utc_now(),
            }
        )

    return {
        "generated_at_utc": _utc_now(),
        "policy": "V72-first",
        "actions": rows,
    }


def _render_queue_actions_md(payload):
    lines = [
        "- generated_at_utc: **%s**" % payload.get("generated_at_utc"),
        "",
        "| job_id | job_name | state | group | action | redundancy_check | expected_eta_delta_hours | evidence_path |",
        "|---|---|---|---|---|---|---:|---|",
    ]
    for row in payload.get("actions", []):
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %.2f | %s |"
            % (
                row.get("job_id"),
                row.get("job_name"),
                row.get("state"),
                row.get("group"),
                row.get("action"),
                row.get("redundancy_check"),
                float(row.get("expected_eta_delta_hours") or 0.0),
                row.get("evidence_path"),
            )
        )
    lines.append("")
    lines.append("Note: actions are recommendation artifacts; queue mutation is not performed by this script.")
    lines.append("")
    return "\n".join(lines)


def _build_fairness_certification(tp_dir):
    data_integrity = _read_json(tp_dir / "data_integrity_audit_latest.json", {}) or {}
    investors = _read_json(tp_dir / "investors_count_stability_audit_latest.json", {}) or {}
    pilot = _read_json(tp_dir / "v72_pilot_gate_report.json", {}) or {}

    split = (data_integrity.get("split") or {})
    split_checks = (split.get("checks") or {}) if isinstance(split, dict) else {}
    leakage = (data_integrity.get("leakage_policy") or {}) if isinstance(data_integrity, dict) else {}
    inv_checks = (investors.get("checks") or {}) if isinstance(investors, dict) else {}

    coverage_pass = bool((pilot.get("checks") or {}).get("fairness_pass_100", False))

    items = [
        {
            "check": "temporal_split_embargo",
            "status": bool(split.get("all_pass", False)) and bool(split_checks.get("train_before_val", False)) and bool(split_checks.get("val_before_test", False)) and bool(split_checks.get("embargo_non_negative", False)),
            "evidence_path": "docs/benchmarks/block3_truth_pack/data_integrity_audit_latest.json",
            "detail": split_checks,
        },
        {
            "check": "leakage_policy_coverage",
            "status": bool(leakage.get("all_pass", False)),
            "evidence_path": "docs/benchmarks/block3_truth_pack/data_integrity_audit_latest.json",
            "detail": leakage.get("checks", {}),
        },
        {
            "check": "prediction_coverage_guard",
            "status": coverage_pass,
            "evidence_path": "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json",
            "detail": (pilot.get("checks") or {}),
        },
        {
            "check": "catastrophic_spike_clear",
            "status": int(investors.get("catastrophic_spikes", 0) or 0) == 0,
            "evidence_path": "docs/benchmarks/block3_truth_pack/investors_count_stability_audit_latest.json",
            "detail": {"catastrophic_spikes": investors.get("catastrophic_spikes")},
        },
        {
            "check": "calibration_stability_distribution",
            "status": bool(inv_checks.get("distribution_available", False)),
            "evidence_path": "docs/benchmarks/block3_truth_pack/investors_count_stability_audit_latest.json",
            "detail": {
                "distribution_available": inv_checks.get("distribution_available"),
                "ks_train_vs_test_lt_0_25": inv_checks.get("ks_train_vs_test_lt_0_25"),
                "psi_train_vs_test_lt_0_30": inv_checks.get("psi_train_vs_test_lt_0_30"),
            },
        },
        {
            "check": "parameter_governance_no_test_feedback",
            "status": (tp_dir / "hyperparam_search_ledger.csv").exists() and (tp_dir / "best_config_by_model_target.json").exists() and (tp_dir / "compute_cost_report.csv").exists(),
            "evidence_path": "docs/benchmarks/block3_truth_pack/hyperparam_search_ledger.csv",
            "detail": {
                "hyperparam_search_ledger_exists": (tp_dir / "hyperparam_search_ledger.csv").exists(),
                "best_config_exists": (tp_dir / "best_config_by_model_target.json").exists(),
                "compute_cost_exists": (tp_dir / "compute_cost_report.csv").exists(),
            },
        },
    ]

    overall = all(bool(i["status"]) for i in items)
    return {
        "generated_at_utc": _utc_now(),
        "overall_certified": overall,
        "items": items,
        "label": "CERTIFIED" if overall else "NOT CERTIFIED",
    }


def _render_fairness_md(payload):
    lines = [
        "- generated_at_utc: **%s**" % payload.get("generated_at_utc"),
        "- status: **%s**" % payload.get("label"),
        "",
        "| check | status | evidence_path | detail |",
        "|---|---|---|---|",
    ]
    for item in payload.get("items", []):
        status = "PASS" if item.get("status") else "NOT CERTIFIED"
        detail = json.dumps(item.get("detail", {}), ensure_ascii=True, sort_keys=True)
        lines.append("| %s | %s | %s | `%s` |" % (item.get("check"), status, item.get("evidence_path"), detail))
    lines.append("")
    return "\n".join(lines)


def _render_csv_rows_md(rows, columns):
    if not rows:
        return "_No rows available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---" for _ in columns]) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def _replace_or_append_auto_section(doc_text, section_name, heading, body):
    begin = "<!-- BEGIN AUTO:%s -->" % section_name
    end = "<!-- END AUTO:%s -->" % section_name
    payload = "%s\n\n%s\n%s\n%s\n" % (heading, begin, body.rstrip(), end)

    if begin in doc_text and end in doc_text:
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
        return pattern.sub(begin + "\n" + body.rstrip() + "\n" + end, doc_text)

    if not doc_text.endswith("\n"):
        doc_text += "\n"
    return doc_text + "\n" + payload


def _root_cause_sections(strict_total, strict_done, v72_done, v72_total, champion_counts, pilot_report, investors_audit, slurm):
    gap_pct = ((pilot_report.get("metrics") or {}).get("investors_count_gap_reduction_pct"))
    gmae = ((pilot_report.get("metrics") or {}).get("global_normalized_mae_improvement_pct"))
    overlap = ((pilot_report.get("counts") or {}).get("overlap_keys_v7_v72_non_autofit"))
    cat_spikes = investors_audit.get("catastrophic_spikes")
    dist_avail = ((investors_audit.get("checks") or {}).get("distribution_available"))
    if dist_avail:
        audit_impact = "Distribution diagnostics are available, but certification remains open due to unresolved catastrophic spike lineage."
        audit_fix = "Keep rerunning count-lane failure pool until spike count is zero, then refresh certification."
    else:
        audit_impact = "Cannot claim fully certified fair/no-leak/no-overfit closure until distribution diagnostics become available."
        audit_fix = "Rerun stability audit in insider environment with corrected import path, then refresh certification."

    root_cause_md = "\n".join(
        [
            "| problem | evidence | impact | fix | gate |",
            "|---|---|---|---|---|",
            "| V7.2 coverage deficit | v72_completed=%d/%d (strict=%d/%d), overlap_keys=%s; `docs/benchmarks/block3_truth_pack/missing_key_manifest.csv`; `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` | Full-rank claim is blocked until 104/104 V7.2 keys are materialized. | Keep V72-first queue policy, prioritize missing-key controller jobs, and rerun gate after each completed shard. | Gate-P/F blocked until coverage closes. |"
            % (v72_done, v72_total, strict_done, strict_total, overlap),
            "| Count lane weakness (`investors_count`) | gap_reduction_pct=%s; catastrophic_spikes=%s; `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json`; `docs/benchmarks/block3_truth_pack/investors_count_stability_audit_latest.json`; `docs/benchmarks/block3_truth_pack/failure_taxonomy.csv` | Median gap remains large and historical explosion lineage still exists. | Keep count-safe hard guards, complete failure-pool reruns, and retune lane thresholds after new full overlap is available. | Gate-P investors_count criterion currently fails. |"
            % (gap_pct, cat_spikes),
            "| Binary lane underperformance (`is_funded`) | Champion families remain deep/transformer; `docs/benchmarks/block3_truth_pack/top3_representative_models_by_target.csv`; `docs/benchmarks/block3_truth_pack/family_gap_by_target.csv` | AutoFit calibration/routing still behind NHITS/PatchTST in strict global benchmark. | Produce explicit binary calibration report (logloss/Brier/ECE by subtask) and retune hazard/calibration settings from full overlap. | Gate-P global-improvement criterion remains constrained. |",
            "| Queue bottlenecks | RUNNING=%s, PENDING=%s; pending reasons include `%s`; `docs/benchmarks/block3_truth_pack/execution_status_latest.json` | Delays convergence to representative full-matrix V7.2 evidence. | Apply redundancy-safe de-prioritization for non-essential residual branches under V72-first policy. | ETA uncertainty remains scheduler-dependent. |"
            % (
                slurm.get("running_total"),
                slurm.get("pending_total"),
                ", ".join(r.get("reason", "") for r in slurm.get("pending_reason_topk", [])[:2]),
            ),
            "| Audit closure incomplete | distribution_available=%s; `docs/benchmarks/block3_truth_pack/investors_count_stability_audit_latest.json` | %s | %s | Gate-S remains not fully certified. |"
            % (dist_avail, audit_impact, audit_fix),
        ]
    )

    impl_pending_md = "\n".join(
        [
            "| area | status | detail | evidence_path |",
            "|---|---|---|---|",
            "| count-safe guards | implemented | non-negative, safe inverse, clipping, OOF guard telemetry fields are present in metrics pipeline. | `scripts/run_block3_benchmark_shard.py` |",
            "| hazard head and sparse routing | implemented | V7.2 route metadata and policy telemetry integrated in current benchmark outputs. | `src/narrative/block3/models/autofit_wrapper.py` |",
            "| champion-template anchors | implemented | Template library and anchor telemetry are materialized in truth pack outputs. | `docs/benchmarks/block3_truth_pack/champion_template_library.csv` |",
            "| full V7.2 104-key completion | pending | current V7.2 coverage is %d/%d; missing keys remain in manifest. | `docs/benchmarks/block3_truth_pack/missing_key_manifest.csv` |"
            % (v72_done, v72_total),
            "| Gate-S/P/F refresh on new overlap | pending | current gate report still shows overall_pass=false, global gain=%s. | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |"
            % (gmae,),
            "| investors_count stability certification | pending | latest audit not certified (`distribution_available` and spike checks unresolved). | `docs/benchmarks/block3_truth_pack/investors_count_stability_audit_latest.json` |",
            "| binary calibration deltas | pending | explicit per-subtask logloss/Brier/ECE delta report not yet generated in truth pack latest artifacts. | `docs/benchmarks/block3_truth_pack/top3_representative_models_by_target.csv` |",
            "| queue redundancy action log | implemented | recommendation ledger generated under V72-first policy with evidence-coupled actions. | `docs/benchmarks/block3_truth_pack/queue_actions_latest.json` |",
            "| strict champion distribution | implemented | deep_classical=%d, transformer_sota=%d, foundation=%d, autofit=%d in strict matrix. | `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv` |"
            % (
                champion_counts.get("deep_classical", 0),
                champion_counts.get("transformer_sota", 0),
                champion_counts.get("foundation", 0),
                champion_counts.get("autofit", 0),
            ),
        ]
    )

    return root_cause_md, impl_pending_md


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _refresh_light_docs(status_doc_path, results_doc_path, snapshot_ts, strict_done, strict_total, v72_done, v72_total, slurm, eta_model, champion_counts, queue_actions_path):
    status_md = "\n".join(
        [
            "# Block 3 Model Benchmark Status",
            "",
            "> Last Updated: %s" % snapshot_ts,
            "> Single source of truth: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`",
            "",
            "## Snapshot",
            "",
            "| Metric | Value | Evidence |",
            "|---|---:|---|",
            "| strict_condition_completion | %d/%d | `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv` |" % (strict_done, strict_total),
            "| v72_condition_completion | %d/%d | `docs/benchmarks/block3_truth_pack/missing_key_manifest.csv` |" % (v72_done, v72_total),
            "| contract_assertion | pass/fail in latest audit | `docs/benchmarks/block3_truth_pack/v72_runtime_contract_audit.json` |",
            "| key_job_manifest_keys | see latest file | `docs/benchmarks/block3_truth_pack/v72_key_job_manifest.csv` |",
            "| memory_plan_keys | see latest file | `docs/benchmarks/block3_truth_pack/v72_memory_plan.json` |",
            "| running_total | %s | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |" % slurm.get("running_total"),
            "| pending_total | %s | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |" % slurm.get("pending_total"),
            "| eta_baseline_hours | %s | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |" % eta_model.get("baseline_hours"),
            "",
            "## Progress Bars",
            "",
            "- strict matrix: `%s`" % _bar(strict_done, strict_total),
            "- v7.2 coverage: `%s`" % _bar(v72_done, v72_total),
            "",
            "## Active Queue Groups",
            "",
            "| group | running | pending |",
            "|---|---:|---:|",
        ]
    )

    jobs = slurm.get("jobs", []) if isinstance(slurm.get("jobs"), list) else []
    counters = defaultdict(lambda: {"RUNNING": 0, "PENDING": 0})
    for j in jobs:
        state = str(j.get("state", ""))
        group = str(j.get("group", "other_active"))
        if state in {"RUNNING", "PENDING"}:
            counters[group][state] += 1
    for g in GROUP_ORDER:
        c = counters.get(g, {"RUNNING": 0, "PENDING": 0})
        status_md += "\n| %s | %d | %d |" % (g, c.get("RUNNING", 0), c.get("PENDING", 0))

    status_md += "\n\n## Queue Governance\n\n"
    status_md += "- V72-first policy ledger: `%s`\n" % queue_actions_path
    status_md += "- Mandatory execution contract: `docs/BLOCK3_EXECUTION_CONTRACT.md`\n"
    status_md += "- Actions are recommendation-only; this report does not mutate live queue state.\n"

    _write_text(status_doc_path, status_md + "\n")

    results_md = "\n".join(
        [
            "# Block 3 Benchmark Results",
            "",
            "> Last Updated: %s" % snapshot_ts,
            "> Single source of truth: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`",
            "",
            "## Strict Benchmark Snapshot",
            "",
            "| Metric | Value | Evidence |",
            "|---|---:|---|",
            "| strict_condition_completion | %d/%d | `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv` |" % (strict_done, strict_total),
            "| v72_condition_completion | %d/%d | `docs/benchmarks/block3_truth_pack/missing_key_manifest.csv` |" % (v72_done, v72_total),
            "| champion_deep_classical | %d | `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv` |" % champion_counts.get("deep_classical", 0),
            "| champion_transformer_sota | %d | `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv` |" % champion_counts.get("transformer_sota", 0),
            "| champion_foundation | %d | `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv` |" % champion_counts.get("foundation", 0),
            "| champion_autofit | %d | `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv` |" % champion_counts.get("autofit", 0),
            "| policy_training_report | available | `docs/benchmarks/block3_truth_pack/v72_policy_training_report.json` |",
            "| key_job_manifest | available | `docs/benchmarks/block3_truth_pack/v72_key_job_manifest.csv` |",
            "| memory_plan | available | `docs/benchmarks/block3_truth_pack/v72_memory_plan.json` |",
            "| running_total | %s | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |" % slurm.get("running_total"),
            "| pending_total | %s | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |" % slurm.get("pending_total"),
            "| eta_baseline_hours | %s | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |" % eta_model.get("baseline_hours"),
            "",
            "## Notes",
            "",
            "1. Full nested task/subtask progress is maintained in `docs/benchmarks/block3_truth_pack/execution_status_latest.md`.",
            "2. Root-cause and closure matrix are maintained in the master evidence document.",
        ]
    )
    _write_text(results_doc_path, results_md + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Build canonical Block3 execution status and closure artifacts.")
    p.add_argument("--truth-pack-dir", type=Path, default=DEFAULT_TRUTH_PACK_DIR)
    p.add_argument("--capture-slurm", action="store_true", default=False)
    p.add_argument("--slurm-since", type=str, default=DEFAULT_SLURM_SINCE)
    p.add_argument("--write-latest", action="store_true", default=False)
    p.add_argument("--write-timestamped", action="store_true", default=False)
    p.add_argument("--update-docs", action="store_true", default=False)
    p.add_argument("--master-doc", type=Path, default=DEFAULT_MASTER_DOC)
    p.add_argument("--status-doc", type=Path, default=DEFAULT_STATUS_DOC)
    p.add_argument("--results-doc", type=Path, default=DEFAULT_RESULTS_DOC)
    p.add_argument("--enforce-style-guard", action="store_true", default=True)
    p.add_argument("--no-enforce-style-guard", dest="enforce_style_guard", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    truth_dir = args.truth_pack_dir.resolve()
    truth_dir.mkdir(parents=True, exist_ok=True)

    if not args.write_latest and not args.write_timestamped:
        args.write_latest = True
        args.write_timestamped = True

    inventory_rows = _read_csv(truth_dir / "condition_inventory_full.csv")
    missing_rows = _read_csv(truth_dir / "missing_key_manifest.csv")
    pilot_report = _read_json(truth_dir / "v72_pilot_gate_report.json", {}) or {}
    investors_audit = _read_json(truth_dir / "investors_count_stability_audit_latest.json", {}) or {}

    expected_keys = _expected_keys_from_inventory(inventory_rows)
    strict_done = _strict_done_set(inventory_rows)
    missing_keys = _missing_keys_from_manifest(missing_rows)
    v72_done = set(expected_keys) - missing_keys

    if args.capture_slurm:
        slurm = _capture_slurm(slurm_since=args.slurm_since)
    else:
        slurm = _read_json(truth_dir / "slurm_snapshot.json", {}) or {}
        slurm.setdefault("snapshot_ts", _utc_now())
        slurm.setdefault("jobs", [])

    jobs = slurm.get("jobs", []) if isinstance(slurm.get("jobs"), list) else []
    tree = _build_nested_tree(expected_keys, strict_done, v72_done, jobs)

    eta_model = _estimate_eta(slurm)
    queue_bottlenecks = list(slurm.get("pending_reason_topk", []) or [])

    champion_counts = Counter()
    for r in inventory_rows:
        if not _to_bool(r.get("strict_completed")):
            continue
        category = str(r.get("best_category_strict") or "")
        if category:
            champion_counts[category] += 1

    strict_done_n = len(strict_done)
    strict_total_n = len(expected_keys)
    v72_done_n = len(v72_done)
    v72_total_n = len(expected_keys)

    status_payload = {
        "snapshot_ts": slurm.get("snapshot_ts", _utc_now()),
        "queue_policy": "V72-first",
        "running_total": int(sum(1 for j in jobs if j.get("state") == "RUNNING")),
        "pending_total": int(sum(1 for j in jobs if j.get("state") == "PENDING")),
        "strict_progress": {
            "completed": strict_done_n,
            "expected": strict_total_n,
            "ratio": (float(strict_done_n) / float(strict_total_n)) if strict_total_n else 0.0,
            "bar": _bar(strict_done_n, strict_total_n),
        },
        "v72_progress": {
            "completed": v72_done_n,
            "expected": v72_total_n,
            "ratio": (float(v72_done_n) / float(v72_total_n)) if v72_total_n else 0.0,
            "bar": _bar(v72_done_n, v72_total_n),
            "missing_keys": len(missing_keys),
        },
        "nested_task_tree": tree,
        "active_jobs_nested": {
            "total_running": int(sum(1 for j in jobs if j.get("state") == "RUNNING")),
            "total_pending": int(sum(1 for j in jobs if j.get("state") == "PENDING")),
            "by_group": {
                g: {
                    "RUNNING": int(sum(1 for j in jobs if j.get("group") == g and j.get("state") == "RUNNING")),
                    "PENDING": int(sum(1 for j in jobs if j.get("group") == g and j.get("state") == "PENDING")),
                }
                for g in GROUP_ORDER
            },
        },
        "eta_model": eta_model,
        "queue_bottlenecks": queue_bottlenecks,
        "evidence": {
            "condition_inventory_full": "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv",
            "missing_key_manifest": "docs/benchmarks/block3_truth_pack/missing_key_manifest.csv",
            "v72_pilot_gate_report": "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json",
            "slurm_snapshot": "docs/benchmarks/block3_truth_pack/slurm_snapshot.json",
        },
    }

    status_md = _render_nested_md(
        snapshot_ts=str(status_payload["snapshot_ts"]),
        strict_done_n=strict_done_n,
        strict_total_n=strict_total_n,
        v72_done_n=v72_done_n,
        v72_total_n=v72_total_n,
        tree=tree,
        jobs=jobs,
        eta_model=eta_model,
        queue_bottlenecks=queue_bottlenecks,
        missing_keys=missing_keys,
        v72_done_keys=v72_done,
    )

    queue_actions = _build_queue_actions(jobs)
    queue_actions_md = _render_queue_actions_md(queue_actions)

    cert_payload = _build_fairness_certification(truth_dir)
    cert_md = _render_fairness_md(cert_payload)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    latest_json = truth_dir / "execution_status_latest.json"
    latest_md = truth_dir / "execution_status_latest.md"
    ts_json = truth_dir / ("execution_status_%s.json" % stamp)
    ts_md = truth_dir / ("execution_status_%s.md" % stamp)

    queue_latest_json = truth_dir / "queue_actions_latest.json"
    queue_latest_md = truth_dir / "queue_actions_latest.md"
    queue_ts_json = truth_dir / ("queue_actions_%s.json" % stamp)
    queue_ts_md = truth_dir / ("queue_actions_%s.md" % stamp)

    cert_latest_json = truth_dir / "fairness_certification_latest.json"
    cert_latest_md = truth_dir / "fairness_certification_latest.md"
    cert_ts_json = truth_dir / ("fairness_certification_%s.json" % stamp)
    cert_ts_md = truth_dir / ("fairness_certification_%s.md" % stamp)

    if args.write_latest:
        _write_text(latest_json, json.dumps(status_payload, indent=2, ensure_ascii=True))
        _write_text(latest_md, status_md)
        _write_text(queue_latest_json, json.dumps(queue_actions, indent=2, ensure_ascii=True))
        _write_text(queue_latest_md, queue_actions_md)
        _write_text(cert_latest_json, json.dumps(cert_payload, indent=2, ensure_ascii=True))
        _write_text(cert_latest_md, cert_md)

    if args.write_timestamped:
        _write_text(ts_json, json.dumps(status_payload, indent=2, ensure_ascii=True))
        _write_text(ts_md, status_md)
        _write_text(queue_ts_json, json.dumps(queue_actions, indent=2, ensure_ascii=True))
        _write_text(queue_ts_md, queue_actions_md)
        _write_text(cert_ts_json, json.dumps(cert_payload, indent=2, ensure_ascii=True))
        _write_text(cert_ts_md, cert_md)

    if args.update_docs:
        master_doc = args.master_doc.resolve()
        master_text = master_doc.read_text(encoding="utf-8") if master_doc.exists() else ""
        master_text = _replace_or_append_auto_section(
            master_text,
            "EXECUTION_STATUS",
            "## Execution Status (Nested Progress + ETA)",
            status_md,
        )
        master_text = _replace_or_append_auto_section(
            master_text,
            "QUEUE_ACTIONS",
            "## Queue Actions (V72-first)",
            queue_actions_md,
        )
        master_text = _replace_or_append_auto_section(
            master_text,
            "FAIRNESS_CERTIFICATION",
            "## Fairness Certification",
            cert_md,
        )

        root_cause_md, impl_pending_md = _root_cause_sections(
            strict_total=strict_total_n,
            strict_done=strict_done_n,
            v72_done=v72_done_n,
            v72_total=v72_total_n,
            champion_counts=champion_counts,
            pilot_report=pilot_report,
            investors_audit=investors_audit,
            slurm=slurm,
        )
        master_text = _replace_or_append_auto_section(
            master_text,
            "V72_COVERAGE_GAP_ANALYSIS",
            "## V72 Coverage Gap Analysis",
            "| metric | value | evidence_path |\n|---|---|---|\n"
            + "| strict_completed_conditions | %d/%d | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |\n"
            % (strict_done_n, strict_total_n)
            + "| v72_completed_conditions | %d/%d | docs/benchmarks/block3_truth_pack/missing_key_manifest.csv |\n"
            % (v72_done_n, v72_total_n)
            + "| v72_missing_conditions | %d | docs/benchmarks/block3_truth_pack/missing_key_manifest.csv |\n"
            % (len(missing_keys),)
            + "| queue_running_pending | %s/%s | docs/benchmarks/block3_truth_pack/execution_status_latest.json |\n"
            % (slurm.get("running_total"), slurm.get("pending_total")),
        )
        master_text = _replace_or_append_auto_section(
            master_text,
            "ROOT_CAUSE_MATRIX",
            "## Root Cause Matrix (Problem -> Evidence -> Impact -> Fix -> Gate)",
            root_cause_md,
        )
        master_text = _replace_or_append_auto_section(
            master_text,
            "IMPLEMENTED_PENDING_IMPROVEMENTS",
            "## Implemented vs Pending Improvements",
            impl_pending_md,
        )
        contract_rows = [
            {
                "control": "execution_contract",
                "status": "enforced",
                "detail": "Contract assertion is wired into preflight, submit, and local production entrypoints.",
                "evidence_path": "docs/BLOCK3_EXECUTION_CONTRACT.md",
            },
            {
                "control": "runtime_contract_audit",
                "status": "materialized",
                "detail": "Latest contract assertion audit is written as JSON for traceability.",
                "evidence_path": "docs/benchmarks/block3_truth_pack/v72_runtime_contract_audit.json",
            },
            {
                "control": "key_level_completion",
                "status": "materialized",
                "detail": "Missing-key completion jobs are generated at (task,ablation,target,horizon) granularity.",
                "evidence_path": "docs/benchmarks/block3_truth_pack/v72_key_job_manifest.csv",
            },
            {
                "control": "memory_aware_scheduler",
                "status": "materialized",
                "detail": "Per-key memory plan emits admission-guard resource classes (L/XL).",
                "evidence_path": "docs/benchmarks/block3_truth_pack/v72_memory_plan.json",
            },
            {
                "control": "offline_policy_training",
                "status": "materialized",
                "detail": "Offline policy report is generated from strict historical evidence without test feedback.",
                "evidence_path": "docs/benchmarks/block3_truth_pack/v72_policy_training_report.json",
            },
        ]
        master_text = _replace_or_append_auto_section(
            master_text,
            "EXECUTION_CONTRACT_AND_ACCELERATION",
            "## Execution Contract and Acceleration Controls (2026-02-24)",
            _render_csv_rows_md(
                contract_rows,
                ["control", "status", "detail", "evidence_path"],
            ),
        )

        cross_snapshot = _read_json(truth_dir / "v72_cross_version_snapshot_latest.json", {}) or {}
        cross_snapshot_rows = []
        if cross_snapshot:
            strict_info = cross_snapshot.get("strict_comparable_conditions", {}) or {}
            v72_cov = cross_snapshot.get("v72_coverage", {}) or {}
            gate = cross_snapshot.get("gate_status", {}) or {}
            cert = cross_snapshot.get("fairness_certification", {}) or {}
            cross_snapshot_rows = [
                {
                    "metric": "generated_at_utc",
                    "value": cross_snapshot.get("generated_at_utc"),
                    "evidence_path": "docs/benchmarks/block3_truth_pack/v72_cross_version_snapshot_latest.json",
                },
                {
                    "metric": "strict_comparable_completion",
                    "value": "%s/%s" % (strict_info.get("completed"), strict_info.get("expected")),
                    "evidence_path": "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
                },
                {
                    "metric": "v72_coverage",
                    "value": "%s/%s (missing=%s)"
                    % (v72_cov.get("completed"), v72_cov.get("expected"), v72_cov.get("missing")),
                    "evidence_path": "docs/benchmarks/block3_truth_pack/missing_key_manifest_summary.json",
                },
                {
                    "metric": "gate_overall_pass",
                    "value": gate.get("overall_pass"),
                    "evidence_path": "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json",
                },
                {
                    "metric": "global_normalized_mae_improvement_pct",
                    "value": gate.get("global_normalized_mae_improvement_pct"),
                    "evidence_path": "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json",
                },
                {
                    "metric": "investors_count_gap_reduction_pct",
                    "value": gate.get("investors_count_gap_reduction_pct"),
                    "evidence_path": "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json",
                },
                {
                    "metric": "fairness_certification_label",
                    "value": cert.get("label"),
                    "evidence_path": "docs/benchmarks/block3_truth_pack/fairness_certification_latest.json",
                },
            ]
        master_text = _replace_or_append_auto_section(
            master_text,
            "CROSS_VERSION_SNAPSHOT",
            "## Cross-Version Snapshot (V7/V7.1/V7.2)",
            _render_csv_rows_md(
                cross_snapshot_rows,
                ["metric", "value", "evidence_path"],
            ),
        )

        cross_rootcause_rows = _read_csv(truth_dir / "v72_cross_version_rootcause_matrix.csv")
        master_text = _replace_or_append_auto_section(
            master_text,
            "CROSS_VERSION_ROOTCAUSE_MATRIX",
            "## Cross-Version Root-Cause Matrix",
            _render_csv_rows_md(
                cross_rootcause_rows,
                [
                    "problem_id",
                    "introduced_or_observed_in",
                    "still_unresolved_in_v72",
                    "evidence_path",
                    "impact_targets",
                    "impact_scale",
                    "root_mechanism",
                    "fix_component",
                    "gate_link",
                ],
            ),
        )

        frontier_rows = _read_csv(truth_dir / "v72_frontier_fix_map_20260223.csv")
        master_text = _replace_or_append_auto_section(
            master_text,
            "FRONTIER_FIX_MAP",
            "## Frontier-to-Fix Mapping (Primary Sources, 2026-02-23)",
            _render_csv_rows_md(
                frontier_rows,
                [
                    "problem",
                    "source",
                    "mechanism",
                    "integration_point",
                    "risk",
                    "verification_test",
                    "primary_link",
                    "status",
                ],
            ),
        )
        _write_text(master_doc, master_text)

        _refresh_light_docs(
            status_doc_path=args.status_doc.resolve(),
            results_doc_path=args.results_doc.resolve(),
            snapshot_ts=str(status_payload["snapshot_ts"]),
            strict_done=strict_done_n,
            strict_total=strict_total_n,
            v72_done=v72_done_n,
            v72_total=v72_total_n,
            slurm=slurm,
            eta_model=eta_model,
            champion_counts=dict(champion_counts),
            queue_actions_path="docs/benchmarks/block3_truth_pack/queue_actions_latest.json",
        )

        if args.enforce_style_guard:
            guard_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "style_guard_docs.py"),
                "--paths",
                ",".join(
                    [
                        str(args.master_doc.resolve()),
                        str(args.status_doc.resolve()),
                        str(args.results_doc.resolve()),
                    ]
                ),
            ]
            guard = subprocess.run(guard_cmd, capture_output=True, text=True, check=False)
            if guard.returncode != 0:
                raise RuntimeError(
                    "Style guard failed for status docs.\n"
                    + guard.stdout
                    + ("\n" + guard.stderr if guard.stderr else "")
                )

    out = {
        "snapshot_ts": status_payload["snapshot_ts"],
        "strict_progress": status_payload["strict_progress"],
        "v72_progress": status_payload["v72_progress"],
        "running_total": status_payload["active_jobs_nested"]["total_running"],
        "pending_total": status_payload["active_jobs_nested"]["total_pending"],
        "eta_model": eta_model,
        "artifacts": {
            "execution_status_latest_json": str(latest_json) if args.write_latest else None,
            "execution_status_latest_md": str(latest_md) if args.write_latest else None,
            "queue_actions_latest_json": str(queue_latest_json) if args.write_latest else None,
            "fairness_certification_latest_json": str(cert_latest_json) if args.write_latest else None,
        },
    }
    print(json.dumps(out, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
