#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--h1-job-id", type=int, default=5310439)
    ap.add_argument("--min-h1-wins", type=int, default=5, help="Promotion threshold for the 12-cell h1 gate (>4/12).")
    ap.add_argument("--min-full-wins", type=int, default=13, help="Promotion threshold for the 48-cell full investors gate (>12/48).")
    ap.add_argument(
        "--state-json",
        type=Path,
        default=REPO_ROOT / "runs" / "benchmarks" / "v741_lite_20260406" / "v741_lite_ladder_state_20260406.json",
    )
    ap.add_argument(
        "--state-md",
        type=Path,
        default=REPO_ROOT / "runs" / "benchmarks" / "v741_lite_20260406" / "v741_lite_ladder_status_20260406.md",
    )
    ap.add_argument(
        "--h1-summary-md",
        type=Path,
        default=REPO_ROOT / "docs" / "references" / "V741_LITE_SHARED112_INVESTORS_H1_GATE_20260406.md",
    )
    ap.add_argument(
        "--full-summary-md",
        type=Path,
        default=REPO_ROOT / "docs" / "references" / "V741_LITE_SHARED112_INVESTORS_FULL_GATE_20260406.md",
    )
    ap.add_argument(
        "--binary-summary-md",
        type=Path,
        default=REPO_ROOT / "docs" / "references" / "V741_LITE_SHARED112_BINARY_GUARD_20260406.md",
    )
    ap.add_argument(
        "--funding-summary-md",
        type=Path,
        default=REPO_ROOT / "docs" / "references" / "V741_LITE_SHARED112_FUNDING_GUARD_20260406.md",
    )
    ap.add_argument(
        "--full-script",
        type=Path,
        default=REPO_ROOT / "scripts" / "slurm" / "v741_lite" / "v741_lite_shared112_investors_full_gate_gpu.sh",
    )
    ap.add_argument(
        "--binary-script",
        type=Path,
        default=REPO_ROOT / "scripts" / "slurm" / "v741_lite" / "v741_lite_shared112_binary_guard_gpu.sh",
    )
    ap.add_argument(
        "--funding-script",
        type=Path,
        default=REPO_ROOT / "scripts" / "slurm" / "v741_lite" / "v741_lite_shared112_funding_guard_gpu.sh",
    )
    return ap.parse_args()


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(args: argparse.Namespace, state: Dict[str, Any]) -> None:
    args.state_json.parent.mkdir(parents=True, exist_ok=True)
    args.state_md.parent.mkdir(parents=True, exist_ok=True)
    args.state_json.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# V741-Lite Ladder Status",
        "",
        f"- Updated: `{_utc_now()}`",
        f"- Decision: `{state.get('decision', 'running')}`",
        f"- Poll seconds: `{args.poll_seconds}`",
        "",
        "## Thresholds",
        "",
        f"- h1 promotion threshold: `>= {args.min_h1_wins}` wins",
        f"- full investors promotion threshold: `>= {args.min_full_wins}` wins",
        "",
        "## Steps",
        "",
    ]
    for key in ("h1_gate", "full_investors", "binary_guard", "funding_guard"):
        step = state.get("steps", {}).get(key, {})
        if not step:
            lines.append(f"- `{key}`: pending")
            continue
        bits = [f"state={step.get('job_state', step.get('status', '-'))}"]
        if step.get("job_id") is not None:
            bits.append(f"job_id={step['job_id']}")
        if step.get("wins") is not None:
            bits.append(f"wins={step['wins']}")
        if step.get("ties") is not None:
            bits.append(f"ties={step['ties']}")
        if step.get("losses") is not None:
            bits.append(f"losses={step['losses']}")
        if step.get("summary_md"):
            bits.append(f"summary={step['summary_md']}")
        lines.append(f"- `{key}`: " + ", ".join(bits))
    args.state_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _parse_summary_counts(summary_path: Path) -> Optional[Dict[str, int]]:
    if not summary_path.exists():
        return None
    text = summary_path.read_text(encoding="utf-8")
    wins = re.search(r"- Wins: `([0-9]+)`", text)
    ties = re.search(r"- Ties: `([0-9]+)`", text)
    losses = re.search(r"- Losses: `([0-9]+)`", text)
    if not wins or not ties or not losses:
        return None
    return {
        "wins": int(wins.group(1)),
        "ties": int(ties.group(1)),
        "losses": int(losses.group(1)),
    }


def _job_state(job_id: int) -> Optional[str]:
    squeue = _run(["squeue", "-j", str(job_id), "-h", "-o", "%T"])
    if squeue.returncode == 0 and squeue.stdout.strip():
        return squeue.stdout.strip().splitlines()[0].strip()

    sacct = _run(["sacct", "-j", str(job_id), "-n", "-X", "-o", "JobIDRaw,State", "-P"])
    if sacct.returncode != 0:
        return None
    for line in sacct.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 1)
        if len(parts) != 2:
            continue
        job_id_raw, state = parts
        if job_id_raw.strip() == str(job_id):
            return state.strip().split()[0]
    return None


def _submit_job(script_path: Path) -> int:
    proc = _run(["sbatch", str(script_path)])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"sbatch failed for {script_path}")
    match = re.search(r"Submitted batch job ([0-9]+)", proc.stdout)
    if not match:
        raise RuntimeError(f"Could not parse sbatch output: {proc.stdout.strip()}")
    return int(match.group(1))


def _wait_for_completion(job_id: int, label: str, args: argparse.Namespace, state: Dict[str, Any], step_key: str) -> str:
    terminal_states = {
        "COMPLETED",
        "FAILED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "CANCELLED",
        "PREEMPTED",
        "NODE_FAIL",
        "BOOT_FAIL",
        "DEADLINE",
    }
    while True:
        job_state = _job_state(job_id)
        step = state.setdefault("steps", {}).setdefault(step_key, {})
        step["job_state"] = job_state or "UNKNOWN"
        step["checked_at"] = _utc_now()
        _write_state(args, state)
        if job_state in terminal_states:
            print(f"[v741-lite-ladder] {label} job {job_id} finished with state={job_state}", flush=True)
            return job_state
        print(f"[v741-lite-ladder] waiting for {label} job {job_id}, state={job_state or 'UNKNOWN'}", flush=True)
        time.sleep(args.poll_seconds)


def _ensure_step(
    step_key: str,
    label: str,
    summary_md: Path,
    args: argparse.Namespace,
    state: Dict[str, Any],
    script_path: Optional[Path] = None,
    existing_job_id: Optional[int] = None,
) -> Dict[str, Any]:
    step = state.setdefault("steps", {}).setdefault(step_key, {})
    step["summary_md"] = str(summary_md)
    parsed = _parse_summary_counts(summary_md)
    if parsed is not None:
        step.update(parsed)
        step["status"] = "summary_present"
        step["job_state"] = step.get("job_state", "COMPLETED")
        _write_state(args, state)
        return step

    job_id = existing_job_id or step.get("job_id")
    if job_id is None:
        if script_path is None:
            raise RuntimeError(f"No job id or script for step {step_key}")
        job_id = _submit_job(script_path)
        print(f"[v741-lite-ladder] submitted {label} job {job_id}", flush=True)
        step["job_id"] = job_id
        step["submitted_at"] = _utc_now()
        _write_state(args, state)

    terminal_state = _wait_for_completion(int(job_id), label, args, state, step_key)
    step["job_id"] = int(job_id)
    step["job_state"] = terminal_state
    if terminal_state != "COMPLETED":
        step["status"] = "job_failed"
        _write_state(args, state)
        return step

    parsed = _parse_summary_counts(summary_md)
    if parsed is None:
        step["status"] = "completed_missing_summary"
        _write_state(args, state)
        return step

    step.update(parsed)
    step["status"] = "completed"
    _write_state(args, state)
    return step


def main() -> int:
    args = _parse_args()
    state = _load_state(args.state_json)
    state.setdefault("started_at", _utc_now())
    state["last_started_at"] = _utc_now()
    state["thresholds"] = {
        "min_h1_wins": args.min_h1_wins,
        "min_full_wins": args.min_full_wins,
    }
    _write_state(args, state)

    h1 = _ensure_step(
        step_key="h1_gate",
        label="h1_gate",
        summary_md=args.h1_summary_md,
        args=args,
        state=state,
        existing_job_id=args.h1_job_id,
    )
    if h1.get("job_state") != "COMPLETED":
        state["decision"] = "stop_h1_job_failure"
        _write_state(args, state)
        return 1
    if int(h1.get("wins", 0)) < args.min_h1_wins:
        state["decision"] = "stop_h1_threshold"
        _write_state(args, state)
        print(
            f"[v741-lite-ladder] h1 gate stayed below threshold: wins={h1.get('wins', 0)} < {args.min_h1_wins}",
            flush=True,
        )
        return 2

    full_step = _ensure_step(
        step_key="full_investors",
        label="full_investors",
        summary_md=args.full_summary_md,
        args=args,
        state=state,
        script_path=args.full_script,
    )
    if full_step.get("job_state") != "COMPLETED":
        state["decision"] = "stop_full_job_failure"
        _write_state(args, state)
        return 3
    if int(full_step.get("wins", 0)) < args.min_full_wins:
        state["decision"] = "stop_full_threshold"
        _write_state(args, state)
        print(
            f"[v741-lite-ladder] full investors gate stayed below threshold: wins={full_step.get('wins', 0)} < {args.min_full_wins}",
            flush=True,
        )
        return 4

    _ensure_step(
        step_key="binary_guard",
        label="binary_guard",
        summary_md=args.binary_summary_md,
        args=args,
        state=state,
        script_path=args.binary_script,
    )
    _ensure_step(
        step_key="funding_guard",
        label="funding_guard",
        summary_md=args.funding_summary_md,
        args=args,
        state=state,
        script_path=args.funding_script,
    )

    state["decision"] = "guards_completed"
    state["completed_at"] = _utc_now()
    _write_state(args, state)
    print("[v741-lite-ladder] ladder completed through guard submissions/results", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())