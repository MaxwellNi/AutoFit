#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_MD = REPO_ROOT / "docs" / "RUN_QUEUE_PROGRESS_CURRENT.md"


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True)


def _parse_elapsed_to_seconds(text: str) -> int:
    parts = text.strip().split("-")
    days = 0
    hhmmss = parts[-1]
    if len(parts) == 2:
        days = int(parts[0])
    fields = [int(x) for x in hhmmss.split(":")]
    if len(fields) == 3:
        h, m, s = fields
    elif len(fields) == 2:
        h = 0
        m, s = fields
    else:
        raise ValueError(f"Unsupported elapsed format: {text}")
    return days * 86400 + h * 3600 + m * 60 + s


def _format_seconds(seconds: int) -> str:
    days, rem = divmod(max(0, seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days > 0:
        return f"{days}d{hours:02d}h{minutes:02d}m"
    return f"{hours}h{minutes:02d}m"


def _parse_jobs() -> list[dict[str, str]]:
    raw = _run(
        [
            "squeue",
            "-u",
            "npin",
            "-h",
            "-o",
            "%i|%P|%j|%T|%M|%l|%D|%R",
        ]
    )
    jobs: list[dict[str, str]] = []
    for line in raw.splitlines():
        jobid, partition, name, state, elapsed, limit, nodes, reason = line.split("|", 7)
        jobs.append(
            {
                "jobid": jobid.strip(),
                "partition": partition.strip(),
                "name": name.strip(),
                "state": state.strip(),
                "elapsed": elapsed.strip(),
                "limit": limit.strip(),
                "nodes": nodes.strip(),
                "reason": reason.strip(),
            }
        )
    return jobs


def _row_for_running(job: dict[str, str]) -> dict[str, str]:
    elapsed_s = _parse_elapsed_to_seconds(job["elapsed"])
    limit_s = _parse_elapsed_to_seconds(job["limit"])
    progress = (elapsed_s / limit_s * 100.0) if limit_s > 0 else 0.0
    return {
        **job,
        "progress": f"{progress:.1f}%",
        "time_left": _format_seconds(limit_s - elapsed_s),
    }


def _write_md(jobs: list[dict[str, str]]) -> None:
    running = [j for j in jobs if j["state"] == "RUNNING"]
    pending = [j for j in jobs if j["state"] == "PENDING"]
    by_part = {
        "gpu": {"running": 0, "pending": 0},
        "l40s": {"running": 0, "pending": 0},
        "hopper": {"running": 0, "pending": 0},
    }
    for j in jobs:
        part = j["partition"]
        if part not in by_part:
            by_part[part] = {"running": 0, "pending": 0}
        key = "running" if j["state"] == "RUNNING" else "pending"
        by_part[part][key] += 1

    lines = [
        "# Current Queue Progress",
        "",
        f"> Snapshot time: {dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "> Source: live `squeue -u npin`.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total jobs | {len(jobs)} |",
        f"| Running | {len(running)} |",
        f"| Pending | {len(pending)} |",
        f"| gpu running | {by_part.get('gpu', {}).get('running', 0)} |",
        f"| gpu pending | {by_part.get('gpu', {}).get('pending', 0)} |",
        f"| l40s running | {by_part.get('l40s', {}).get('running', 0)} |",
        f"| l40s pending | {by_part.get('l40s', {}).get('pending', 0)} |",
        f"| hopper running | {by_part.get('hopper', {}).get('running', 0)} |",
        f"| hopper pending | {by_part.get('hopper', {}).get('pending', 0)} |",
        "",
        "## Running Jobs",
        "",
        "| jobid | job | partition | elapsed | limit | progress | time left | node |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for job in sorted(running, key=lambda j: (_parse_elapsed_to_seconds(j["elapsed"]), j["jobid"]), reverse=True):
        row = _row_for_running(job)
        lines.append(
            f"| {row['jobid']} | {row['name']} | {row['partition']} | {row['elapsed']} | {row['limit']} | {row['progress']} | {row['time_left']} | {row['reason']} |"
        )

    lines.extend(
        [
            "",
            "## Pending Jobs",
            "",
            "| jobid | job | partition | state | limit | reason |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for job in sorted(pending, key=lambda j: (j["partition"], j["jobid"])):
        lines.append(
            f"| {job['jobid']} | {job['name']} | {job['partition']} | {job['state']} | {job['limit']} | {job['reason']} |"
        )

    OUT_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    jobs = _parse_jobs()
    _write_md(jobs)
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
