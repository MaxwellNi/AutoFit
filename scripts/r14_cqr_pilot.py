#!/usr/bin/env python3
"""Diagnostic CQR-lite asymmetric interval pilot from landed predictions."""

from __future__ import annotations

from r14_interval_pilot_lib import evaluate, print_summary, write_report


def main() -> int:
    report = evaluate("cqr_lite")
    out_json, out_md = write_report(report, "r14_cqr_pilot")
    print_summary(report, out_json, out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())