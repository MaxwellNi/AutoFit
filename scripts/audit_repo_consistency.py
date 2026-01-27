from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".md", ".toml", ".json"}
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    "node_modules",
    "runs",
    "data",
}
IGNORE_DATA_SUBDIRS = {"raw", "processed", "final"}


@dataclass
class Finding:
    file: str
    line: int
    text: str


def _iter_files(base: Path) -> Iterable[Path]:
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        parts = set(path.parts)
        if parts & IGNORE_DIRS:
            if "data" in parts and not (parts & IGNORE_DATA_SUBDIRS):
                pass
            else:
                continue
        yield path


def _scan_patterns(paths: Iterable[Path], patterns: List[re.Pattern]) -> List[Finding]:
    findings: List[Finding] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for idx, line in enumerate(text, start=1):
            for pat in patterns:
                if pat.search(line):
                    findings.append(Finding(str(path), idx, line.strip()))
                    break
    return findings


def audit_parquet_only(repo_root: Path) -> Dict[str, object]:
    scan_dirs = [repo_root / "src", repo_root / "scripts", repo_root / "configs"]
    patterns = [
        re.compile(r"load_offers_csv"),
        re.compile(r"read_csv\("),
        re.compile(r"--csv_path"),
        re.compile(r"csv_path"),
    ]
    findings = []
    for d in scan_dirs:
        if not d.exists():
            continue
        for f in _scan_patterns(_iter_files(d), patterns):
            if f.file.endswith("audit_repo_consistency.py"):
                continue
            findings.append(f)

        # heuristic scan for raw .csv paths
        for f in _scan_patterns(_iter_files(d), [re.compile(r"\.csv")]):
            if f.file.endswith("audit_repo_consistency.py"):
                continue
            line = f.text
            if "to_csv" in line:
                findings.append(f)
                continue
            if "data/raw" in line or "offers_current" in line or "input_path" in line:
                findings.append(f)
    return {"issues": [f.__dict__ for f in findings], "count": len(findings)}


def audit_aws_keys(repo_root: Path) -> Dict[str, object]:
    patterns = [
        re.compile(r"AKIA[0-9A-Z]{6,}"),
        re.compile(r"X-Amz-Credential="),
        re.compile(r"X-Amz-Signature="),
    ]
    secret_patterns = [
        re.compile(r"AWS_SECRET_ACCESS_KEY\s*=\s*[\"']"),
        re.compile(r"AWS_ACCESS_KEY_ID\s*=\s*[\"']"),
    ]

    findings = [
        f for f in _scan_patterns(_iter_files(repo_root), patterns)
        if not f.file.endswith("audit_repo_consistency.py")
    ]
    secret_findings = [
        f for f in _scan_patterns(_iter_files(repo_root), secret_patterns)
        if not f.file.endswith("audit_repo_consistency.py")
    ]

    return {
        "token_hits": [f.__dict__ for f in findings],
        "hardcoded_key_hits": [f.__dict__ for f in secret_findings],
        "token_count": len(findings),
        "hardcoded_count": len(secret_findings),
    }


def audit_official_scripts(repo_root: Path) -> Dict[str, object]:
    scripts = [
        "scripts/run_full_benchmark.py",
        "scripts/run_auto_fit.py",
        "scripts/train_complete_pipeline.py",
        "scripts/run_official_benchmark.py",
        "scripts/run_official_models.py",
        "scripts/test_local_small_scale.py",
    ]
    results = {}
    for script in scripts:
        path = repo_root / script
        if not path.exists():
            results[script] = {"exists": False}
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        results[script] = {
            "exists": True,
            "uses_parquet_catalog": "parquet_catalog" in text or "scan_snapshots" in text,
            "uses_load_offers_csv": "load_offers_csv" in text or "read_csv(" in text,
        }
    return results


def audit_paper_tables(repo_root: Path) -> Dict[str, object]:
    runs_dir = repo_root / "runs"
    required = [
        "main_results.parquet",
        "ablation.parquet",
        "faithfulness.parquet",
        "efficiency.parquet",
    ]
    entries = []
    if runs_dir.exists():
        for pt in runs_dir.rglob("paper_tables"):
            if not pt.is_dir():
                continue
            missing = [f for f in required if not (pt / f).exists()]
            entries.append(
                {
                    "path": str(pt),
                    "missing": missing,
                    "ok": len(missing) == 0,
                }
            )
    return {"entries": entries, "count": len(entries)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Repo consistency audit")
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    out_dir = args.output_dir or (repo_root / "runs" / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    report = {
        "parquet_only": audit_parquet_only(repo_root),
        "aws_keys": audit_aws_keys(repo_root),
        "official_scripts": audit_official_scripts(repo_root),
        "paper_tables": audit_paper_tables(repo_root),
    }

    json_path = out_dir / "audit_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = ["# Repo Audit Report", ""]
    md_lines.append(f"- output_dir: {out_dir}")
    md_lines.append("")
    md_lines.append("## parquet-only")
    md_lines.append(f"- issues: {report['parquet_only']['count']}")
    for item in report["parquet_only"]["issues"]:
        md_lines.append(f"  - {item['file']}:{item['line']} | {item['text']}")
    md_lines.append("")
    md_lines.append("## aws-keys")
    md_lines.append(f"- token_hits: {report['aws_keys']['token_count']}")
    for item in report["aws_keys"]["token_hits"]:
        md_lines.append(f"  - {item['file']}:{item['line']} | {item['text']}")
    md_lines.append(f"- hardcoded_key_hits: {report['aws_keys']['hardcoded_count']}")
    for item in report["aws_keys"]["hardcoded_key_hits"]:
        md_lines.append(f"  - {item['file']}:{item['line']} | {item['text']}")
    md_lines.append("")
    md_lines.append("## official_scripts")
    for script, info in report["official_scripts"].items():
        md_lines.append(f"- {script}: {info}")
    md_lines.append("")
    md_lines.append("## paper_tables")
    md_lines.append(f"- entries: {report['paper_tables']['count']}")
    for entry in report["paper_tables"]["entries"]:
        md_lines.append(f"  - {entry['path']} | ok={entry['ok']} | missing={entry['missing']}")

    md_path = out_dir / "audit_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    log_path = out_dir / "logs" / "audit.log"
    log_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Audit report saved to: {json_path}")
    print(f"Audit markdown saved to: {md_path}")
    print(f"Audit log saved to: {log_path}")


if __name__ == "__main__":
    main()
