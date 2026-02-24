#!/usr/bin/env python3
"""Fail-fast assertion for the mandatory Block3 execution contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


POLICY_TEXT = (
    "ONE-LINER TS AGENT PROMPT: Build time-series models ONLY with provable "
    "zero-leakage (feature availability-time ≤ t_obs_end, correct label alignment to "
    "t_target, no centered/forward ops, train-only fit for any "
    "stats/preprocess/feature-select/PCA/outlier rules, as-of snapshots for revisable "
    "data, no entity/event/near-duplicate split leakage), strict train==infer parity "
    "(same pipeline + cache keyed by data_version/split/preproc_hash, eval in deployed "
    "inference mode rolling/recursive, correct train/eval modes incl. dropout/BN, "
    "past-only padding/interpolation, toy unit tests for window indexing), and "
    "deployment-real evaluation evidence (overall + sliced metrics with worst-slice, "
    "tail+business metrics, calibration/coverage if probabilistic, drift monitor + "
    "retrain/rollback plan) with full reproducibility logs (seeds/env/hashes/commit, "
    "leakage+split audits, reproducible run command); if any requirement can’t be met, "
    "STOP and report the exact blocker."
)
CONTRACT_VERSION = "2026-02-24"
POLICY_HASH = hashlib.sha256(POLICY_TEXT.encode("utf-8")).hexdigest()
REQUIRED_SECTIONS = [
    "## Scope",
    "## Non-Negotiable Controls",
    "## Runtime Requirements",
    "## Leakage and Split Audit Requirements",
    "## Train-Infer Parity Requirements",
    "## Reproducibility Requirements",
    "## Forbidden Run Modes",
    "## Blocker-Stop Policy",
]
FORBIDDEN_ENV_FLAGS = [
    "ALLOW_UNSAFE_SKIP_PREFLIGHT",
    "BLOCK3_DISABLE_LEAKAGE_CHECKS",
    "BLOCK3_DISABLE_SPLIT_AUDIT",
    "BLOCK3_DISABLE_COVERAGE_GUARD",
    "BLOCK3_ALLOW_TEST_FEEDBACK",
    "BLOCK3_USE_TEST_FOR_SELECTION",
    "BLOCK3_SKIP_EXECUTION_CONTRACT",
]


def _truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ContractAudit:
    generated_at_utc: str
    entrypoint: str
    contract_path: str
    contract_version: str
    policy_hash: str
    contract_version_match: bool
    policy_hash_match: bool
    policy_text_present: bool
    required_sections_present: bool
    missing_sections: List[str]
    python_executable: str
    python_version: str
    insider_runtime: bool
    python_version_ok: bool
    forbidden_env_flags: Dict[str, str]
    no_forbidden_flags: bool
    pass_all: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "entrypoint": self.entrypoint,
            "contract_path": self.contract_path,
            "contract_version": self.contract_version,
            "policy_hash": self.policy_hash,
            "contract_version_match": self.contract_version_match,
            "policy_hash_match": self.policy_hash_match,
            "policy_text_present": self.policy_text_present,
            "required_sections_present": self.required_sections_present,
            "missing_sections": self.missing_sections,
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "insider_runtime": self.insider_runtime,
            "python_version_ok": self.python_version_ok,
            "forbidden_env_flags": self.forbidden_env_flags,
            "no_forbidden_flags": self.no_forbidden_flags,
            "pass_all": self.pass_all,
        }


def _parse_declared_value(doc_text: str, prefix: str) -> str:
    for line in doc_text.splitlines():
        if prefix in line:
            idx = line.find("`")
            if idx >= 0 and line.endswith("`"):
                return line[idx + 1 : -1]
    return ""


def _runtime_ok(require_insider: bool) -> Tuple[bool, bool, str]:
    py_ok = (sys.version_info.major, sys.version_info.minor) >= (3, 11)
    exe = sys.executable
    insider = ("insider" in exe) if exe else False
    if not require_insider:
        insider = True
    return insider, py_ok, exe


def build_audit(repo_root: Path, entrypoint: str, require_insider: bool) -> ContractAudit:
    contract_path = repo_root / "docs" / "BLOCK3_EXECUTION_CONTRACT.md"
    if not contract_path.exists():
        raise SystemExit(f"FATAL: contract file missing: {contract_path}")

    doc_text = contract_path.read_text(encoding="utf-8")
    declared_version = _parse_declared_value(doc_text, "Contract-Version:")
    declared_hash = _parse_declared_value(doc_text, "Policy-Hash-SHA256:")
    version_match = declared_version == CONTRACT_VERSION
    hash_match = declared_hash == POLICY_HASH
    policy_present = POLICY_TEXT in doc_text

    missing_sections = [s for s in REQUIRED_SECTIONS if s not in doc_text]
    sections_present = not missing_sections

    insider_runtime, python_version_ok, py_exe = _runtime_ok(require_insider=require_insider)
    forbidden = {k: os.environ.get(k, "") for k in FORBIDDEN_ENV_FLAGS if _truthy(os.environ.get(k))}
    no_forbidden = not forbidden

    pass_all = all(
        [
            version_match,
            hash_match,
            policy_present,
            sections_present,
            insider_runtime,
            python_version_ok,
            no_forbidden,
        ]
    )

    return ContractAudit(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        entrypoint=entrypoint,
        contract_path=str(contract_path),
        contract_version=declared_version,
        policy_hash=declared_hash,
        contract_version_match=version_match,
        policy_hash_match=hash_match,
        policy_text_present=policy_present,
        required_sections_present=sections_present,
        missing_sections=missing_sections,
        python_executable=py_exe,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        insider_runtime=insider_runtime,
        python_version_ok=python_version_ok,
        forbidden_env_flags=forbidden,
        no_forbidden_flags=no_forbidden,
        pass_all=pass_all,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Assert mandatory Block3 execution contract.")
    parser.add_argument("--entrypoint", type=str, default="unknown")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("docs/benchmarks/block3_truth_pack/v72_runtime_contract_audit.json"),
    )
    parser.add_argument(
        "--allow-non-insider",
        action="store_true",
        help="Only for metadata-only checks; production paths must not use this.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    audit = build_audit(
        repo_root=repo_root,
        entrypoint=args.entrypoint,
        require_insider=not args.allow_non_insider,
    )

    out_path = args.audit_json
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit.to_dict(), indent=2), encoding="utf-8")

    if not audit.pass_all:
        print("Block3 execution contract assertion FAILED")
        print(json.dumps(audit.to_dict(), indent=2))
        raise SystemExit(2)

    print("Block3 execution contract assertion PASS")
    print(json.dumps(audit.to_dict(), indent=2))


if __name__ == "__main__":
    main()
