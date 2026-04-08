"""Machine-readable registry helpers for official-source verification status."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_SOURCE_REGISTRY = (
    REPO_ROOT / "configs" / "single_model_true_champion" / "official_source_registry.yaml"
)

VERIFIED_STATUSES = {"verified", "verified_partial"}
ALLOWED_STATUSES = VERIFIED_STATUSES | {"pending_recheck", "research_only"}


def load_official_source_registry(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_OFFICIAL_SOURCE_REGISTRY
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [key for key in ("metadata", "guardrails", "entries") if key not in data]
    if missing:
        raise ValueError(f"official-source registry missing keys: {', '.join(missing)}")
    return data


def validate_official_source_registry(registry: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    entries = registry.get("entries", {})
    if not isinstance(entries, dict) or not entries:
        errors.append("official-source registry must contain a non-empty entries mapping")
        return errors

    for signal_name, entry in entries.items():
        status = str(entry.get("status", "")).strip()
        if status not in ALLOWED_STATUSES:
            errors.append(f"{signal_name}: unknown status '{status}'")
        urls = [entry.get("official_page"), entry.get("official_pdf"), entry.get("official_repo")]
        if status in VERIFIED_STATUSES and not any(urls):
            errors.append(f"{signal_name}: verified entries must include at least one official URL")
        if status == "pending_recheck" and not str(entry.get("verification_gap", "")).strip():
            errors.append(f"{signal_name}: pending_recheck entries must document verification_gap")
        if entry.get("conservative_wording") is None:
            errors.append(f"{signal_name}: conservative_wording must be explicit")
    return errors


def build_official_source_rows(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for signal_name in sorted(registry.get("entries", {}).keys(), key=str.casefold):
        entry = registry["entries"][signal_name]
        rows.append(
            {
                "signal": signal_name,
                "status": str(entry.get("status", "")),
                "venue": str(entry.get("venue", "")),
                "year": int(entry.get("year", 0)) if entry.get("year") is not None else None,
                "official_title": str(entry.get("official_title", signal_name)),
                "official_page": str(entry.get("official_page", "")),
                "official_pdf": str(entry.get("official_pdf", "")),
                "official_repo": str(entry.get("official_repo", "")),
                "verification_basis": str(entry.get("verification_basis", "")),
                "verification_gap": str(entry.get("verification_gap", "")),
                "conservative_wording": bool(entry.get("conservative_wording", True)),
                "mechanism_tags": list(entry.get("mechanism_tags", [])),
            }
        )
    return rows


def summarize_official_source_registry(registry: Dict[str, Any]) -> Dict[str, Any]:
    rows = build_official_source_rows(registry)
    status_counts: Dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    return {
        "registry_name": str(registry.get("metadata", {}).get("name", "official_source_registry")),
        "registry_version": str(registry.get("metadata", {}).get("version", "")),
        "entry_count": len(rows),
        "status_counts": status_counts,
        "pending_recheck_count": status_counts.get("pending_recheck", 0),
        "verified_count": sum(status_counts.get(status, 0) for status in VERIFIED_STATUSES),
    }