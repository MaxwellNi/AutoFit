"""Registry helpers for the repo-tracked public-pack dataset scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PUBLIC_PACK_CONFIG = REPO_ROOT / "configs" / "public_pack" / "dataset_families.yaml"


@dataclass(frozen=True)
class PublicPackCell:
    """A runner-ready cell expanded from the public-pack family scaffold."""

    pack: str
    family: str
    display_name: str
    variant: str
    task_type: str
    context_length: Optional[int]
    prediction_length: Optional[int]
    verification_tier: str
    enabled: bool
    raw_root: str
    processed_root: str
    cache_root: str
    official_source_signals: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack": self.pack,
            "family": self.family,
            "display_name": self.display_name,
            "variant": self.variant,
            "task_type": self.task_type,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "verification_tier": self.verification_tier,
            "enabled": self.enabled,
            "raw_root": self.raw_root,
            "processed_root": self.processed_root,
            "cache_root": self.cache_root,
            "official_source_signals": list(self.official_source_signals),
        }


def _resolve_repo_relative(path_text: str) -> Path:
    return (REPO_ROOT / path_text).resolve()


def load_public_pack_registry(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_PUBLIC_PACK_CONFIG
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [key for key in ("metadata", "guardrails", "packs", "families") if key not in data]
    if missing:
        raise ValueError(f"public-pack config missing keys: {', '.join(missing)}")
    family_names = set(data["families"].keys())
    for pack_name, pack_spec in data["packs"].items():
        for family_name in pack_spec.get("families", []):
            if family_name not in family_names:
                raise ValueError(
                    f"public-pack pack '{pack_name}' references unknown family '{family_name}'"
                )
    return data


def _matching_pack_names(registry: Dict[str, Any], family_name: str) -> List[str]:
    matched: List[str] = []
    for pack_name, pack_spec in registry.get("packs", {}).items():
        families = pack_spec.get("families", [])
        if family_name in families:
            matched.append(str(pack_name))
    return matched


def _family_matches_request(requested: str, family_name: str, family_spec: Dict[str, Any]) -> bool:
    requested_norm = requested.strip().casefold()
    if requested_norm == family_name.casefold():
        return True
    display_name = str(family_spec.get("display_name", "")).strip().casefold()
    if requested_norm == display_name:
        return True
    aliases = [str(alias).strip().casefold() for alias in family_spec.get("aliases", [])]
    return requested_norm in aliases


def _resolve_requested_family_names(
    registry: Dict[str, Any],
    requested_families: Optional[Sequence[str]],
) -> Optional[set]:
    if not requested_families:
        return None
    resolved: set = set()
    for requested in requested_families:
        matched = False
        for family_name, family_spec in registry.get("families", {}).items():
            if _family_matches_request(requested, str(family_name), family_spec):
                resolved.add(str(family_name))
                matched = True
        if not matched:
            raise ValueError(f"unknown public-pack family selector: {requested}")
    return resolved


def _selected_families(
    registry: Dict[str, Any],
    pack: str = "",
    requested_families: Optional[Sequence[str]] = None,
    enabled_only: bool = False,
    verification_tier: str = "",
) -> List[Tuple[str, str, Dict[str, Any]]]:
    requested_names = _resolve_requested_family_names(registry, requested_families)
    pack_filter = pack.strip()
    if pack_filter:
        pack_spec = registry.get("packs", {}).get(pack_filter)
        if pack_spec is None:
            raise ValueError(f"unknown public-pack pack: {pack_filter}")
        pack_family_names = set(pack_spec.get("families", []))
    else:
        pack_family_names = None

    selected: List[Tuple[str, str, Dict[str, Any]]] = []
    for family_name, family_spec in registry.get("families", {}).items():
        if pack_family_names is not None and family_name not in pack_family_names:
            continue
        if requested_names is not None and family_name not in requested_names:
            continue
        if enabled_only and not bool(family_spec.get("enabled", False)):
            continue
        if verification_tier and str(family_spec.get("verification_tier", "")) != verification_tier:
            continue
        pack_names = _matching_pack_names(registry, str(family_name))
        pack_name = pack_filter or (pack_names[0] if pack_names else "unassigned")
        selected.append((pack_name, str(family_name), family_spec))
    return selected


def expand_public_pack_cells(
    registry: Dict[str, Any],
    pack: str = "",
    requested_families: Optional[Sequence[str]] = None,
    enabled_only: bool = False,
    verification_tier: str = "",
) -> List[PublicPackCell]:
    cells: List[PublicPackCell] = []
    for pack_name, family_name, family_spec in _selected_families(
        registry,
        pack=pack,
        requested_families=requested_families,
        enabled_only=enabled_only,
        verification_tier=verification_tier,
    ):
        protocol = family_spec.get("representative_protocol", {})
        variants = list(family_spec.get("variants") or [family_spec.get("display_name", family_name)])
        context_lengths = list(protocol.get("context_lengths") or [None])
        prediction_lengths = list(protocol.get("prediction_lengths") or [None])
        for variant in variants:
            for context_length in context_lengths:
                for prediction_length in prediction_lengths:
                    cells.append(
                        PublicPackCell(
                            pack=pack_name,
                            family=family_name,
                            display_name=str(family_spec.get("display_name", family_name)),
                            variant=str(variant),
                            task_type=str(family_spec.get("task_type", "unknown")),
                            context_length=int(context_length) if context_length is not None else None,
                            prediction_length=int(prediction_length) if prediction_length is not None else None,
                            verification_tier=str(family_spec.get("verification_tier", "")),
                            enabled=bool(family_spec.get("enabled", False)),
                            raw_root=str(family_spec.get("dataset_roots", {}).get("raw", "")),
                            processed_root=str(family_spec.get("dataset_roots", {}).get("processed", "")),
                            cache_root=str(family_spec.get("dataset_roots", {}).get("cache", "")),
                            official_source_signals=tuple(
                                str(signal) for signal in family_spec.get("official_source_signals", [])
                            ),
                        )
                    )
    return cells


def validate_public_pack_roots(
    registry: Dict[str, Any],
    pack: str = "",
    requested_families: Optional[Sequence[str]] = None,
    enabled_only: bool = False,
    verification_tier: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pack_name, family_name, family_spec in _selected_families(
        registry,
        pack=pack,
        requested_families=requested_families,
        enabled_only=enabled_only,
        verification_tier=verification_tier,
    ):
        roots = family_spec.get("dataset_roots", {})
        raw_root = str(roots.get("raw", ""))
        processed_root = str(roots.get("processed", ""))
        cache_root = str(roots.get("cache", ""))
        rows.append(
            {
                "pack": pack_name,
                "family": family_name,
                "display_name": str(family_spec.get("display_name", family_name)),
                "enabled": bool(family_spec.get("enabled", False)),
                "verification_tier": str(family_spec.get("verification_tier", "")),
                "raw_root": raw_root,
                "raw_exists": _resolve_repo_relative(raw_root).exists() if raw_root else False,
                "processed_root": processed_root,
                "processed_exists": _resolve_repo_relative(processed_root).exists() if processed_root else False,
                "cache_root": cache_root,
                "cache_exists": _resolve_repo_relative(cache_root).exists() if cache_root else False,
            }
        )
    return rows


def ensure_public_pack_directories(
    registry: Dict[str, Any],
    pack: str = "",
    requested_families: Optional[Sequence[str]] = None,
    enabled_only: bool = False,
    verification_tier: str = "",
    ensure_processed: bool = False,
    ensure_cache: bool = False,
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for row in validate_public_pack_roots(
        registry,
        pack=pack,
        requested_families=requested_families,
        enabled_only=enabled_only,
        verification_tier=verification_tier,
    ):
        created_processed = False
        created_cache = False
        if ensure_processed and row["processed_root"]:
            processed_path = _resolve_repo_relative(str(row["processed_root"]))
            processed_path.mkdir(parents=True, exist_ok=True)
            created_processed = True
            row["processed_exists"] = True
        if ensure_cache and row["cache_root"]:
            cache_path = _resolve_repo_relative(str(row["cache_root"]))
            cache_path.mkdir(parents=True, exist_ok=True)
            created_cache = True
            row["cache_exists"] = True
        row["created_processed"] = created_processed
        row["created_cache"] = created_cache
        actions.append(row)
    return actions


def summarize_public_pack_selection(
    registry: Dict[str, Any],
    cells: Sequence[Any],
    root_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    def _cell_value(cell: Any, key: str) -> Any:
        if isinstance(cell, dict):
            return cell.get(key)
        return getattr(cell, key)

    unique_families = sorted({_cell_value(cell, "family") for cell in cells})
    unique_variants = sorted({_cell_value(cell, "variant") for cell in cells})
    missing_raw = [row["family"] for row in root_rows if not row.get("raw_exists")]
    return {
        "registry_name": str(registry.get("metadata", {}).get("name", "public_pack")),
        "registry_version": str(registry.get("metadata", {}).get("version", "")),
        "family_count": len(unique_families),
        "variant_count": len(unique_variants),
        "cell_count": len(cells),
        "missing_raw_families": missing_raw,
        "enabled_family_count": sum(1 for row in root_rows if row.get("enabled")),
    }