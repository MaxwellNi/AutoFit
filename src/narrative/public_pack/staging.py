"""Helpers for staging downloadable public-pack datasets locally."""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

import pandas as pd

from .registry import REPO_ROOT


_DEFAULT_TIMEOUT_SECONDS = 60
_STAGING_USER_AGENT = "Block3PublicPackStager/1.0"
_RAW_COPY_SOURCE_FORMAT = "raw_copy"
_NUMERIC_CSV_GZIP_SOURCE_FORMAT = "numeric_csv_gzip"

SUPPORTED_PUBLIC_PACK_DOWNLOADS: Dict[str, Dict[str, Any]] = {
    "ett": {
        "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
    },
    "ecl": {
        "Electricity": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv",
    },
    "traffic": {
        "Traffic": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv",
    },
    "weather": {
        "Weather": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv",
    },
    "exchange": {
        "Exchange-rate": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/exchange_rate/exchange_rate.csv",
    },
    "ili": {
        "ILI": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/illness/national_illness.csv",
    },
    "solar": {
        "Solar": {
            "url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz",
            "source_format": "numeric_csv_gzip",
            "destination_suffix": ".csv",
        },
    },
}

SUPPORTED_PUBLIC_PACK_DOWNLOAD_FAMILIES: Tuple[str, ...] = tuple(
    sorted(SUPPORTED_PUBLIC_PACK_DOWNLOADS)
)


@dataclass(frozen=True)
class PublicPackDownload:
    family: str
    variant: str
    url: str
    destination: Path
    source_format: str = _RAW_COPY_SOURCE_FORMAT

    def to_dict(self) -> Dict[str, Any]:
        destination = str(self.destination)
        try:
            destination = str(self.destination.relative_to(REPO_ROOT))
        except ValueError:
            pass
        return {
            "family": self.family,
            "variant": self.variant,
            "url": self.url,
            "destination": destination,
            "source_format": self.source_format,
        }


def _normalize_download_spec(spec: Any) -> Dict[str, str]:
    if isinstance(spec, str):
        return {
            "url": spec,
            "source_format": _RAW_COPY_SOURCE_FORMAT,
            "destination_suffix": Path(spec).suffix or ".csv",
        }
    if isinstance(spec, dict):
        url = str(spec.get("url", "")).strip()
        if not url:
            raise ValueError(f"public-pack download spec missing url: {spec}")
        source_format = str(spec.get("source_format", _RAW_COPY_SOURCE_FORMAT)).strip()
        destination_suffix = str(spec.get("destination_suffix", Path(url).suffix or ".csv")).strip()
        return {
            "url": url,
            "source_format": source_format or _RAW_COPY_SOURCE_FORMAT,
            "destination_suffix": destination_suffix or ".csv",
        }
    raise TypeError(f"unsupported public-pack download spec: {type(spec).__name__}")


def _resolve_requested_family_names(
    registry: Dict[str, Any],
    requested_families: Optional[Sequence[str]],
) -> List[str]:
    if not requested_families:
        return list(SUPPORTED_PUBLIC_PACK_DOWNLOAD_FAMILIES)

    resolved: List[str] = []
    seen = set()
    for requested in requested_families:
        requested_norm = requested.strip().casefold()
        matched_name = None
        for family_name, family_spec in registry.get("families", {}).items():
            aliases = [str(alias).strip().casefold() for alias in family_spec.get("aliases", [])]
            candidates = [
                str(family_name).strip().casefold(),
                str(family_spec.get("display_name", family_name)).strip().casefold(),
                *aliases,
            ]
            if requested_norm in candidates:
                matched_name = str(family_name)
                break
        if matched_name is None:
            raise ValueError(f"unknown public-pack family selector: {requested}")
        if matched_name not in seen:
            seen.add(matched_name)
            resolved.append(matched_name)
    return resolved


def resolve_public_pack_downloads(
    registry: Dict[str, Any],
    requested_families: Optional[Sequence[str]] = None,
) -> List[PublicPackDownload]:
    downloads: List[PublicPackDownload] = []
    for family_name in _resolve_requested_family_names(registry, requested_families):
        family_spec = registry.get("families", {}).get(family_name)
        if family_spec is None:
            raise ValueError(f"public-pack registry missing family: {family_name}")

        download_map = SUPPORTED_PUBLIC_PACK_DOWNLOADS.get(family_name)
        if download_map is None:
            raise ValueError(
                f"public-pack staging is not implemented for family '{family_name}' yet; "
                f"supported={list(SUPPORTED_PUBLIC_PACK_DOWNLOAD_FAMILIES)}"
            )

        raw_root_text = str(family_spec.get("dataset_roots", {}).get("raw", "")).strip()
        if not raw_root_text:
            raise ValueError(f"public-pack family '{family_name}' is missing a raw dataset root")
        raw_root = (REPO_ROOT / raw_root_text).resolve()

        variants = list(family_spec.get("variants") or [])
        if not variants:
            raise ValueError(f"public-pack family '{family_name}' does not define any variants")

        for variant in variants:
            variant_name = str(variant)
            spec = download_map.get(variant_name)
            if not spec:
                raise ValueError(
                    f"public-pack staging is missing a URL for family='{family_name}', "
                    f"variant='{variant_name}'"
                )
            normalized_spec = _normalize_download_spec(spec)
            downloads.append(
                PublicPackDownload(
                    family=family_name,
                    variant=variant_name,
                    url=normalized_spec["url"],
                    destination=raw_root / f"{variant_name}{normalized_spec['destination_suffix']}",
                    source_format=normalized_spec["source_format"],
                )
            )
    return downloads


def _download_bytes(url: str, timeout: int) -> bytes:
    request = Request(url, headers={"User-Agent": _STAGING_USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        chunks: List[bytes] = []
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def _write_download(download: PublicPackDownload, timeout: int) -> None:
    download.destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = download.destination.with_suffix(download.destination.suffix + ".tmp")
    try:
        payload = _download_bytes(download.url, timeout=timeout)
        if download.source_format == _RAW_COPY_SOURCE_FORMAT:
            tmp_path.write_bytes(payload)
        elif download.source_format == _NUMERIC_CSV_GZIP_SOURCE_FORMAT:
            text = gzip.decompress(payload).decode("utf-8", errors="replace")
            frame = pd.read_csv(io.StringIO(text), header=None)
            if frame.shape[1] > 0 and frame.iloc[:, -1].isna().all():
                frame = frame.iloc[:, :-1]
            frame.columns = [f"series_{index}" for index in range(frame.shape[1])]
            frame.to_csv(tmp_path, index=False)
        else:
            raise ValueError(f"unsupported public-pack source_format: {download.source_format}")
        tmp_path.replace(download.destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def stage_public_pack_downloads(
    downloads: Sequence[PublicPackDownload],
    *,
    overwrite: bool = False,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for download in downloads:
        existed_before = download.destination.exists()
        if existed_before and not overwrite:
            status = "skipped_existing"
        else:
            _write_download(download, timeout=timeout)
            status = "overwritten" if existed_before else "downloaded"
        rows.append(
            {
                **download.to_dict(),
                "status": status,
                "size_bytes": int(download.destination.stat().st_size),
            }
        )
    return rows


__all__ = [
    "PublicPackDownload",
    "SUPPORTED_PUBLIC_PACK_DOWNLOAD_FAMILIES",
    "SUPPORTED_PUBLIC_PACK_DOWNLOADS",
    "resolve_public_pack_downloads",
    "stage_public_pack_downloads",
]