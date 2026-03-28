#!/usr/bin/env python3
"""Helpers for optional runtime dependencies used by Block 3 models."""
from __future__ import annotations

import ctypes
import logging
import os
import site
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def get_optional_vendor_dir() -> Path:
    """Return the default user-local site-packages path for optional deps."""
    env_override = os.getenv("BLOCK3_OPTIONAL_VENDOR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    return Path.home() / ".cache" / "block3_optional_pydeps" / py_tag


def ensure_optional_vendor_on_path() -> Path:
    """Add the optional vendor directory to sys.path when it exists."""
    vendor = get_optional_vendor_dir()
    if vendor.exists():
        site.addsitedir(str(vendor))
        if str(vendor) in sys.path:
            sys.path.remove(str(vendor))
        # Append to the end so core env packages (numpy/pandas/torch/etc.)
        # continue to come from the canonical insider environment.
        sys.path.append(str(vendor))
    return vendor


def _candidate_lib_dirs() -> list[Path]:
    candidates: list[Path] = []
    exe = Path(sys.executable).resolve()
    candidates.append(exe.parent.parent / "lib")
    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib")
    return [p for p in candidates if p.exists()]


def ensure_insider_libstdcpp() -> Path | None:
    """Preload a modern libstdc++ from the active micromamba env if present."""
    for lib_dir in _candidate_lib_dirs():
        for name in ("libstdc++.so.6.0.34", "libstdc++.so.6"):
            lib_path = lib_dir / name
            if not lib_path.exists():
                continue
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                current = os.environ.get("LD_LIBRARY_PATH", "")
                parts = [p for p in current.split(":") if p]
                if str(lib_dir) not in parts:
                    os.environ["LD_LIBRARY_PATH"] = (
                        f"{lib_dir}:{current}" if current else str(lib_dir)
                    )
                return lib_path
            except OSError as e:
                logger.debug("Failed to preload %s: %s", lib_path, e)
                continue
    return None


__all__ = [
    "ensure_insider_libstdcpp",
    "ensure_optional_vendor_on_path",
    "get_optional_vendor_dir",
]
