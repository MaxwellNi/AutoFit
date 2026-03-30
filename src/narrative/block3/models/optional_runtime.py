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


def get_tabpfn_vendor_dir() -> Path:
    """Return the preferred override path for a latest-source TabPFN install."""
    env_override = os.getenv("BLOCK3_TABPFN_VENDOR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    return Path.home() / ".cache" / "block3_optional_pydeps" / f"{py_tag}_tabpfn_latest"


def get_optional_repo_root() -> Path:
    """Return the default user-local root for optional vendor repos."""
    env_override = os.getenv("BLOCK3_OPTIONAL_REPO_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return Path.home() / ".cache" / "block3_optional_repos"


def get_lightgts_repo_dir() -> Path:
    """Return the preferred LightGTS repo path.

    Resolution order:
    1. explicit `BLOCK3_LIGHTGTS_REPO`
    2. user-local optional repo root
    3. `/tmp/LightGTS` as a practical same-node audit fallback
    """
    env_override = os.getenv("BLOCK3_LIGHTGTS_REPO")
    if env_override:
        return Path(env_override).expanduser().resolve()
    default_repo = get_optional_repo_root() / "LightGTS"
    if default_repo.exists():
        return default_repo.resolve()
    tmp_repo = Path("/tmp/LightGTS")
    if tmp_repo.exists():
        return tmp_repo.resolve()
    return default_repo.resolve()


def get_olinear_repo_dir() -> Path:
    """Return the preferred OLinear repo path."""
    env_override = os.getenv("BLOCK3_OLINEAR_REPO")
    if env_override:
        return Path(env_override).expanduser().resolve()
    default_repo = get_optional_repo_root() / "OLinear"
    return default_repo.resolve()


def get_probts_repo_dir() -> Path:
    """Return the preferred ProbTS repo path."""
    env_override = os.getenv("BLOCK3_PROBTS_REPO")
    if env_override:
        return Path(env_override).expanduser().resolve()
    default_repo = get_optional_repo_root() / "ProbTS"
    return default_repo.resolve()


def get_units_repo_dir() -> Path:
    """Return the preferred UniTS repo path."""
    env_override = os.getenv("BLOCK3_UNITS_REPO")
    if env_override:
        return Path(env_override).expanduser().resolve()
    default_repo = get_optional_repo_root() / "UniTS"
    return default_repo.resolve()


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


def ensure_tabpfn_vendor_on_path() -> Path:
    """Prepend a latest-source TabPFN vendor dir if it exists.

    This is intentionally more aggressive than ``ensure_optional_vendor_on_path``:
    the current shared insider environment may carry an older ``tabpfn`` wheel,
    while we sometimes need to override only that package with a newer official
    GitHub install without mutating the shared env.
    """
    vendor = get_tabpfn_vendor_dir()
    if vendor.exists():
        site.addsitedir(str(vendor))
        if str(vendor) in sys.path:
            sys.path.remove(str(vendor))
        sys.path.insert(0, str(vendor))
    return vendor


def ensure_lightgts_repo_on_path() -> Path:
    """Expose an audited LightGTS repo on `sys.path` when available.

    The official repo is script-driven rather than a pip package. For Block 3 we
    therefore prefer a vendor-repo import pattern similar to a shallow vendored
    source tree. This helper adds both the repo root and its `src/` subtree so
    later wrappers can import either top-level scripts or model modules.
    """
    repo = get_lightgts_repo_dir()
    if not repo.exists():
        return repo
    for path in (repo, repo / "src"):
        s = str(path)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
    return repo


def ensure_olinear_repo_on_path() -> Path:
    """Expose an audited OLinear repo on `sys.path` when available."""
    repo = get_olinear_repo_dir()
    if not repo.exists():
        return repo
    s = str(repo)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)
    return repo


def ensure_probts_repo_on_path() -> Path:
    """Expose an audited ProbTS repo on `sys.path` when available."""
    repo = get_probts_repo_dir()
    if not repo.exists():
        return repo
    for path in (repo, repo / "probts"):
        s = str(path)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
    return repo


def ensure_units_repo_on_path() -> Path:
    """Expose an audited UniTS repo on `sys.path` when available."""
    repo = get_units_repo_dir()
    if not repo.exists():
        return repo
    s = str(repo)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)
    return repo


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
    "ensure_lightgts_repo_on_path",
    "ensure_olinear_repo_on_path",
    "ensure_probts_repo_on_path",
    "ensure_optional_vendor_on_path",
    "ensure_tabpfn_vendor_on_path",
    "ensure_units_repo_on_path",
    "get_lightgts_repo_dir",
    "get_olinear_repo_dir",
    "get_optional_vendor_dir",
    "get_optional_repo_root",
    "get_probts_repo_dir",
    "get_tabpfn_vendor_dir",
    "get_units_repo_dir",
]
