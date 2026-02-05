#!/usr/bin/env python3
"""
Block 3 Unified Dataset Interface.

Provides read-only access to frozen WIDE2 artifacts (stamp=20260203_225620).
All join rules are explicit and auditable.

Join Keys:
- entity_id + crawled_date_day: offers_core <-> offers_text
- cik + crawled_date_day: offers_core <-> edgar_store
- entity_id + crawled_date_day: offers_core <-> multiscale

This module is designed for Block 3 modeling and must never modify freeze artifacts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq
import yaml

try:
    from deltalake import DeltaTable
    HAS_DELTA = True
except ImportError:
    HAS_DELTA = False


@dataclass
class FreezePointer:
    """Parsed FULL_SCALE_POINTER.yaml."""
    stamp: str
    variant: str
    offers_core_daily_dir: Path
    offers_core_snapshot_dir: Path
    offers_text_dir: Path
    edgar_store_dir: Path
    multiscale_dir: Path
    snapshots_offer_day: Path
    snapshots_cik_day: Path
    analysis_dir: Path
    
    @classmethod
    def load(cls, path: Path = Path("docs/audits/FULL_SCALE_POINTER.yaml")) -> "FreezePointer":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            stamp=data["stamp"],
            variant=data["variant"],
            offers_core_daily_dir=Path(data["offers_core_daily"]["dir"]),
            offers_core_snapshot_dir=Path(data["offers_core_snapshot"]["dir"]),
            offers_text_dir=Path(data["offers_text"]["dir"]),
            edgar_store_dir=Path(data["edgar_store_full_daily"]["dir"]),
            multiscale_dir=Path(data["multiscale_full"]["dir"]),
            snapshots_offer_day=Path(data["snapshots_index"]["offer_day"]),
            snapshots_cik_day=Path(data["snapshots_index"]["cik_day"]),
            analysis_dir=Path(data["analysis"]["dir"]),
        )


@dataclass
class Block3Dataset:
    """
    Unified dataset interface for Block 3 modeling.
    
    Provides lazy loading and explicit join methods for all freeze artifacts.
    """
    pointer: FreezePointer
    _offers_core_daily: Optional[pd.DataFrame] = field(default=None, repr=False)
    _offers_text: Optional[pd.DataFrame] = field(default=None, repr=False)
    _edgar_store: Optional[pd.DataFrame] = field(default=None, repr=False)
    _snapshots_offer_day: Optional[pd.DataFrame] = field(default=None, repr=False)
    _snapshots_cik_day: Optional[pd.DataFrame] = field(default=None, repr=False)
    
    @classmethod
    def from_pointer(cls, pointer_path: Path = Path("docs/audits/FULL_SCALE_POINTER.yaml")) -> "Block3Dataset":
        pointer = FreezePointer.load(pointer_path)
        return cls(pointer=pointer)
    
    # -------------------------------------------------------------------------
    # Lazy loaders
    # -------------------------------------------------------------------------
    
    def get_offers_core_daily(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load offers_core_daily parquet."""
        if self._offers_core_daily is None:
            path = self.pointer.offers_core_daily_dir
            # Try direct parquet file first
            pq_path = path / "offers_core_daily.parquet"
            if pq_path.exists():
                self._offers_core_daily = pd.read_parquet(pq_path, columns=columns)
            elif HAS_DELTA:
                # Fallback to delta table
                try:
                    dt = DeltaTable(str(path))
                    self._offers_core_daily = dt.to_pandas(columns=columns)
                except Exception:
                    # Last resort: read as parquet directory
                    self._offers_core_daily = pd.read_parquet(path, columns=columns)
            else:
                # Read as parquet directory
                self._offers_core_daily = pd.read_parquet(path, columns=columns)
        return self._offers_core_daily
    
    def get_offers_text(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load offers_text parquet."""
        if self._offers_text is None:
            pq_path = self.pointer.offers_text_dir / "offers_text.parquet"
            self._offers_text = pd.read_parquet(pq_path, columns=columns)
        return self._offers_text
    
    def get_edgar_store(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load edgar_store_full_daily."""
        if self._edgar_store is None:
            path = self.pointer.edgar_store_dir / "edgar_features"
            if path.exists():
                self._edgar_store = pd.read_parquet(path, columns=columns)
            else:
                # Try delta table
                if HAS_DELTA:
                    dt = DeltaTable(str(self.pointer.edgar_store_dir))
                    self._edgar_store = dt.to_pandas(columns=columns)
                else:
                    self._edgar_store = pd.read_parquet(self.pointer.edgar_store_dir, columns=columns)
        return self._edgar_store
    
    def get_snapshots_offer_day(self) -> pd.DataFrame:
        """Load snapshots_offer_day index."""
        if self._snapshots_offer_day is None:
            self._snapshots_offer_day = pd.read_parquet(self.pointer.snapshots_offer_day)
        return self._snapshots_offer_day
    
    def get_snapshots_cik_day(self) -> pd.DataFrame:
        """Load snapshots_cik_day index."""
        if self._snapshots_cik_day is None:
            self._snapshots_cik_day = pd.read_parquet(self.pointer.snapshots_cik_day)
        return self._snapshots_cik_day
    
    # -------------------------------------------------------------------------
    # Join methods (explicit keys for auditability)
    # -------------------------------------------------------------------------
    
    def join_core_with_text(
        self,
        core_df: Optional[pd.DataFrame] = None,
        text_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Join offers_core_daily with offers_text.
        
        Join keys: entity_id, crawled_date_day (or snapshot_ts for text)
        """
        if core_df is None:
            core_df = self.get_offers_core_daily()
        
        text_df = self.get_offers_text(columns=text_columns)
        
        # Normalize key columns
        core_df = core_df.copy()
        text_df = text_df.copy()
        
        # Text uses snapshot_ts, need to extract date
        if "crawled_date_day" not in text_df.columns and "snapshot_ts" in text_df.columns:
            text_df["crawled_date_day"] = pd.to_datetime(text_df["snapshot_ts"]).dt.date.astype(str)
        
        # Merge
        merged = core_df.merge(
            text_df,
            on=["entity_id", "crawled_date_day"],
            how="left",
            suffixes=("", "_text"),
        )
        
        return merged
    
    def join_core_with_edgar(
        self,
        core_df: Optional[pd.DataFrame] = None,
        edgar_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Join offers_core_daily with edgar_store.
        
        Join keys: cik, crawled_date_day
        Note: Only rows with non-null cik will have edgar features.
        """
        if core_df is None:
            core_df = self.get_offers_core_daily()
        
        edgar_df = self.get_edgar_store(columns=edgar_columns)
        
        # Filter core to rows with cik
        core_with_cik = core_df[core_df["cik"].notna()].copy()
        
        # Merge
        merged = core_with_cik.merge(
            edgar_df,
            on=["cik", "crawled_date_day"],
            how="left",
            suffixes=("", "_edgar"),
        )
        
        return merged
    
    def get_panel_dataset(
        self,
        target_cols: List[str] = ["funding_raised_usd", "investors_count"],
        include_text: bool = False,
        include_edgar: bool = False,
        include_multiscale: bool = False,
    ) -> pd.DataFrame:
        """
        Build a panel dataset for modeling.
        
        Returns entity-day panel with requested features.
        """
        # Start with core
        df = self.get_offers_core_daily()
        
        # Add text features
        if include_text:
            df = self.join_core_with_text(df)
        
        # Add edgar features
        if include_edgar:
            # Join where cik available
            edgar_cols = [c for c in self.get_edgar_store().columns if c not in ["cik", "crawled_date_day"]]
            edgar_df = self.get_edgar_store()
            df = df.merge(
                edgar_df,
                on=["cik", "crawled_date_day"],
                how="left",
                suffixes=("", "_edgar"),
            )
        
        # TODO: Add multiscale features
        if include_multiscale:
            pass  # Will implement after checking multiscale structure
        
        return df
    
    # -------------------------------------------------------------------------
    # Consistency checks
    # -------------------------------------------------------------------------
    
    def run_consistency_checks(self, n_sample: int = 1000) -> Dict[str, Any]:
        """
        Run consistency checks on a sample of data.
        
        Checks:
        1. Key uniqueness in each table
        2. Duplicate keys
        3. Null keys
        4. Time alignment
        """
        results = {}
        
        # Sample from core
        core = self.get_offers_core_daily()
        sample_idx = core.sample(min(n_sample, len(core))).index
        core_sample = core.loc[sample_idx]
        
        # Check 1: Key uniqueness in core (entity_id, crawled_date_day)
        core_keys = core_sample[["entity_id", "crawled_date_day"]]
        core_key_counts = core_keys.groupby(["entity_id", "crawled_date_day"]).size()
        results["core_key_duplicates"] = int((core_key_counts > 1).sum())
        results["core_key_nulls"] = int(core_keys.isnull().any(axis=1).sum())
        
        # Check 2: Text join alignment
        text = self.get_offers_text()
        text_entities = set(text["entity_id"].unique())
        core_entities = set(core_sample["entity_id"].unique())
        results["text_entity_coverage"] = len(core_entities & text_entities) / max(len(core_entities), 1)
        
        # Check 3: EDGAR join alignment
        edgar = self.get_edgar_store()
        core_ciks = set(core_sample[core_sample["cik"].notna()]["cik"].unique())
        edgar_ciks = set(edgar["cik"].unique())
        results["edgar_cik_coverage"] = len(core_ciks & edgar_ciks) / max(len(core_ciks), 1)
        
        # Check 4: Time range alignment
        core_dates = pd.to_datetime(core["crawled_date_day"])
        results["core_date_range"] = {
            "min": str(core_dates.min()),
            "max": str(core_dates.max()),
            "n_unique_days": int(core_dates.nunique()),
        }
        
        return results


def load_dataset() -> Block3Dataset:
    """Convenience function to load the Block 3 dataset."""
    return Block3Dataset.from_pointer()


if __name__ == "__main__":
    # Quick test
    ds = load_dataset()
    print(f"Loaded dataset for stamp: {ds.pointer.stamp}")
    print(f"Variant: {ds.pointer.variant}")
    
    # Run consistency checks
    print("\nRunning consistency checks...")
    checks = ds.run_consistency_checks(n_sample=1000)
    for k, v in checks.items():
        print(f"  {k}: {v}")
