#!/usr/bin/env python3
"""Lane-private investor mark encoding for the single-model mainline scaffold."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd


_INSTITUTION_KEYWORDS: Tuple[str, ...] = (
    "venture",
    "ventures",
    "capital",
    "partners",
    "partner",
    "management",
    "equity",
    "private equity",
    "growth",
    "holdings",
    "fund",
    "funds",
    "asset",
    "investments",
    "advisor",
    "advisors",
    "vc",
    "cvc",
    "accelerator",
    "family office",
)
_RETAIL_KEYWORDS: Tuple[str, ...] = (
    "crowd",
    "crowdfunding",
    "community",
    "retail",
    "individual",
    "backers",
    "angel list",
)
_LEAD_KEYWORDS: Tuple[str, ...] = (
    "lead",
    "co-lead",
    "syndicate",
    "syndication",
    "anchor",
)
_ORG_STOPWORDS: Tuple[str, ...] = (
    "inc",
    "llc",
    "ltd",
    "lp",
    "llp",
    "plc",
    "corp",
    "corporation",
    "company",
    "co",
    "limited",
    "group",
)


@dataclass(frozen=True)
class InvestorMarkEncoderSpec:
    lane_name: str = "investors"
    max_refs_per_row: int = 8
    min_syndicate_size: int = 2
    institutional_keywords: Tuple[str, ...] = _INSTITUTION_KEYWORDS
    retail_keywords: Tuple[str, ...] = _RETAIL_KEYWORDS
    lead_keywords: Tuple[str, ...] = _LEAD_KEYWORDS

    def as_dict(self) -> Dict[str, object]:
        return {
            "lane_name": self.lane_name,
            "max_refs_per_row": self.max_refs_per_row,
            "min_syndicate_size": self.min_syndicate_size,
            "institutional_keywords": self.institutional_keywords,
            "retail_keywords": self.retail_keywords,
            "lead_keywords": self.lead_keywords,
        }


class InvestorMarkEncoder:
    def __init__(self, spec: InvestorMarkEncoderSpec | None = None):
        self.spec = spec or InvestorMarkEncoderSpec()

    def build_mark_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(index=frame.index)

        parsed_lists = [self._parse_investor_list(value) for value in frame.get("investors__json", pd.Series(index=frame.index))]
        investor_len = self._resolve_investor_lengths(frame, parsed_lists)
        website_series = frame.get("investor_website", pd.Series("", index=frame.index, dtype="object")).fillna("")
        investment_type = frame.get("investment_type", pd.Series("", index=frame.index, dtype="object")).fillna("")
        website_domains = [self._extract_domain(value) for value in website_series]
        combined_text = [
            " ".join(list_refs + [str(website_series.iloc[idx]), str(investment_type.iloc[idx])]).strip().lower()
            for idx, list_refs in enumerate(parsed_lists)
        ]

        mark = pd.DataFrame(index=frame.index)
        mark["mark_list_present"] = (investor_len > 0).astype(np.float32)
        mark["mark_list_length_log"] = np.log1p(investor_len).astype(np.float32)
        mark["mark_website_present"] = website_series.astype(str).str.strip().ne("").astype(np.float32)
        mark["mark_hash_present"] = frame.get("investors__hash", pd.Series(index=frame.index)).notna().astype(np.float32)

        institutional_score = np.asarray(
            [self._keyword_score(text, self.spec.institutional_keywords) for text in combined_text],
            dtype=np.float32,
        )
        retail_score = np.asarray(
            [self._keyword_score(text, self.spec.retail_keywords) for text in combined_text],
            dtype=np.float32,
        )
        lead_score = np.asarray(
            [self._keyword_score(text, self.spec.lead_keywords) for text in combined_text],
            dtype=np.float32,
        )
        domain_institutional_score = np.asarray(
            [self._keyword_score(domain, self.spec.institutional_keywords) for domain in website_domains],
            dtype=np.float32,
        )
        mark["mark_institutional_keyword_score"] = np.maximum(institutional_score, domain_institutional_score)
        mark["mark_retail_keyword_score"] = retail_score
        mark["mark_lead_keyword_score"] = lead_score

        non_national = _numeric_series(frame, "non_national_investors")
        non_accredited = self._first_available_numeric(
            frame,
            [
                "last_number_non_accredited_investors",
                "mean_number_non_accredited_investors",
                "ema_number_non_accredited_investors",
            ],
        )
        already_invested = self._first_available_numeric(
            frame,
            [
                "last_total_number_already_invested",
                "mean_total_number_already_invested",
                "ema_total_number_already_invested",
            ],
        )
        minimum_investment = self._first_available_numeric(
            frame,
            [
                "last_minimum_investment_accepted",
                "mean_minimum_investment_accepted",
                "ema_minimum_investment_accepted",
            ],
        )
        total_offering = self._first_available_numeric(
            frame,
            [
                "last_total_offering_amount",
                "mean_total_offering_amount",
                "ema_total_offering_amount",
            ],
        )
        total_sold = self._first_available_numeric(
            frame,
            [
                "last_total_amount_sold",
                "mean_total_amount_sold",
                "ema_total_amount_sold",
            ],
        )
        total_remaining = self._first_available_numeric(
            frame,
            [
                "last_total_remaining",
                "mean_total_remaining",
                "ema_total_remaining",
            ],
        )

        denom_investor = np.maximum.reduce(
            [
                investor_len.to_numpy(dtype=np.float64, copy=False),
                already_invested.to_numpy(dtype=np.float64, copy=False),
                np.ones(len(frame), dtype=np.float64),
            ]
        )
        non_national_share = np.clip(non_national.to_numpy(dtype=np.float64, copy=False) / denom_investor, 0.0, 1.0)
        non_accredited_share = np.clip(non_accredited.to_numpy(dtype=np.float64, copy=False) / denom_investor, 0.0, 1.0)
        offering_progress = np.clip(
            total_sold.to_numpy(dtype=np.float64, copy=False) / np.maximum(total_offering.to_numpy(dtype=np.float64, copy=False), 1.0),
            0.0,
            3.0,
        )
        remaining_ratio = np.clip(
            total_remaining.to_numpy(dtype=np.float64, copy=False) / np.maximum(total_offering.to_numpy(dtype=np.float64, copy=False), 1.0),
            0.0,
            3.0,
        )
        sold_per_investor = total_sold.to_numpy(dtype=np.float64, copy=False) / denom_investor
        min_investment_scaled = np.tanh(np.log1p(np.clip(minimum_investment.to_numpy(dtype=np.float64, copy=False), 0.0, None)) / 6.0)
        sold_per_investor_scaled = np.tanh(np.log1p(np.clip(sold_per_investor, 0.0, None)) / 8.0)
        concentrated_capital = np.clip(0.55 * sold_per_investor_scaled + 0.45 * min_investment_scaled, 0.0, 1.0)

        mark["mark_non_national_share"] = non_national_share.astype(np.float32)
        mark["mark_non_accredited_share"] = non_accredited_share.astype(np.float32)
        mark["mark_minimum_investment_log"] = np.log1p(np.clip(minimum_investment, 0.0, None)).astype(np.float32)
        mark["mark_offering_progress"] = offering_progress.astype(np.float32)
        mark["mark_remaining_ratio"] = remaining_ratio.astype(np.float32)
        mark["mark_sold_per_investor_log"] = np.log1p(np.clip(sold_per_investor, 0.0, None)).astype(np.float32)
        mark["mark_concentrated_capital_score"] = concentrated_capital.astype(np.float32)
        mark["mark_syndicate_size_score"] = np.clip(investor_len.to_numpy(dtype=np.float64, copy=False) / max(self.spec.min_syndicate_size + 2, 1), 0.0, 1.0).astype(np.float32)

        institutional_like = np.clip(
            0.25 * mark["mark_institutional_keyword_score"].to_numpy(dtype=np.float64, copy=False)
            + 0.20 * concentrated_capital
            + 0.20 * (1.0 - non_accredited_share)
            + 0.15 * np.clip(offering_progress, 0.0, 1.0)
            + 0.10 * non_national_share
            + 0.10 * mark["mark_website_present"].to_numpy(dtype=np.float64, copy=False),
            0.0,
            1.0,
        )
        retail_like = np.clip(
            0.35 * non_accredited_share
            + 0.25 * mark["mark_retail_keyword_score"].to_numpy(dtype=np.float64, copy=False)
            + 0.20 * (1.0 - min_investment_scaled)
            + 0.20 * (1.0 - sold_per_investor_scaled),
            0.0,
            1.0,
        )
        large_investor_event = np.clip(
            0.40 * concentrated_capital
            + 0.25 * institutional_like
            + 0.20 * np.clip(1.0 / np.maximum(investor_len.to_numpy(dtype=np.float64, copy=False), 1.0), 0.0, 1.0)
            + 0.15 * lead_score,
            0.0,
            1.0,
        )
        mark["mark_institutional_like_score"] = institutional_like.astype(np.float32)
        mark["mark_retail_like_score"] = retail_like.astype(np.float32)
        mark["mark_large_investor_event_score"] = large_investor_event.astype(np.float32)

        current_hash = self._resolve_current_hash(frame, parsed_lists)
        entity_ids = frame.get("entity_id", pd.Series(index=frame.index, dtype="object")).astype(str)
        prev_hash = current_hash.groupby(entity_ids, sort=False).shift(1)
        mark["mark_repeat_list_flag"] = (current_hash.notna() & prev_hash.notna() & (current_hash == prev_hash)).astype(np.float32)
        mark["mark_list_changed_flag"] = (current_hash.notna() & prev_hash.notna() & (current_hash != prev_hash)).astype(np.float32)
        return mark.astype(np.float32)

    def build_entity_resolution_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        investor_json = frame.get("investors__json", pd.Series(index=frame.index, dtype="object"))
        investor_website = frame.get("investor_website", pd.Series(index=frame.index, dtype="object"))
        for idx, row in enumerate(frame.itertuples(index=False)):
            entity_id = getattr(row, "entity_id", None)
            crawled_date_day = getattr(row, "crawled_date_day", None)
            refs = self._parse_investor_list(investor_json.iloc[idx])[: self.spec.max_refs_per_row]
            website = str(investor_website.iloc[idx]).strip() if pd.notna(investor_website.iloc[idx]) else ""
            for raw_ref in refs:
                rows.append(
                    {
                        "entity_id": entity_id,
                        "crawled_date_day": crawled_date_day,
                        "raw_reference": raw_ref,
                        "canonical_reference": canonicalize_investor_reference(raw_ref),
                        "domain": self._extract_domain(raw_ref) or self._extract_domain(website),
                        "source_kind": "investors__json",
                        "institutional_keyword_score": self._keyword_score(raw_ref, self.spec.institutional_keywords),
                    }
                )
            if website:
                rows.append(
                    {
                        "entity_id": entity_id,
                        "crawled_date_day": crawled_date_day,
                        "raw_reference": website,
                        "canonical_reference": canonicalize_investor_reference(website),
                        "domain": self._extract_domain(website),
                        "source_kind": "investor_website",
                        "institutional_keyword_score": self._keyword_score(website, self.spec.institutional_keywords),
                    }
                )
        return pd.DataFrame(rows)

    def _parse_investor_list(self, value: object) -> list[str]:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return []
        parsed = _try_parse_structured_list(text)
        if parsed:
            return [item for item in parsed if item][: self.spec.max_refs_per_row]
        if any(sep in text for sep in (";", "|", "\n", ",")):
            refs = [part.strip() for part in re.split(r"[;|\n,]+", text) if part.strip()]
            return refs[: self.spec.max_refs_per_row]
        return [text]

    def _resolve_investor_lengths(self, frame: pd.DataFrame, parsed_lists: Sequence[Sequence[str]]) -> pd.Series:
        raw_len = _numeric_series(frame, "investors__len")
        inferred = pd.Series([float(len(items)) for items in parsed_lists], index=frame.index, dtype=np.float64)
        return raw_len.where(raw_len > 0.0, inferred).fillna(0.0)

    def _resolve_current_hash(self, frame: pd.DataFrame, parsed_lists: Sequence[Sequence[str]]) -> pd.Series:
        raw_hash = frame.get("investors__hash", pd.Series(index=frame.index, dtype="object"))
        resolved = []
        for idx, value in enumerate(raw_hash):
            if pd.notna(value) and str(value).strip():
                resolved.append(str(value).strip())
                continue
            canonical = [canonicalize_investor_reference(item) for item in parsed_lists[idx] if item]
            resolved.append("|".join(sorted(set(canonical))) if canonical else np.nan)
        return pd.Series(resolved, index=frame.index, dtype="object")

    def _extract_domain(self, value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        if not text:
            return ""
        candidate = text if "://" in text else f"https://{text}"
        try:
            parsed = urlparse(candidate)
        except Exception:
            return ""
        host = parsed.netloc or parsed.path
        host = host.lower().strip()
        if host.startswith("www."):
            host = host[4:]
        if not host or "." not in host:
            return ""
        parts = [part for part in host.split(".") if part]
        return ".".join(parts[-2:]) if len(parts) >= 2 else host

    def _keyword_score(self, text: object, keywords: Iterable[str]) -> float:
        content = str(text).lower()
        if not content:
            return 0.0
        keyword_list = tuple(keywords)
        hits = sum(1 for keyword in keyword_list if keyword in content)
        return float(np.clip(hits / max(len(keyword_list), 1), 0.0, 1.0))

    def _first_available_numeric(self, frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
        series = pd.Series(0.0, index=frame.index, dtype=np.float64)
        for column in columns:
            if column in frame.columns:
                candidate = _numeric_series(frame, column)
                series = series.where(series > 0.0, candidate)
        return series.fillna(0.0)


def canonicalize_investor_reference(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    candidate = text if "://" in text else f"https://{text}"
    try:
        parsed = urlparse(candidate)
    except Exception:
        parsed = None
    host = ""
    if parsed is not None:
        host = (parsed.netloc or "").lower().strip()
        if host.startswith("www."):
            host = host[4:]
    if host:
        parts = [part for part in host.split(".") if part]
        if len(parts) >= 2:
            return ".".join(parts[-2:])
    normalized = re.sub(r"[^a-z0-9\s]+", " ", text)
    tokens = [token for token in normalized.split() if token and token not in _ORG_STOPWORDS]
    return " ".join(tokens[:4])


def _try_parse_structured_list(text: str) -> list[str]:
    for parser in (json.loads, ast.literal_eval):
        try:
            payload = parser(text)
        except Exception:
            continue
        refs = _flatten_investor_payload(payload)
        if refs:
            return refs
    return []


def _flatten_investor_payload(payload: object) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, str):
        text = payload.strip()
        return [text] if text else []
    if isinstance(payload, dict):
        refs: list[str] = []
        for key in ("name", "investor", "website", "url"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                refs.append(value.strip())
        if refs:
            return refs
        flattened: list[str] = []
        for value in payload.values():
            flattened.extend(_flatten_investor_payload(value))
        return flattened
    if isinstance(payload, (list, tuple, set)):
        flattened: list[str] = []
        for item in payload:
            flattened.extend(_flatten_investor_payload(item))
        return flattened
    text = str(payload).strip()
    return [text] if text else []


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)


__all__ = ["InvestorMarkEncoder", "InvestorMarkEncoderSpec", "canonicalize_investor_reference"]