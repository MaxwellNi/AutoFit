from narrative.official_sources import (
    build_official_source_rows,
    load_official_source_registry,
    summarize_official_source_registry,
    validate_official_source_registry,
)


def test_official_source_registry_is_valid():
    registry = load_official_source_registry()
    assert validate_official_source_registry(registry) == []


def test_official_source_registry_summary_counts_pending_and_verified():
    registry = load_official_source_registry()
    summary = summarize_official_source_registry(registry)
    assert summary["entry_count"] == 4
    assert summary["verified_count"] == 2
    assert summary["pending_recheck_count"] == 2


def test_official_source_registry_rows_include_elastst_urls():
    registry = load_official_source_registry()
    rows = build_official_source_rows(registry)
    elastst = next(row for row in rows if row["signal"] == "ElasTST")
    assert "ProbTS" in elastst["official_repo"]
    assert elastst["official_pdf"].startswith("https://openreview.net/")