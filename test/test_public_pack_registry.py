from narrative.public_pack.registry import (
    expand_public_pack_cells,
    filter_public_pack_cells,
    load_public_pack_registry,
    summarize_public_pack_selection,
    validate_public_pack_roots,
)


def test_expand_public_pack_cells_for_ett_family():
    registry = load_public_pack_registry()
    cells = expand_public_pack_cells(registry, requested_families=["ett"])
    assert len(cells) == 16
    assert cells[0].family == "ett"
    assert cells[0].variant == "ETTh1"
    assert cells[-1].prediction_length == 720


def test_expand_public_pack_cells_respects_enabled_filter():
    registry = load_public_pack_registry()
    cells = expand_public_pack_cells(registry, enabled_only=True)
    assert cells == []


def test_validate_public_pack_roots_and_summary_shape():
    registry = load_public_pack_registry()
    rows = validate_public_pack_roots(registry, requested_families=["pems"])
    assert len(rows) == 1
    assert rows[0]["family"] == "pems"
    cells = expand_public_pack_cells(registry, requested_families=["pems"])
    summary = summarize_public_pack_selection(registry, cells, rows)
    assert summary["family_count"] == 1
    assert summary["cell_count"] == len(cells)


def test_filter_public_pack_cells_deduplicates_override_collisions():
    registry = load_public_pack_registry()
    cells = expand_public_pack_cells(registry, requested_families=["ett"])

    filtered = filter_public_pack_cells(
        cells,
        requested_variants=["ETTh1"],
        context_length=96,
        prediction_length=96,
    )

    assert len(filtered) == 1
    assert filtered[0].variant == "ETTh1"


def test_filter_public_pack_cells_keeps_distinct_protocol_cells_without_overrides():
    registry = load_public_pack_registry()
    cells = expand_public_pack_cells(registry, requested_families=["ett"])

    filtered = filter_public_pack_cells(cells, requested_variants=["ETTh1"])

    assert len(filtered) == 4