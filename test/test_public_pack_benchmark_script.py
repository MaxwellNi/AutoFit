from narrative.public_pack.registry import (
    expand_public_pack_cells,
    filter_public_pack_cells,
    load_public_pack_registry,
)


def test_filter_cells_deduplicates_effective_override_cells():
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


def test_filter_cells_keeps_distinct_cells_without_overrides():
    registry = load_public_pack_registry()
    cells = expand_public_pack_cells(registry, requested_families=["ett"])

    filtered = filter_public_pack_cells(cells, requested_variants=["ETTh1"])

    assert len(filtered) == 4