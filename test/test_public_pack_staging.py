import gzip

from narrative.public_pack import staging as staging_module
from narrative.public_pack.registry import load_public_pack_registry
from narrative.public_pack.staging import (
    PublicPackDownload,
    resolve_public_pack_downloads,
    stage_public_pack_downloads,
)


def test_resolve_public_pack_downloads_for_ett_family():
    registry = load_public_pack_registry()

    downloads = resolve_public_pack_downloads(registry, requested_families=["ett"])

    assert len(downloads) == 4
    assert downloads[0].family == "ett"
    assert downloads[0].variant == "ETTh1"
    assert downloads[0].destination.name == "ETTh1.csv"
    assert downloads[0].url.endswith("/ETTh1.csv")


def test_resolve_public_pack_downloads_for_solar_family_uses_gzip_conversion():
    registry = load_public_pack_registry()

    downloads = resolve_public_pack_downloads(registry, requested_families=["solar"])

    assert len(downloads) == 1
    assert downloads[0].variant == "Solar"
    assert downloads[0].destination.name == "Solar.csv"
    assert downloads[0].source_format == "numeric_csv_gzip"


def test_resolve_public_pack_downloads_rejects_unsupported_family():
    registry = load_public_pack_registry()

    try:
        resolve_public_pack_downloads(registry, requested_families=["m4"])
    except ValueError as exc:
        assert "not implemented" in str(exc)
    else:
        raise AssertionError("expected unsupported public-pack family to raise ValueError")


def test_stage_public_pack_downloads_skips_existing_file(tmp_path):
    destination = tmp_path / "ETTh1.csv"
    destination.write_text("date,target\n2024-01-01,1\n", encoding="utf-8")
    download = PublicPackDownload(
        family="ett",
        variant="ETTh1",
        url="https://example.com/ETTh1.csv",
        destination=destination,
    )

    rows = stage_public_pack_downloads([download], overwrite=False)

    assert rows[0]["status"] == "skipped_existing"
    assert rows[0]["size_bytes"] == destination.stat().st_size


def test_stage_public_pack_downloads_converts_numeric_gzip_to_csv(tmp_path, monkeypatch):
    payload = gzip.compress(b"1,2,\n3,4,\n")
    monkeypatch.setattr(staging_module, "_download_bytes", lambda url, timeout: payload)
    destination = tmp_path / "Solar.csv"
    download = PublicPackDownload(
        family="solar",
        variant="Solar",
        url="https://example.com/solar_AL.txt.gz",
        destination=destination,
        source_format="numeric_csv_gzip",
    )

    rows = stage_public_pack_downloads([download], overwrite=True)

    assert rows[0]["status"] == "downloaded"
    assert destination.read_text(encoding="utf-8").splitlines()[0] == "series_0,series_1"