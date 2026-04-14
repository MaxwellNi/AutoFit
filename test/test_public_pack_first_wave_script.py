import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_public_pack_first_wave.py"
_SPEC = importlib.util.spec_from_file_location("run_public_pack_first_wave", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_split_supported_stage_families_respects_supported_surface():
    supported, unsupported = _MODULE._split_supported_stage_families(["ett", "solar", "m4"])

    assert supported == ["ett", "solar"]
    assert unsupported == ["m4"]


def test_partition_available_models_separates_available_and_unavailable():
    available, unavailable = _MODULE._partition_available_models(
        ["SAMformer", "MissingModel", "Prophet"],
        availability_fn=lambda name: name != "MissingModel",
    )

    assert available == ["SAMformer", "Prophet"]
    assert unavailable == ["MissingModel"]