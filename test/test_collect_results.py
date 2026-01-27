from pathlib import Path
import importlib.util
import pandas as pd


def _load_collect_results():
    spec = importlib.util.spec_from_file_location(
        "collect_results", Path(__file__).parent.parent / "scripts" / "collect_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.collect_results

def _load_collect_results():
    spec = importlib.util.spec_from_file_location(
        "collect_results", Path(__file__).parent.parent / "scripts" / "collect_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.collect_results


def test_collect_results(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    run = runs_dir / "exp1"
    (run / "paper_tables").mkdir(parents=True)
    (run / "explain").mkdir(parents=True)
    (run / "best_config.yaml").write_text("test: 1")
    pd.DataFrame({"a": [1]}).to_parquet(run / "paper_tables" / "main_results_table.parquet", index=False)
    (run / "explain" / "faithfulness.json").write_text("{}")

    out_dir = tmp_path / "collected"
    collect_results = _load_collect_results()
    collect_results = _load_collect_results()
    collect_results(runs_dir, out_dir)
    assert (out_dir / "paper_tables").exists()
    assert (out_dir / "paper_tables" / "exp1_main_results_table.parquet").exists()
    assert (out_dir / "explain" / "exp1_faithfulness.json").exists()
    assert (out_dir / "configs" / "exp1_best_config.yaml").exists()
