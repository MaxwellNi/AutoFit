import pandas as pd

from narrative.explainability.exporter import export_explainability


def test_explainability_outputs_faithfulness(tmp_path):
    df = pd.DataFrame(
        {
            "platform_name": ["p"],
            "offer_id": ["1"],
            "tone_optimism": [0.1],
            "edgar_feat": [1.0],
            "open_year": [2023],
            "target": [1.0],
        }
    )
    out = export_explainability(tmp_path, df=df, target_col="target")
    assert (out / "shap_global.json").exists()
    assert (out / "tcav_concepts.json").exists()
    assert (out / "ig_highlights.json").exists()
    assert (out / "lime_cases.json").exists()
    assert (out / "llm_report.md").exists()
    assert (out / "faithfulness.json").exists()
    assert (out / "attributions.json").exists()
