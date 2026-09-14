from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyse_divya_results",
    REPO_ROOT / "scripts" / "analyse_divya_results.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DivyaAnalysisTests(unittest.TestCase):
    def test_string_false_is_not_converted_to_true(self):
        result = MODULE._as_boolean(pd.Series(["True", "False", "0", "1"]))
        self.assertEqual(result.tolist(), [True, False, False, True])

    def test_only_divya_models_are_loaded(self):
        frame = pd.DataFrame([
            {
                "image_id": "1",
                "image_name": "1.jpg",
                "model": "clip",
                "task": "classification",
                "study_label": "roof",
                "income_quartile": "Q1",
                "metric_value": False,
                "raw_output": "stove",
            },
            {
                "image_id": "1",
                "image_name": "1.jpg",
                "model": "internvl",
                "task": "classification",
                "study_label": "roof",
                "income_quartile": "Q1",
                "metric_value": True,
                "raw_output": "roof",
            },
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            frame.to_csv(path, index=False)
            loaded = MODULE._load_predictions(path)
        self.assertEqual(loaded["model"].tolist(), ["clip"])
        self.assertEqual(loaded["metric_value"].tolist(), [False])

    def test_clip_confusion_table_preserves_true_and_predicted_categories(self):
        frame = pd.DataFrame([
            {
                "image_id": "1",
                "model": "clip",
                "task": "classification",
                "study_label": "roof",
                "predicted_label": "roof",
            },
            {
                "image_id": "2",
                "model": "clip",
                "task": "classification",
                "study_label": "roof",
                "predicted_label": "stove",
            },
        ])
        table = MODULE._clip_confusion_table(frame)
        self.assertEqual(table.loc["roof", "roof"], 1)
        self.assertEqual(table.loc["roof", "stove"], 1)

    def test_prompt_delta_is_reported_in_percentage_points(self):
        category = pd.DataFrame([
            {
                "model": "blip_baseline",
                "task": "captioning",
                "study_label": "roof",
                "score": 0.50,
            },
            {
                "model": "blip_prompted",
                "task": "captioning",
                "study_label": "roof",
                "score": 0.25,
            },
        ])
        result = MODULE._prompt_delta_table(category)
        self.assertAlmostEqual(result.loc[0, "change_percentage_points"], -25.0)


if __name__ == "__main__":
    unittest.main()
