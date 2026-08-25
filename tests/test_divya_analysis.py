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


if __name__ == "__main__":
    unittest.main()
