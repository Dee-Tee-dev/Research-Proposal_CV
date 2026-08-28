from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_divya_results",
    REPO_ROOT / "scripts" / "finalize_divya_results.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def complete_frame() -> pd.DataFrame:
    categories = (
        "footwear",
        "light source",
        "roof",
        "stove",
        "switch",
        "trash container",
    )
    groups = (
        ("clip", "classification"),
        ("blip_baseline", "captioning"),
        ("blip_prompted", "captioning"),
        ("qwen", "classification"),
        ("qwen", "captioning"),
    )
    rows = []
    for category_index, category in enumerate(categories):
        for quartile_index, quartile in enumerate(("Q1", "Q2", "Q3", "Q4")):
            for item in range(7):
                image_id = f"{category_index}-{quartile_index}-{item}"
                for model, task in groups:
                    rows.append({
                        "image_id": image_id,
                        "model": model,
                        "task": task,
                        "study_label": category,
                        "income_quartile": quartile,
                        "metric_value": True,
                    })
    return pd.DataFrame(rows)


class FinalizeDivyaResultsTests(unittest.TestCase):
    def test_complete_balanced_frame_passes(self):
        MODULE.validate_complete_predictions(complete_frame())

    def test_missing_qwen_row_fails(self):
        frame = complete_frame()
        frame = frame.drop(frame[(frame["model"] == "qwen")].index[0])
        with self.assertRaisesRegex(ValueError, "has 167 rows"):
            MODULE.validate_complete_predictions(frame)

    def test_duplicate_image_id_fails(self):
        frame = complete_frame()
        mask = (frame["model"] == "qwen") & (frame["task"] == "captioning")
        indices = frame[mask].index[:2]
        frame.loc[indices[1], "image_id"] = frame.loc[indices[0], "image_id"]
        with self.assertRaisesRegex(ValueError, "unique image IDs"):
            MODULE.validate_complete_predictions(frame)


if __name__ == "__main__":
    unittest.main()
