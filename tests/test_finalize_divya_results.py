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

    def test_paper_summary_contains_all_conditions_and_denominators(self):
        frame = complete_frame()
        gaps = pd.DataFrame([
            {
                "model": model,
                "task": task,
                "q4_minus_q1_gap": 0.0,
                "gap_ci_95_low": 0.0,
                "gap_ci_95_high": 0.0,
            }
            for model, task in MODULE.EXPECTED_GROUPS
        ])
        summary = MODULE.quantitative_summary_markdown(frame, gaps)
        self.assertIn("Qwen classification accuracy", summary)
        self.assertIn("Qwen caption recall", summary)
        self.assertEqual(summary.count("168/168 (100.0%)"), 5)
        self.assertIn("2,000 category-stratified bootstrap", summary)

    def test_qwen_review_queue_is_separate_and_has_84_unique_images(self):
        rows = []
        for index in range(84):
            for model in ("blip_baseline", "blip_prompted", "qwen"):
                rows.append({
                    "image_id": f"image-{index:03d}",
                    "model": model,
                    "task": "captioning",
                    "manual_object_correct": "",
                })
        queue = MODULE.qwen_review_queue(pd.DataFrame(rows))
        self.assertEqual(len(queue), 84)
        self.assertEqual(queue["image_id"].nunique(), 84)
        self.assertEqual(queue["model"].unique().tolist(), ["qwen"])

    def test_incomplete_qwen_review_queue_fails(self):
        frame = pd.DataFrame([
            {"image_id": "one", "model": "qwen", "task": "captioning"},
        ])
        with self.assertRaisesRegex(ValueError, "expected 84"):
            MODULE.qwen_review_queue(frame)


if __name__ == "__main__":
    unittest.main()
