from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_qwen_review",
    REPO_ROOT / "scripts" / "summarize_qwen_review.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def completed_review() -> pd.DataFrame:
    rows = []
    for index in range(84):
        automatic = index < 40
        decision = "yes" if index < 50 else ("uncertain" if index < 55 else "no")
        rows.append({
            "image_id": f"image-{index:03d}",
            "model": "qwen",
            "metric_value": automatic,
            "manual_object_correct": decision,
            "caption_quality": "accurate" if decision == "yes" else "inaccurate",
            "error_type": "none" if decision == "yes" else "wrong_object",
            "review_notes": "reviewed against image",
        })
    return pd.DataFrame(rows)


class SummarizeQwenReviewTests(unittest.TestCase):
    def test_summary_counts_metric_errors_and_bounds(self):
        summary = MODULE.summarize_review(completed_review())
        self.assertEqual(summary["automatic_matches"], 40)
        self.assertEqual(summary["semantic_yes"], 50)
        self.assertEqual(summary["uncertain"], 5)
        self.assertEqual(summary["semantic_upper_count"], 55)
        self.assertEqual(summary["clear_false_negatives"], 10)
        self.assertEqual(summary["clear_false_positives"], 0)

    def test_blank_decision_is_rejected(self):
        frame = completed_review()
        frame.loc[0, "manual_object_correct"] = ""
        with self.assertRaisesRegex(ValueError, "semantic decisions"):
            MODULE.validate_completed_review(frame)

    def test_wrong_row_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "84 unique images"):
            MODULE.validate_completed_review(completed_review().iloc[:-1])

    def test_non_qwen_row_is_rejected(self):
        frame = completed_review()
        frame.loc[0, "model"] = "blip_baseline"
        with self.assertRaisesRegex(ValueError, "Qwen captions only"):
            MODULE.validate_completed_review(frame)

    def test_markdown_contains_denominators(self):
        markdown = MODULE.summary_markdown(MODULE.summarize_review(completed_review()))
        self.assertIn("40/84", markdown)
        self.assertIn("50/84", markdown)
        self.assertIn("55/84", markdown)


if __name__ == "__main__":
    unittest.main()
