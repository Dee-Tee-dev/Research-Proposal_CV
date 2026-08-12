from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vlm_gap.metrics import summarize_benchmark, summarize_income_gaps  # noqa: E402
from vlm_gap.models import parse_category_response  # noqa: E402


class CategoryParserTests(unittest.TestCase):
    def test_exact_multiword_label(self):
        self.assertEqual(parse_category_response("trash container"), "trash container")

    def test_label_inside_short_answer(self):
        self.assertEqual(parse_category_response("The answer is: light source."), "light source")

    def test_partial_word_is_not_accepted(self):
        self.assertIsNone(parse_category_response("a lighted room"))

    def test_unknown_answer_is_unparsed(self):
        self.assertIsNone(parse_category_response("chair"))


class BenchmarkMetricTests(unittest.TestCase):
    @staticmethod
    def frame() -> pd.DataFrame:
        rows = []
        for quartile, values in (("Q1", [0, 0]), ("Q4", [1, 1])):
            for label, value in zip(("roof", "stove"), values):
                rows.append({
                    "model": "clip",
                    "task": "classification",
                    "image_id": f"{quartile}-{label}",
                    "study_label": label,
                    "income_quartile": quartile,
                    "metric_value": value,
                })
        return pd.DataFrame(rows)

    def test_long_format_summary(self):
        summary = summarize_benchmark(self.frame(), "income_quartile")
        self.assertEqual(summary["n"].tolist(), [2, 2])
        self.assertEqual(summary["metric"].unique().tolist(), ["top1_accuracy"])

    def test_income_gap_direction(self):
        gaps = summarize_income_gaps(self.frame(), bootstrap_samples=20, seed=4)
        self.assertEqual(gaps.loc[0, "q4_minus_q1_gap"], 1.0)
        self.assertEqual(gaps.loc[0, "gap_ci_95_low"], 1.0)
        self.assertEqual(gaps.loc[0, "gap_ci_95_high"], 1.0)


if __name__ == "__main__":
    unittest.main()
