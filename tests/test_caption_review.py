from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_divya_caption_review",
    REPO_ROOT / "scripts" / "finalize_divya_caption_review.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CaptionReviewTests(unittest.TestCase):
    def test_clear_synonym_is_recorded_as_false_negative(self):
        frame = pd.DataFrame([{
            "image_id": "5d4be09ccf0b3a0f3f3398bb",
            "model": "blip_baseline",
            "metric_value": False,
            "manual_object_correct": "",
            "caption_quality": "",
            "error_type": "",
            "review_notes": "",
        }])
        result = MODULE.finalize(frame)
        self.assertEqual(result.loc[0, "manual_object_correct"], "yes")
        self.assertEqual(result.loc[0, "error_type"], "accepted_synonym_missing")

    def test_automatic_match_still_receives_context_confirmation(self):
        frame = pd.DataFrame([{
            "image_id": "example",
            "model": "blip_baseline",
            "metric_value": True,
            "manual_object_correct": "",
            "caption_quality": "",
            "error_type": "",
            "review_notes": "",
        }])
        result = MODULE.finalize(frame)
        self.assertEqual(result.loc[0, "manual_object_correct"], "yes")
        self.assertEqual(result.loc[0, "caption_quality"], "accurate")


if __name__ == "__main__":
    unittest.main()
