from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "create_qwen_review_sheets",
    REPO_ROOT / "scripts" / "create_qwen_review_sheets.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def review_frame(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "image_id": f"image-{index}",
            "image_name": f"image-{index}.jpg",
            "study_label": "roof",
            "income_quartile": "Q1",
            "model": "qwen",
            "raw_output": "A short description of a roof.",
            "metric_value": True,
        }
        for index in range(rows)
    ])


class QwenReviewSheetTests(unittest.TestCase):
    def test_four_rows_create_one_sheet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            output = root / "sheets"
            images.mkdir()
            frame = review_frame()
            for image_name in frame["image_name"]:
                Image.new("RGB", (120, 90), "navy").save(images / image_name)
            sheets = MODULE.create_review_sheets(
                frame,
                images,
                output,
                expected_rows=4,
            )
            self.assertEqual(len(sheets), 1)
            with Image.open(sheets[0]) as rendered:
                self.assertEqual(rendered.size, MODULE.PAGE_SIZE)

    def test_duplicate_image_is_rejected(self):
        frame = pd.concat([review_frame(1), review_frame(1)], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.validate_queue(frame, expected_rows=2)

    def test_non_qwen_model_is_rejected(self):
        frame = review_frame(1)
        frame.loc[0, "model"] = "blip_baseline"
        with self.assertRaisesRegex(ValueError, "Qwen captions only"):
            MODULE.validate_queue(frame, expected_rows=1)


if __name__ == "__main__":
    unittest.main()
