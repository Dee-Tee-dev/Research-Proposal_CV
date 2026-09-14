from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_qwen_resumable",
    REPO_ROOT / "scripts" / "run_qwen_resumable.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def checkpoint_frame() -> pd.DataFrame:
    base = {
        "image_id": "image-1",
        "image_name": "image-1.jpg",
        "study_label": "roof",
        "income": 100.0,
        "income_quartile": "Q1",
        "region": "af",
        "country_id": "lr",
        "country_name": "Liberia",
        "model": "qwen",
        "accepted_caption_terms": '["roof"]',
        "raw_output": "roof",
        "predicted_label": "roof",
        "confidence": None,
        "correct_rank": None,
        "boxes": None,
        "metric_value": True,
    }
    return pd.DataFrame([
        base | {"task": "classification"},
        base | {"task": "captioning", "predicted_label": None},
    ], columns=MODULE.PREDICTION_COLUMNS)


class QwenResumableTests(unittest.TestCase):
    def test_complete_image_is_resumable(self):
        self.assertEqual(
            MODULE._validate_checkpoint(checkpoint_frame()),
            {"image-1"},
        )

    def test_incomplete_image_is_rejected(self):
        frame = checkpoint_frame().iloc[:1]
        with self.assertRaisesRegex(ValueError, "incomplete image"):
            MODULE._validate_checkpoint(frame)

    def test_duplicate_task_is_rejected(self):
        frame = pd.concat(
            [checkpoint_frame(), checkpoint_frame().iloc[:1]],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE._validate_checkpoint(frame)

    def test_atomic_checkpoint_replaces_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.csv"
            MODULE._atomic_write_csv(checkpoint_frame(), path)
            loaded = pd.read_csv(path)
        self.assertEqual(len(loaded), 2)
        self.assertFalse(path.with_suffix(".csv.tmp").exists())

    def test_changed_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            MODULE._validate_resume_config(path, {"rows": 168, "pixels": 1})
            with self.assertRaisesRegex(ValueError, "changed run configuration"):
                MODULE._validate_resume_config(path, {"rows": 168, "pixels": 2})


if __name__ == "__main__":
    unittest.main()
