from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vlm_gap.config import CATEGORY_LABELS, DEFAULT_MANIFEST  # noqa: E402
from vlm_gap.data import load_manifest, validate_manifest  # noqa: E402
from vlm_gap.metrics import caption_mentions_object  # noqa: E402


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_manifest(DEFAULT_MANIFEST)

    def test_manifest_validation(self):
        self.assertEqual(validate_manifest(self.rows), [])

    def test_ids_are_unique(self):
        ids = [row.image_id for row in self.rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_category_quartile_cell_has_seven_rows(self):
        counts = Counter(
            (row.study_label, row.income_quartile) for row in self.rows
        )
        for label in CATEGORY_LABELS:
            for quartile in ("Q1", "Q2", "Q3", "Q4"):
                self.assertEqual(counts[(label, quartile)], 7)

    def test_all_source_rows_are_single_class(self):
        self.assertTrue(all(row.class_id for row in self.rows))


class CaptionMetricTests(unittest.TestCase):
    def test_whole_term_match(self):
        self.assertTrue(
            caption_mentions_object(
                "A small lantern is hanging from the ceiling.",
                ("light", "lamp", "lantern"),
            )
        )

    def test_partial_word_does_not_match(self):
        self.assertFalse(caption_mentions_object("A lampshade.", ("lamp",)))


if __name__ == "__main__":
    unittest.main()

