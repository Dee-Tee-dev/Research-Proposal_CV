from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import CATEGORY_LABELS


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    row_index: int
    image_name: str
    class_id: str
    study_label: str
    income: float
    income_quartile: str
    region: str
    country_id: str
    country_name: str
    topics: tuple[str, ...]
    source_synonyms: tuple[str, ...]
    accepted_caption_terms: tuple[str, ...]

    def image_path(self, image_dir: Path) -> Path:
        return image_dir / self.image_name


def _json_tuple(value: str) -> tuple[str, ...]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list, received: {value!r}")
    return tuple(str(item) for item in parsed)


def load_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))

    rows = []
    for record in records:
        rows.append(
            ManifestRow(
                image_id=record["id"],
                row_index=int(record["row_index"]),
                image_name=record["image_name"],
                class_id=record["class_id"],
                study_label=record["study_label"],
                income=float(record["income"]),
                income_quartile=record["income_quartile"],
                region=record["region"],
                country_id=record["country_id"],
                country_name=record["country_name"],
                topics=_json_tuple(record["topics"]),
                source_synonyms=_json_tuple(record["source_synonyms"]),
                accepted_caption_terms=_json_tuple(record["accepted_caption_terms"]),
            )
        )
    return rows


def validate_manifest(rows: Iterable[ManifestRow]) -> list[str]:
    rows = list(rows)
    errors = []
    ids = [row.image_id for row in rows]
    if len(rows) != 168:
        errors.append(f"Expected 168 rows, found {len(rows)}")
    if len(ids) != len(set(ids)):
        errors.append("Image IDs are not unique")

    expected_quartiles = {"Q1", "Q2", "Q3", "Q4"}
    counts = {}
    for row in rows:
        key = (row.study_label, row.income_quartile)
        counts[key] = counts.get(key, 0) + 1
        if not row.accepted_caption_terms:
            errors.append(f"{row.image_id}: accepted caption terms are empty")

    for label in CATEGORY_LABELS:
        for quartile in expected_quartiles:
            count = counts.get((label, quartile), 0)
            if count != 7:
                errors.append(f"{label}/{quartile}: expected 7 rows, found {count}")

    labels = {row.study_label for row in rows}
    if labels != set(CATEGORY_LABELS):
        errors.append(f"Unexpected category labels: {sorted(labels)}")
    quartiles = {row.income_quartile for row in rows}
    if quartiles != expected_quartiles:
        errors.append(f"Unexpected quartiles: {sorted(quartiles)}")
    return errors

