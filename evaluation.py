from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .config import OBJECT_PROMPT
from .data import ManifestRow
from .metrics import caption_mentions_object, summarize_predictions
from .models import BLIPCaptioner, CLIPClassifier


def evaluate_rows(
    rows: list[ManifestRow],
    image_dir: Path,
    clip: CLIPClassifier,
    blip: BLIPCaptioner,
) -> pd.DataFrame:
    results = []
    for row in tqdm(rows, desc="Evaluating images"):
        image_path = row.image_path(image_dir)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing {image_path}. Run scripts/download_images.py first."
            )
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
            clip_result = clip.classify(image, correct_label=row.study_label)
            baseline_caption = blip.caption(image)
            prompted_caption = blip.caption(image, prompt=OBJECT_PROMPT)

        results.append(
            {
                "image_id": row.image_id,
                "image_name": row.image_name,
                "study_label": row.study_label,
                "income": row.income,
                "income_quartile": row.income_quartile,
                "region": row.region,
                "country_id": row.country_id,
                "country_name": row.country_name,
                "clip_prediction": clip_result.label,
                "clip_confidence": clip_result.confidence,
                "clip_top1_correct": clip_result.label == row.study_label,
                "clip_correct_rank": clip_result.correct_rank,
                "clip_scores": json.dumps(clip_result.scores, sort_keys=True),
                "baseline_caption": baseline_caption,
                "prompted_caption": prompted_caption,
                "baseline_term_match": caption_mentions_object(
                    baseline_caption,
                    row.accepted_caption_terms,
                ),
                "prompted_term_match": caption_mentions_object(
                    prompted_caption,
                    row.accepted_caption_terms,
                ),
                "accepted_caption_terms": json.dumps(row.accepted_caption_terms),
            }
        )
    return pd.DataFrame(results)


def write_results(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "predictions.csv", index=False)
    summarize_predictions(df, "income_quartile").to_csv(
        output_dir / "summary_by_income.csv",
        index=False,
    )
    summarize_predictions(df, "study_label").to_csv(
        output_dir / "summary_by_category.csv",
        index=False,
    )

