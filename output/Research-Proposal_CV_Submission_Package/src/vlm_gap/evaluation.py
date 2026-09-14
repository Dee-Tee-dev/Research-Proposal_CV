from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .config import OBJECT_PROMPT
from .data import ManifestRow
from .metrics import (
    caption_mentions_object,
    summarize_benchmark,
    summarize_income_gaps,
    summarize_predictions,
)
from .models import (
    BLIPCaptioner,
    CLIPClassifier,
    InternVisionLanguageModel,
    QwenVisionLanguageModel,
    YOLOWorldDetector,
)


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


def _metadata(row: ManifestRow) -> dict[str, object]:
    return {
        "image_id": row.image_id,
        "image_name": row.image_name,
        "study_label": row.study_label,
        "income": row.income,
        "income_quartile": row.income_quartile,
        "region": row.region,
        "country_id": row.country_id,
        "country_name": row.country_name,
    }


def evaluate_benchmark(
    rows: list[ManifestRow],
    image_dir: Path,
    models: dict[str, object],
) -> pd.DataFrame:
    """Run selected baselines and return one row per image, model, and task."""
    records: list[dict[str, object]] = []
    for row in tqdm(rows, desc="Running benchmark"):
        image_path = row.image_path(image_dir)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing {image_path}. Run scripts/download_images.py first."
            )
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
            for name, model in models.items():
                base = _metadata(row) | {
                    "model": name,
                    "accepted_caption_terms": json.dumps(
                        row.accepted_caption_terms
                    ),
                }
                if isinstance(model, CLIPClassifier):
                    prediction = model.classify(image, correct_label=row.study_label)
                    records.append(base | {
                        "task": "classification",
                        "raw_output": prediction.label,
                        "predicted_label": prediction.label,
                        "confidence": prediction.confidence,
                        "correct_rank": prediction.correct_rank,
                        "boxes": None,
                        "metric_value": prediction.label == row.study_label,
                    })
                elif isinstance(model, BLIPCaptioner):
                    for variant, prompt in (("baseline", None), ("prompted", OBJECT_PROMPT)):
                        caption = model.caption(image, prompt=prompt)
                        records.append(base | {
                            "model": f"{name}_{variant}",
                            "task": "captioning",
                            "raw_output": caption,
                            "predicted_label": None,
                            "confidence": None,
                            "correct_rank": None,
                            "boxes": None,
                            "metric_value": caption_mentions_object(
                                caption,
                                row.accepted_caption_terms,
                            ),
                        })
                elif isinstance(
                    model,
                    (QwenVisionLanguageModel, InternVisionLanguageModel),
                ):
                    prediction = model.classify(image)
                    records.append(base | {
                        "task": "classification",
                        "raw_output": prediction.raw_output,
                        "predicted_label": prediction.label,
                        "confidence": None,
                        "correct_rank": None,
                        "boxes": None,
                        "metric_value": prediction.label == row.study_label,
                    })
                    caption = model.caption(image)
                    records.append(base | {
                        "task": "captioning",
                        "raw_output": caption,
                        "predicted_label": None,
                        "confidence": None,
                        "correct_rank": None,
                        "boxes": None,
                        "metric_value": caption_mentions_object(
                            caption,
                            row.accepted_caption_terms,
                        ),
                    })
                elif isinstance(model, YOLOWorldDetector):
                    detection = model.detect(image)
                    records.append(base | {
                        "task": "detection",
                        "raw_output": ", ".join(detection.detected_labels),
                        "predicted_label": detection.label,
                        "confidence": detection.confidence,
                        "correct_rank": None,
                        "boxes": json.dumps(detection.boxes),
                        "metric_value": row.study_label in detection.detected_labels,
                    })
                else:
                    raise TypeError(f"Unsupported model adapter: {type(model).__name__}")
    return pd.DataFrame(records)


def write_benchmark_results(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "benchmark_predictions.csv", index=False)
    summarize_benchmark(df, "income_quartile").to_csv(
        output_dir / "benchmark_by_income.csv",
        index=False,
    )
    summarize_benchmark(df, "study_label").to_csv(
        output_dir / "benchmark_by_category.csv",
        index=False,
    )
    summarize_income_gaps(df).to_csv(
        output_dir / "income_gap_estimates.csv",
        index=False,
    )
    failures = df[df["metric_value"] == False]  # noqa: E712
    failures.to_csv(output_dir / "failure_cases.csv", index=False)
