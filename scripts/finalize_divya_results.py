#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vlm_gap.evaluation import write_benchmark_results  # noqa: E402


EXPECTED_GROUPS = {
    ("clip", "classification"),
    ("blip_baseline", "captioning"),
    ("blip_prompted", "captioning"),
    ("qwen", "classification"),
    ("qwen", "captioning"),
}
EXPECTED_IMAGES = 168
PAPER_ASSET_NAMES = (
    "scores_by_income.csv",
    "scores_by_income.png",
    "scores_by_category.csv",
    "scores_by_category.png",
    "clip_confusion_matrix.csv",
    "clip_confusion_matrix.png",
    "blip_prompt_delta_by_category.csv",
    "blip_prompt_delta_by_category.png",
)
QWEN_REVIEW_ROWS = 84
RESULT_ORDER = (
    ("clip", "classification", "CLIP classification accuracy"),
    ("blip_baseline", "captioning", "BLIP baseline caption recall"),
    ("blip_prompted", "captioning", "BLIP prompted caption recall"),
    ("qwen", "classification", "Qwen classification accuracy"),
    ("qwen", "captioning", "Qwen caption recall"),
)


def _read_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    frame = pd.read_csv(path)
    required = {
        "image_id",
        "model",
        "task",
        "study_label",
        "income_quartile",
        "metric_value",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def _metric_boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    unexpected = set(normalized.unique()) - {"true", "false", "1", "0"}
    if unexpected:
        raise ValueError(f"Unexpected metric values: {sorted(unexpected)}")
    return normalized.isin({"true", "1"})


def validate_complete_predictions(frame: pd.DataFrame) -> None:
    _metric_boolean(frame["metric_value"])
    groups = set(zip(frame["model"], frame["task"]))
    if groups != EXPECTED_GROUPS:
        missing = sorted(EXPECTED_GROUPS - groups)
        unexpected = sorted(groups - EXPECTED_GROUPS)
        raise ValueError(
            f"Unexpected model-task groups; missing={missing}, unexpected={unexpected}"
        )

    reference_ids: set[str] | None = None
    for group in sorted(EXPECTED_GROUPS):
        model, task = group
        subset = frame[(frame["model"] == model) & (frame["task"] == task)]
        ids = set(subset["image_id"].astype(str))
        if len(subset) != EXPECTED_IMAGES:
            raise ValueError(f"{model}/{task} has {len(subset)} rows, expected 168")
        if len(ids) != EXPECTED_IMAGES:
            raise ValueError(
                f"{model}/{task} has {len(ids)} unique image IDs, expected 168"
            )
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"{model}/{task} does not use the same 168 image IDs")

    cell_counts = (
        frame[["image_id", "study_label", "income_quartile"]]
        .drop_duplicates("image_id")
        .groupby(["study_label", "income_quartile"])
        .size()
    )
    if len(cell_counts) != 24 or not cell_counts.eq(7).all():
        raise ValueError("The merged image set is not balanced at seven per category-quartile cell")


def merge_predictions(clip_blip_path: Path, qwen_path: Path) -> pd.DataFrame:
    clip_blip = _read_predictions(clip_blip_path)
    qwen = _read_predictions(qwen_path)
    frame = pd.concat([clip_blip, qwen], ignore_index=True)
    validate_complete_predictions(frame)
    return frame.sort_values(["image_id", "model", "task"]).reset_index(drop=True)


def quantitative_summary_markdown(
    frame: pd.DataFrame,
    gaps: pd.DataFrame,
) -> str:
    lines = [
        "# Divya Quantitative Results",
        "",
        "Generated only after the complete five-condition validation passed.",
        "",
        "| Model and task | Q1 | Q2 | Q3 | Q4 | Overall | Q4–Q1 gap (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, task, label in RESULT_ORDER:
        subset = frame[(frame["model"] == model) & (frame["task"] == task)].copy()
        subset["metric_value"] = _metric_boolean(subset["metric_value"])
        scores = subset.groupby("income_quartile")["metric_value"].mean()
        correct = int(subset["metric_value"].sum())
        total = len(subset)
        gap = gaps[(gaps["model"] == model) & (gaps["task"] == task)]
        if len(gap) != 1:
            raise ValueError(f"Expected one income-gap row for {model}/{task}")
        gap_row = gap.iloc[0]
        cells = [f"{float(scores[quartile]):.1%}" for quartile in ("Q1", "Q2", "Q3", "Q4")]
        overall = f"{correct}/{total} ({correct / total:.1%})"
        interval = (
            f"{float(gap_row['q4_minus_q1_gap']) * 100:.1f} pp "
            f"[{float(gap_row['gap_ci_95_low']) * 100:.1f}, "
            f"{float(gap_row['gap_ci_95_high']) * 100:.1f}]"
        )
        lines.append(
            f"| {label} | {' | '.join(cells)} | {overall} | {interval} |"
        )

    lines.extend([
        "",
        "All quartile cells contain 42 images. Intervals use 2,000 "
        "category-stratified bootstrap resamples with seed 2026. These are "
        "associations within the selected balanced subset, not causal effects.",
        "",
    ])
    return "\n".join(lines)


def qwen_review_queue(review: pd.DataFrame) -> pd.DataFrame:
    qwen = review[review["model"] == "qwen"].copy()
    if len(qwen) != QWEN_REVIEW_ROWS:
        raise ValueError(
            f"Qwen review queue has {len(qwen)} rows, expected {QWEN_REVIEW_ROWS}"
        )
    if qwen["image_id"].nunique() != QWEN_REVIEW_ROWS:
        raise ValueError("Qwen review queue does not contain 84 unique image IDs")
    if set(qwen["task"]) != {"captioning"}:
        raise ValueError("Qwen review queue contains a non-captioning task")
    return qwen.sort_values("image_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, merge, and analyse all of Divya's completed results."
    )
    parser.add_argument(
        "--clip-blip",
        type=Path,
        default=Path("results/divya/clip_blip_full/benchmark_predictions.csv"),
    )
    parser.add_argument(
        "--qwen",
        type=Path,
        default=Path("results/divya/qwen_resumable_full/benchmark_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/divya/combined_full"),
    )
    parser.add_argument(
        "--paper-assets",
        type=Path,
        default=Path("paper/assets/divya"),
    )
    args = parser.parse_args()

    predictions = merge_predictions(args.clip_blip, args.qwen)
    write_benchmark_results(predictions, args.output_dir)
    gaps = pd.read_csv(args.output_dir / "income_gap_estimates.csv")
    (args.output_dir / "divya_quantitative_results.md").write_text(
        quantitative_summary_markdown(predictions, gaps),
        encoding="utf-8",
    )

    analysis_dir = args.output_dir / "analysis"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "analyse_divya_results.py"),
            "--predictions",
            str(args.output_dir / "benchmark_predictions.csv"),
            "--output-dir",
            str(analysis_dir),
        ],
        check=True,
    )
    review = pd.read_csv(analysis_dir / "divya_caption_review.csv")
    qwen_review_queue(review).to_csv(
        REPO_ROOT / "paper" / "review" / "divya_qwen_caption_review.csv",
        index=False,
    )
    args.paper_assets.mkdir(parents=True, exist_ok=True)
    for name in PAPER_ASSET_NAMES:
        source = analysis_dir / name
        destination = args.paper_assets / f"divya_{name}"
        shutil.copy2(source, destination)
    print(
        "Validated 840 prediction rows across five Divya model-task conditions; "
        f"combined results written to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
