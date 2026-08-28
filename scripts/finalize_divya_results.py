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


def validate_complete_predictions(frame: pd.DataFrame) -> None:
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
        default=Path("results/divya/qwen_full/benchmark_predictions.csv"),
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
