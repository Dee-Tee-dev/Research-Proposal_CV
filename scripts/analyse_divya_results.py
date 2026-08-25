#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "vlm-gap-matplotlib"),
)

import matplotlib.pyplot as plt
import pandas as pd


DIVYA_MODELS = ("clip", "blip_baseline", "blip_prompted", "qwen")
MODEL_LABELS = {
    "clip": "CLIP",
    "blip_baseline": "BLIP baseline",
    "blip_prompted": "BLIP prompted",
    "qwen": "Qwen2.5-VL",
}
QUARTILE_ORDER = ("Q1", "Q2", "Q3", "Q4")


def _as_boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    unexpected = set(normalized.unique()) - {"true", "false", "1", "0"}
    if unexpected:
        raise ValueError(f"Unexpected metric values: {sorted(unexpected)}")
    return normalized.isin({"true", "1"})


def _load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "image_id",
        "model",
        "task",
        "study_label",
        "income_quartile",
        "metric_value",
        "raw_output",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    frame = frame[frame["model"].isin(DIVYA_MODELS)].copy()
    if frame.empty:
        raise ValueError("No CLIP, BLIP, or Qwen predictions were found.")
    frame["metric_value"] = _as_boolean(frame["metric_value"])
    frame["model_label"] = frame["model"].map(MODEL_LABELS)
    return frame


def _score_table(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    return (
        frame.groupby(["model", "model_label", "task", group], sort=False)
        .agg(n=("image_id", "count"), score=("metric_value", "mean"))
        .reset_index()
    )


def _plot_income(table: pd.DataFrame, output: Path) -> None:
    pivot = table.pivot(index="income_quartile", columns="model_label", values="score")
    pivot = pivot.reindex(QUARTILE_ORDER)
    axis = pivot.plot(marker="o", linewidth=2, figsize=(8, 4.8))
    axis.set_xlabel("Income quartile")
    axis.set_ylabel("Primary task score")
    axis.set_ylim(0, 1)
    axis.set_title("Divya baselines by income quartile")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_category(table: pd.DataFrame, output: Path) -> None:
    pivot = table.pivot(index="study_label", columns="model_label", values="score")
    axis = pivot.plot(kind="bar", figsize=(10, 5))
    axis.set_xlabel("Object category")
    axis.set_ylabel("Primary task score")
    axis.set_ylim(0, 1)
    axis.set_title("Divya baselines by object category")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close()


def _write_review_queue(frame: pd.DataFrame, output: Path) -> None:
    captions = frame[frame["task"] == "captioning"].copy()
    image_ids = sorted(captions["image_id"].unique())
    divya_ids = set(image_ids[::2])
    review = captions[captions["image_id"].isin(divya_ids)].copy()
    review = review.sort_values(["image_id", "model"])
    review["manual_object_correct"] = ""
    review["caption_quality"] = ""
    review["error_type"] = ""
    review["review_notes"] = ""
    columns = [
        "image_id",
        "image_name",
        "study_label",
        "income_quartile",
        "model",
        "raw_output",
        "metric_value",
        "manual_object_correct",
        "caption_quality",
        "error_type",
        "review_notes",
    ]
    review[columns].to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Divya's quantitative tables, figures, and review queue."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/divya/benchmark_predictions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/divya/analysis"))
    args = parser.parse_args()

    frame = _load_predictions(args.predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    income = _score_table(frame, "income_quartile")
    category = _score_table(frame, "study_label")
    income.to_csv(args.output_dir / "scores_by_income.csv", index=False)
    category.to_csv(args.output_dir / "scores_by_category.csv", index=False)
    _plot_income(income, args.output_dir / "scores_by_income.png")
    _plot_category(category, args.output_dir / "scores_by_category.png")
    _write_review_queue(frame, args.output_dir / "divya_caption_review.csv")
    print(f"Divya analysis written to {args.output_dir}")


if __name__ == "__main__":
    main()
