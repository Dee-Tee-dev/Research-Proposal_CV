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

import matplotlib

matplotlib.use("Agg")

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
CATEGORY_ORDER = (
    "footwear",
    "light source",
    "roof",
    "stove",
    "switch",
    "trash container",
)


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
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    for axis, task, title, ylabel in (
        (axes[0], "classification", "Classification", "Top-1 accuracy"),
        (axes[1], "captioning", "Captioning", "Accepted-term recall"),
    ):
        task_table = table[table["task"] == task]
        pivot = task_table.pivot(
            index="income_quartile",
            columns="model_label",
            values="score",
        ).reindex(QUARTILE_ORDER)
        pivot.plot(marker="o", linewidth=2, ax=axis)
        axis.set_xlabel("Income quartile")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Model", loc="best")
    figure.suptitle("Performance by income quartile")
    plt.tight_layout()
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_category(table: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for axis, task, title, ylabel in (
        (axes[0], "classification", "Classification", "Top-1 accuracy"),
        (axes[1], "captioning", "Captioning", "Accepted-term recall"),
    ):
        task_table = table[table["task"] == task]
        pivot = task_table.pivot(
            index="study_label",
            columns="model_label",
            values="score",
        )
        pivot.plot(kind="bar", ax=axis)
        axis.set_xlabel("Object category")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Model", loc="best")
    figure.suptitle("Performance by object category")
    plt.tight_layout()
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close()


def _clip_confusion_table(frame: pd.DataFrame) -> pd.DataFrame:
    clip = frame[
        (frame["model"] == "clip") & (frame["task"] == "classification")
    ].copy()
    if clip.empty:
        raise ValueError("CLIP classification rows are required for the confusion matrix.")
    if "predicted_label" not in clip.columns:
        raise ValueError("The prediction file does not contain predicted_label.")
    observed = set(clip["study_label"]) | set(clip["predicted_label"])
    labels = [label for label in CATEGORY_ORDER if label in observed]
    return pd.crosstab(
        pd.Categorical(clip["study_label"], categories=labels, ordered=True),
        pd.Categorical(clip["predicted_label"], categories=labels, ordered=True),
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)


def _plot_clip_confusion(counts: pd.DataFrame, output: Path) -> None:
    row_totals = counts.sum(axis=1).replace(0, pd.NA)
    proportions = counts.div(row_totals, axis=0).fillna(0)
    figure, axis = plt.subplots(figsize=(7.3, 6.2))
    image = axis.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(counts.columns)), counts.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(counts.index)), counts.index)
    axis.set_xlabel("Predicted category")
    axis.set_ylabel("True category")
    axis.set_title("CLIP category confusion (n = 168)")
    for row in range(len(counts.index)):
        for column in range(len(counts.columns)):
            value = proportions.iloc[row, column]
            count = int(counts.iloc[row, column])
            colour = "white" if value >= 0.55 else "black"
            axis.text(
                column,
                row,
                f"{count}\n{value:.0%}",
                ha="center",
                va="center",
                color=colour,
                fontsize=8.5,
            )
    colourbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colourbar.set_label("Share within true category")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _prompt_delta_table(category: pd.DataFrame) -> pd.DataFrame:
    captions = category[
        (category["task"] == "captioning")
        & category["model"].isin(("blip_baseline", "blip_prompted"))
    ]
    pivot = captions.pivot(index="study_label", columns="model", values="score")
    required = {"blip_baseline", "blip_prompted"}
    if not required.issubset(pivot.columns):
        raise ValueError("Both BLIP baseline and prompted scores are required.")
    output = pivot.loc[:, ["blip_baseline", "blip_prompted"]].copy()
    output["change_percentage_points"] = (
        output["blip_prompted"] - output["blip_baseline"]
    ) * 100
    labels = [label for label in CATEGORY_ORDER if label in output.index]
    return output.reindex(labels).reset_index()


def _plot_prompt_delta(table: pd.DataFrame, output: Path) -> None:
    plot_table = table.sort_values("change_percentage_points")
    values = plot_table["change_percentage_points"]
    colours = ["#d97706" if value < 0 else "#2563eb" for value in values]
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    bars = axis.barh(plot_table["study_label"], values, color=colours)
    axis.axvline(0, color="#374151", linewidth=1)
    axis.set_xlabel("Prompted minus baseline recall (percentage points)")
    axis.set_ylabel("Object category")
    axis.set_title("Effect of the label-free BLIP prompt by category (n = 28 each)")
    axis.grid(axis="x", alpha=0.25)
    margin = max(2.0, float(values.abs().max()) * 0.05)
    for bar, value in zip(bars, values):
        axis.text(
            value + (margin if value >= 0 else -margin),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )
    lower = min(-2.0, float(values.min()) - 3 * margin)
    upper = max(2.0, float(values.max()) + 3 * margin)
    axis.set_xlim(lower, upper)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


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
        "task",
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
    confusion = _clip_confusion_table(frame)
    confusion.to_csv(args.output_dir / "clip_confusion_matrix.csv")
    _plot_clip_confusion(confusion, args.output_dir / "clip_confusion_matrix.png")
    prompt_delta = _prompt_delta_table(category)
    prompt_delta.to_csv(args.output_dir / "blip_prompt_delta_by_category.csv", index=False)
    _plot_prompt_delta(prompt_delta, args.output_dir / "blip_prompt_delta_by_category.png")
    _write_review_queue(frame, args.output_dir / "divya_caption_review.csv")
    print(f"Divya analysis written to {args.output_dir}")


if __name__ == "__main__":
    main()
