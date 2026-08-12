from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def caption_mentions_object(caption: str, accepted_terms: Iterable[str]) -> bool:
    normalized_caption = f" {normalize_text(caption)} "
    for term in accepted_terms:
        normalized_term = normalize_text(term)
        if normalized_term and f" {normalized_term} " in normalized_caption:
            return True
    return False


def summarize_predictions(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    required = {
        group_column,
        "clip_top1_correct",
        "clip_correct_rank",
        "baseline_term_match",
        "prompted_term_match",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")

    return (
        df.groupby(group_column, dropna=False)
        .agg(
            n=("image_id", "count"),
            clip_top1_accuracy=("clip_top1_correct", "mean"),
            clip_mean_correct_rank=("clip_correct_rank", "mean"),
            baseline_caption_recall=("baseline_term_match", "mean"),
            prompted_caption_recall=("prompted_term_match", "mean"),
        )
        .reset_index()
        .assign(
            caption_recall_change=lambda frame: (
                frame["prompted_caption_recall"]
                - frame["baseline_caption_recall"]
            )
        )
    )


def summarize_benchmark(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Summarize the long-format benchmark using one primary metric per task."""
    required = {group_column, "model", "task", "image_id", "metric_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing benchmark columns: {sorted(missing)}")

    summary = (
        df.groupby(["model", "task", group_column], dropna=False)
        .agg(
            n=("image_id", "count"),
            primary_score=("metric_value", "mean"),
        )
        .reset_index()
    )
    summary["metric"] = summary["task"].map({
        "classification": "top1_accuracy",
        "captioning": "accepted_term_recall",
        "detection": "image_level_hit_rate",
    })
    return summary[
        ["model", "task", group_column, "n", "metric", "primary_score"]
    ]


def summarize_income_gaps(
    df: pd.DataFrame,
    bootstrap_samples: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Estimate Q4-minus-Q1 gaps with category-stratified bootstrap intervals."""
    required = {
        "model",
        "task",
        "study_label",
        "income_quartile",
        "metric_value",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing benchmark columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    records = []
    for (model, task), group in df.groupby(["model", "task"], sort=True):
        q1 = group[group["income_quartile"] == "Q1"]
        q4 = group[group["income_quartile"] == "Q4"]
        if q1.empty or q4.empty:
            continue
        q1_score = float(q1["metric_value"].astype(float).mean())
        q4_score = float(q4["metric_value"].astype(float).mean())
        bootstrapped = []
        for _ in range(bootstrap_samples):
            sampled_scores = {}
            for quartile, quartile_group in (("Q1", q1), ("Q4", q4)):
                cell_means = []
                for _, cell in quartile_group.groupby("study_label"):
                    values = cell["metric_value"].astype(float).to_numpy()
                    sampled = rng.choice(values, size=len(values), replace=True)
                    cell_means.append(float(sampled.mean()))
                sampled_scores[quartile] = float(np.mean(cell_means))
            bootstrapped.append(sampled_scores["Q4"] - sampled_scores["Q1"])
        low, high = np.quantile(bootstrapped, [0.025, 0.975])
        records.append({
            "model": model,
            "task": task,
            "q1_score": q1_score,
            "q4_score": q4_score,
            "q4_minus_q1_gap": q4_score - q1_score,
            "gap_ci_95_low": float(low),
            "gap_ci_95_high": float(high),
            "bootstrap_samples": bootstrap_samples,
        })
    return pd.DataFrame(records)
