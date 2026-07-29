from __future__ import annotations

import re
from collections.abc import Iterable

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

