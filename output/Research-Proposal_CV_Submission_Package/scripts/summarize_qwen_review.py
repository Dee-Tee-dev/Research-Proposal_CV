#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_ROWS = 84
ALLOWED_DECISIONS = {"yes", "no", "uncertain"}
ALLOWED_QUALITY = {"accurate", "inaccurate", "ambiguous", "disfluent"}


def _as_boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    unexpected = set(normalized.unique()) - {"true", "false", "1", "0"}
    if unexpected:
        raise ValueError(f"Unexpected metric values: {sorted(unexpected)}")
    return normalized.isin({"true", "1"})


def validate_completed_review(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "image_id",
        "model",
        "metric_value",
        "manual_object_correct",
        "caption_quality",
        "error_type",
        "review_notes",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Completed review is missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_ROWS or frame["image_id"].nunique() != EXPECTED_ROWS:
        raise ValueError(
            f"Completed Qwen review must contain {EXPECTED_ROWS} unique images"
        )
    if set(frame["model"]) != {"qwen"}:
        raise ValueError("Completed review must contain Qwen captions only")

    reviewed = frame.copy()
    reviewed["metric_value"] = _as_boolean(reviewed["metric_value"])
    reviewed["manual_object_correct"] = (
        reviewed["manual_object_correct"].astype(str).str.strip().str.lower()
    )
    reviewed["caption_quality"] = (
        reviewed["caption_quality"].astype(str).str.strip().str.lower()
    )
    invalid_decisions = set(reviewed["manual_object_correct"]) - ALLOWED_DECISIONS
    if invalid_decisions:
        raise ValueError(f"Invalid or blank semantic decisions: {sorted(invalid_decisions)}")
    invalid_quality = set(reviewed["caption_quality"]) - ALLOWED_QUALITY
    if invalid_quality:
        raise ValueError(f"Invalid or blank quality decisions: {sorted(invalid_quality)}")
    if reviewed["error_type"].isna().any() or reviewed["review_notes"].isna().any():
        raise ValueError("Every Qwen review row must include error_type and review_notes")
    if reviewed["error_type"].astype(str).str.strip().eq("").any():
        raise ValueError("Every Qwen review row must include error_type")
    if reviewed["review_notes"].astype(str).str.strip().eq("").any():
        raise ValueError("Every Qwen review row must include review_notes")
    return reviewed


def summarize_review(frame: pd.DataFrame) -> dict[str, int | float]:
    reviewed = validate_completed_review(frame)
    automatic = reviewed["metric_value"]
    decisions = reviewed["manual_object_correct"]
    yes = int(decisions.eq("yes").sum())
    uncertain = int(decisions.eq("uncertain").sum())
    return {
        "n": len(reviewed),
        "automatic_matches": int(automatic.sum()),
        "automatic_recall": float(automatic.mean()),
        "semantic_yes": yes,
        "semantic_recall": yes / len(reviewed),
        "uncertain": uncertain,
        "semantic_upper_count": yes + uncertain,
        "semantic_upper_bound": (yes + uncertain) / len(reviewed),
        "clear_false_negatives": int((~automatic & decisions.eq("yes")).sum()),
        "clear_false_positives": int((automatic & decisions.eq("no")).sum()),
        "disfluent": int(reviewed["caption_quality"].eq("disfluent").sum()),
    }


def summary_markdown(summary: dict[str, int | float]) -> str:
    return (
        "# Divya Qwen Caption Audit\n\n"
        f"The deterministic review half contains {summary['n']} Qwen captions. "
        f"Automatic accepted-term recall is {summary['automatic_recall']:.1%} "
        f"({summary['automatic_matches']}/{summary['n']}). Strict semantic target "
        f"recall is {summary['semantic_recall']:.1%} "
        f"({summary['semantic_yes']}/{summary['n']}), with "
        f"{summary['uncertain']} uncertain cases and an upper bound of "
        f"{summary['semantic_upper_bound']:.1%} "
        f"({summary['semantic_upper_count']}/{summary['n']}). The audit identifies "
        f"{summary['clear_false_negatives']} clear automatic false negatives, "
        f"{summary['clear_false_positives']} clear automatic false positives, and "
        f"{summary['disfluent']} disfluent captions.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize Divya's completed Qwen caption review."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("paper/review/divya_qwen_caption_review_completed.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/divya/qwen_caption_review"),
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    summary = summarize_review(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qwen_caption_review_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "qwen_caption_review_summary.md").write_text(
        summary_markdown(summary),
        encoding="utf-8",
    )
    print(summary_markdown(summary).strip())


if __name__ == "__main__":
    main()
