#!/usr/bin/env python3
"""Record the decisions from Divya's 84-image Qwen caption audit.

The decision sequence follows the numbered, sorted review sheets produced by
``create_qwen_review_sheets.py``.  Keeping the reviewed judgements in this
small reproducible script makes the completed CSV traceable and prevents a
later analysis run from silently overwriting it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT = Path("paper/review/divya_qwen_caption_review.csv")
OUTPUT = Path("paper/review/divya_qwen_caption_review_completed.csv")

# yes = caption identifies the study object; no = it identifies another object;
# uncertain = the source label/image does not support a confident binary call.
DECISIONS = (
    "y", "y", "y", "y",  # sheet 01
    "y", "y", "y", "y",  # sheet 02
    "n", "y", "y", "y",  # sheet 03
    "n", "y", "y", "u",  # sheet 04
    "y", "u", "y", "y",  # sheet 05
    "n", "y", "n", "u",  # sheet 06
    "n", "y", "y", "y",  # sheet 07
    "y", "u", "y", "y",  # sheet 08
    "u", "y", "n", "y",  # sheet 09
    "n", "y", "u", "y",  # sheet 10
    "n", "n", "u", "y",  # sheet 11
    "u", "y", "y", "y",  # sheet 12
    "y", "y", "n", "y",  # sheet 13
    "y", "y", "y", "y",  # sheet 14
    "u", "u", "y", "y",  # sheet 15
    "y", "y", "y", "y",  # sheet 16
    "y", "y", "n", "y",  # sheet 17
    "n", "y", "y", "u",  # sheet 18
    "n", "n", "y", "y",  # sheet 19
    "y", "y", "y", "y",  # sheet 20
    "y", "y", "y", "y",  # sheet 21
)


def main() -> None:
    frame = pd.read_csv(INPUT)
    if len(frame) != 84 or frame["image_id"].nunique() != 84:
        raise ValueError("Expected the deterministic 84-image Qwen review queue")
    if list(frame["image_id"]) != sorted(frame["image_id"]):
        raise ValueError("Qwen review queue must remain in sorted image-ID order")
    if len(DECISIONS) != len(frame):
        raise ValueError(f"Expected 84 decisions, found {len(DECISIONS)}")
    if set(DECISIONS) - {"y", "n", "u"}:
        raise ValueError("Review decisions must use y, n, or u")

    decision_names = {"y": "yes", "n": "no", "u": "uncertain"}
    frame["manual_object_correct"] = [decision_names[value] for value in DECISIONS]
    frame["caption_quality"] = [
        "accurate" if value == "y" else "inaccurate" if value == "n" else "ambiguous"
        for value in DECISIONS
    ]

    automatic = frame["metric_value"].astype(str).str.lower().isin({"true", "1"})
    error_types: list[str] = []
    notes: list[str] = []
    for auto, decision in zip(automatic, frame["manual_object_correct"]):
        if decision == "uncertain":
            error_types.append("uncertain_source_or_target")
            notes.append("Image or source label does not support a confident binary target decision.")
        elif decision == "yes" and not auto:
            error_types.append("false_negative")
            notes.append("Caption identifies the target semantically, but the accepted-term metric misses it.")
        elif decision == "no" and auto:
            error_types.append("false_positive")
            notes.append("Caption contains a matching term but identifies another main object.")
        elif decision == "yes":
            error_types.append("none")
            notes.append("Caption identifies the study object and agrees with the automatic match.")
        else:
            error_types.append("none")
            notes.append("Caption misses the study object and agrees with the automatic non-match.")
    frame["error_type"] = error_types
    frame["review_notes"] = notes
    frame.to_csv(OUTPUT, index=False)
    print(f"Recorded {len(frame)} reviewed Qwen captions in {OUTPUT}")


if __name__ == "__main__":
    main()
