#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Decisions from visual inspection of Divya's deterministic 84-image half.
BOTH_CLEAR_FALSE_NEGATIVES = {
    "5d4bdf69cf0b3a0f3f337997": "basket used as a container",
    "5d4be03ccf0b3a0f3f338dde": "chandelier misspelled as chandel",
    "5d4be09ccf0b3a0f3f3398bb": "flip-flops",
    "5d4be76ecf0b3a0f3f3456d4": "garbage bag used as a container",
    "5d4be7afcf0b3a0f3f345e48": "bucket used as a waste container",
    "5d4beebecf0b3a0f3f351f34": "flip-flops",
    "5ec4f9e1f0611d7ddd741b27": "garbage bag or basket",
    "5ec4fb75f0611d7ddd74284a": "Crocs footwear",
}

BOTH_AMBIGUOUS = {
    "5d4be7b6cf0b3a0f3f345f1c": "trash is visible but no clear container",
    "5d4be956cf0b3a0f3f3486e4": "waste area and dumpsters share the scene",
    "5d4beee5cf0b3a0f3f3523d2": "open waste area rather than one container",
    "5ec4f81af0611d7ddd740be2": "correct electrical device but not specifically a switch",
}

# Overrides for cases where baseline and prompted captions need different decisions.
ROW_OVERRIDES = {
    ("5d4beab2cf0b3a0f3f34aea4", "blip_baseline"): (
        "yes", "accurate", "accepted_synonym_missing", "electrical switch panel"
    ),
    ("5d4bef87cf0b3a0f3f3536b6", "blip_prompted"): (
        "yes", "accurate", "accepted_synonym_missing", "plastic container"
    ),
    ("5d4bf191cf0b3a0f3f356bb4", "blip_baseline"): (
        "uncertain", "ambiguous", "label_or_scene_ambiguity", "garbage is mentioned but no clear container"
    ),
    ("5d4bf27dcf0b3a0f3f358554", "blip_baseline"): (
        "yes", "accurate", "accepted_synonym_missing", "garbage in a wooden box"
    ),
    ("5d4bf27dcf0b3a0f3f358554", "blip_prompted"): (
        "uncertain", "ambiguous", "label_or_scene_ambiguity", "wooden box is visible but its waste function is unclear"
    ),
    ("5d4bf349cf0b3a0f3f359cee", "blip_prompted"): (
        "uncertain", "ambiguous", "label_or_scene_ambiguity", "small plastic cup or container"
    ),
    ("5f8c634fe894f07b8477dfbe", "blip_baseline"): (
        "uncertain", "ambiguous", "related_object_only", "outlet named on a mixed switchboard"
    ),
    ("5ec4fb75f0611d7ddd74284a", "blip_prompted"): (
        "yes", "disfluent", "repetition", "Crocs identified but repeated many times"
    ),
}


def finalize(frame: pd.DataFrame) -> pd.DataFrame:
    reviewed = frame.copy()
    review_columns = [
        "manual_object_correct",
        "caption_quality",
        "error_type",
        "review_notes",
    ]
    for column in review_columns:
        reviewed[column] = reviewed[column].astype("object")
    for index, row in reviewed.iterrows():
        if bool(row["metric_value"]):
            decision = (
                "yes",
                "accurate",
                "none",
                "accepted-term match confirmed in caption context",
            )
        else:
            decision = (
                "no",
                "inaccurate",
                "wrong_object",
                "caption does not identify the study object",
            )

        image_id = str(row["image_id"])
        if not bool(row["metric_value"]) and image_id in BOTH_CLEAR_FALSE_NEGATIVES:
            decision = (
                "yes",
                "accurate",
                "accepted_synonym_missing",
                BOTH_CLEAR_FALSE_NEGATIVES[image_id],
            )
        if not bool(row["metric_value"]) and image_id in BOTH_AMBIGUOUS:
            decision = (
                "uncertain",
                "ambiguous",
                "label_or_scene_ambiguity",
                BOTH_AMBIGUOUS[image_id],
            )
        decision = ROW_OVERRIDES.get((image_id, str(row["model"])), decision)
        reviewed.loc[index, review_columns] = decision
    return reviewed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("paper/review/divya_caption_review.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/review/divya_caption_review_completed.csv"),
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    reviewed = finalize(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reviewed.to_csv(args.output, index=False)
    print(reviewed.groupby(["model", "manual_object_correct"]).size().to_string())
    print(f"Completed review written to {args.output}")


if __name__ == "__main__":
    main()
