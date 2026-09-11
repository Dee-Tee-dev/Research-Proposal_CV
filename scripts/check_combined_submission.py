from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results/combined_final/benchmark_predictions.csv"
PAPER = ROOT / "paper/final_short_paper.md"
REVIEW = ROOT / "results/human_review/human_review_consensus_final.csv"


EXPECTED = {
    ("blip_baseline", "captioning"): 89,
    ("blip_prompted", "captioning"): 86,
    ("clip", "classification"): 145,
    ("internvl", "captioning"): 98,
    ("internvl", "classification"): 146,
    ("qwen", "captioning"): 106,
    ("qwen", "classification"): 145,
    ("yolo_world", "detection"): 26,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1"}


def main() -> None:
    errors: list[str] = []
    rows = read_csv(PREDICTIONS)
    metadata = json.loads(
        (ROOT / "results/combined_final/clip_proposal_prompt_run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if metadata.get("clip_prompt_template") != "a photo of a {}":
        errors.append("combined CLIP metadata does not record the exact proposal template")
    if len(rows) != 1344:
        errors.append(f"combined predictions: expected 1344 rows, found {len(rows)}")

    keys = [(r["image_id"], r["model"], r["task"]) for r in rows]
    if len(set(keys)) != len(keys):
        errors.append("combined predictions contain duplicate image/model/task keys")
    if any(not r["metric_value"].strip() for r in rows):
        errors.append("combined predictions contain missing metric values")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["task"])].append(row)
    if set(grouped) != set(EXPECTED):
        errors.append(f"unexpected model-task conditions: {sorted(set(grouped) ^ set(EXPECTED))}")

    for condition, expected_correct in EXPECTED.items():
        condition_rows = grouped.get(condition, [])
        if len(condition_rows) != 168:
            errors.append(f"{condition}: expected 168 rows, found {len(condition_rows)}")
            continue
        if len({r["image_id"] for r in condition_rows}) != 168:
            errors.append(f"{condition}: image IDs are not unique")
        correct = sum(as_bool(r["metric_value"]) for r in condition_rows)
        if correct != expected_correct:
            errors.append(f"{condition}: expected {expected_correct} positive scores, found {correct}")
        cells = Counter((r["study_label"], r["income_quartile"]) for r in condition_rows)
        if len(cells) != 24 or set(cells.values()) != {7}:
            errors.append(f"{condition}: category-quartile balance is not 7 in all 24 cells")

    review = read_csv(REVIEW)
    if len(review) != 336:
        errors.append(f"human review: expected 336 final rows, found {len(review)}")
    for field in ("final_object_correct", "final_non_informative", "final_hallucination"):
        if any(r.get(field, "").strip() not in {"0", "1"} for r in review):
            errors.append(f"human review: {field} is incomplete")

    text = PAPER.read_text(encoding="utf-8")
    required = [
        "Diya Tiwari and Riya Katikar",
        "## 1 Introduction and problem motivation",
        "## 2 Methodology",
        "## 3 Results and analysis",
        "## 4 Discussion and limitations",
        "## Appendix A Related work and positioning",
        "do not establish that income caused errors",
        "1,344 prediction rows",
        "Gradio demo",
        "Rojas, W. A. G.",
    ]
    for phrase in required:
        if phrase not in text:
            errors.append(f"paper is missing required text: {phrase!r}")

    if errors:
        print("COMBINED SUBMISSION CHECK FAILED")
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("PASS: 1,344 rows across eight complete model-task conditions")
    print("PASS: every condition uses 168 unique images and 24 balanced cells")
    print("PASS: 336 adjudicated BLIP human-review rows are complete")
    print("PASS: final paper contains every rubric section and required caveats")


if __name__ == "__main__":
    main()
