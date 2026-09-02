#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


REQUIRED_SECTIONS = (
    "## 1. Introduction and Problem Motivation",
    "## 2. Dataset",
    "## 3. Methodology",
    "## 4. Quantitative Results",
)
REQUIRED_RESULT_LABELS = (
    "CLIP classification accuracy",
    "BLIP baseline caption recall",
    "BLIP prompted caption recall",
    "Qwen classification accuracy",
    "Qwen caption recall",
)


def validate_paper(text: str, paper_dir: Path) -> list[str]:
    issues: list[str] = []
    positions = [text.find(section) for section in REQUIRED_SECTIONS]
    missing_sections = [
        section for section, position in zip(REQUIRED_SECTIONS, positions)
        if position < 0
    ]
    if missing_sections:
        issues.append(f"Missing required sections: {missing_sections}")
    elif positions != sorted(positions):
        issues.append("Required sections are not in the expected order")

    lower = text.lower()
    if re.search(r"\bpending\b", lower):
        issues.append("Paper still contains a pending result")
    if "do not submit" in lower or "placeholder" in lower:
        issues.append("Paper still contains an internal submission warning")
    if re.search(r"^- \[ \]", text, flags=re.MULTILINE):
        issues.append("Paper still contains an unchecked checklist item")
    if "### quantitative-results checklist" in lower:
        issues.append("Internal quantitative-results checklist has not been removed")

    for label in REQUIRED_RESULT_LABELS:
        matches = re.findall(
            rf"^\|\s*{re.escape(label)}\s*\|.*$",
            text,
            flags=re.MULTILINE,
        )
        if len(matches) != 1:
            issues.append(f"Expected exactly one completed table row for {label}")
        elif matches[0].lower().count("pending"):
            issues.append(f"Result table row is incomplete: {label}")

    image_paths = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    if not image_paths:
        issues.append("Paper does not reference any result figures")
    for image_path in image_paths:
        if not (paper_dir / image_path).exists():
            issues.append(f"Missing referenced figure: {image_path}")

    if "168 unique images" not in text:
        issues.append("Paper does not state the unique-image sample size")
    if "seven images" not in lower or "category–quartile" not in text:
        issues.append("Paper does not state the balanced cell size")
    if "95%" not in text or "2,000" not in text or "seed 2026" not in text:
        issues.append("Paper does not fully state the bootstrap interval settings")
    if not re.search(
        r"do not\s+by themselves\s+show\s+that income\s+caused",
        lower,
    ):
        issues.append("Paper is missing the required non-causal interpretation caveat")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail if Divya's paper section is not submission-ready."
    )
    parser.add_argument(
        "paper",
        nargs="?",
        type=Path,
        default=Path("paper/divya_sections.md"),
    )
    args = parser.parse_args()
    text = args.paper.read_text(encoding="utf-8")
    issues = validate_paper(text, args.paper.parent)
    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        raise SystemExit(1)
    print(f"PASS: {args.paper} is submission-ready")


if __name__ == "__main__":
    main()
