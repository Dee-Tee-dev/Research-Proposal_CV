#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


PAGE_SIZE = (1500, 1100)
GRID = (2, 2)
MARGIN = 30
GAP = 20
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial.ttf")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = (
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
        if bold
        else FONT_PATH
    )
    return ImageFont.truetype(str(path), size)


def validate_queue(frame: pd.DataFrame, expected_rows: int | None = 84) -> None:
    required = {
        "image_id",
        "image_name",
        "study_label",
        "income_quartile",
        "model",
        "raw_output",
        "metric_value",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Review queue is missing columns: {sorted(missing)}")
    if set(frame["model"]) != {"qwen"}:
        raise ValueError("Review queue must contain Qwen captions only")
    if frame["image_id"].duplicated().any():
        raise ValueError("Review queue contains duplicate image IDs")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Review queue has {len(frame)} rows, expected {expected_rows}")


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#eeeeee")
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def create_review_sheets(
    frame: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    expected_rows: int | None = 84,
) -> list[Path]:
    validate_queue(frame, expected_rows=expected_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values("image_id").reset_index(drop=True)
    columns, rows = GRID
    tile_width = (PAGE_SIZE[0] - 2 * MARGIN - (columns - 1) * GAP) // columns
    tile_height = (PAGE_SIZE[1] - 2 * MARGIN - (rows - 1) * GAP) // rows
    image_size = (300, tile_height - 80)
    title_font = _font(20, bold=True)
    body_font = _font(17)
    small_font = _font(14)
    pages = math.ceil(len(frame) / (columns * rows))
    outputs: list[Path] = []

    for page_index in range(pages):
        page = Image.new("RGB", PAGE_SIZE, "white")
        draw = ImageDraw.Draw(page)
        start = page_index * columns * rows
        page_rows = frame.iloc[start:start + columns * rows]
        for local_index, (_, record) in enumerate(page_rows.iterrows()):
            column = local_index % columns
            row = local_index // columns
            left = MARGIN + column * (tile_width + GAP)
            top = MARGIN + row * (tile_height + GAP)
            draw.rectangle(
                (left, top, left + tile_width, top + tile_height),
                outline="#777777",
                width=2,
            )
            number = start + local_index + 1
            title = (
                f"{number:02d}. {record['study_label']} | "
                f"{record['income_quartile']} | auto={record['metric_value']}"
            )
            draw.text((left + 14, top + 12), title, fill="black", font=title_font)

            image_path = image_dir / str(record["image_name"])
            if not image_path.exists():
                raise FileNotFoundError(f"Missing review image: {image_path}")
            with Image.open(image_path) as source:
                fitted = _fit_image(source, image_size)
            image_top = top + 55
            page.paste(fitted, (left + 14, image_top))

            text_left = left + image_size[0] + 32
            text_width_chars = max(25, (tile_width - image_size[0] - 55) // 9)
            caption = str(record["raw_output"])
            lines = textwrap.wrap(caption, width=text_width_chars) or ["(empty caption)"]
            draw.text((text_left, image_top), "Qwen caption:", fill="black", font=title_font)
            y = image_top + 34
            for line in lines[:12]:
                draw.text((text_left, y), line, fill="black", font=body_font)
                y += 24
            y += 16
            draw.text((text_left, y), "Decision: yes / no / uncertain", fill="black", font=body_font)
            y += 30
            draw.text((text_left, y), "Quality: accurate / inaccurate / ambiguous / disfluent", fill="black", font=small_font)
            y += 26
            draw.text((text_left, y), "Notes:", fill="black", font=body_font)
            draw.text(
                (left + 14, top + tile_height - 24),
                f"image_id: {record['image_id']}",
                fill="#444444",
                font=small_font,
            )

        output = output_dir / f"qwen_review_{page_index + 1:02d}_of_{pages:02d}.png"
        page.save(output, format="PNG", optimize=True)
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create contact sheets for Divya's 84-image Qwen caption review."
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("paper/review/divya_qwen_caption_review.csv"),
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/divya/qwen_review_sheets"),
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.queue)
    outputs = create_review_sheets(frame, args.image_dir, args.output_dir)
    print(f"Created {len(outputs)} Qwen review sheets in {args.output_dir}")


if __name__ == "__main__":
    main()
