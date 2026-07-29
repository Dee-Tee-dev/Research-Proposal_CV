#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vlm_gap.config import DEFAULT_MANIFEST  # noqa: E402
from vlm_gap.data import load_manifest, validate_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the fixed study manifest.")
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Also verify that every selected image exists and can be opened.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest(DEFAULT_MANIFEST)
    errors = validate_manifest(rows)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"PASS: {len(rows)} unique balanced manifest rows")

    if args.check_images:
        image_dir = REPO_ROOT / "data" / "images"
        missing: list[str] = []
        invalid: list[str] = []
        for row in rows:
            image_path = row.image_path(image_dir)
            if not image_path.is_file():
                missing.append(row.image_name)
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, ValueError):
                invalid.append(row.image_name)

        if missing or invalid:
            for filename in missing:
                print(f"ERROR: missing image: {filename}")
            for filename in invalid:
                print(f"ERROR: invalid image: {filename}")
            raise SystemExit(1)
        print(f"PASS: {len(rows)} selected images exist and are readable")


if __name__ == "__main__":
    main()
