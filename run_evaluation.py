#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vlm_gap.config import (  # noqa: E402
    DEFAULT_IMAGE_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_RESULTS_DIR,
)
from vlm_gap.data import load_manifest, validate_manifest  # noqa: E402
from vlm_gap.evaluation import evaluate_rows, write_results  # noqa: E402
from vlm_gap.models import BLIPCaptioner, CLIPClassifier, choose_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    errors = validate_manifest(rows)
    if errors:
        raise SystemExit("\n".join(errors))
    if args.limit is not None:
        rows = rows[: args.limit]

    device = args.device or choose_device()
    print(f"Using device: {device}")
    clip = CLIPClassifier(device=device)
    blip = BLIPCaptioner(device=device)
    predictions = evaluate_rows(rows, args.image_dir, clip, blip)
    write_results(predictions, args.output_dir)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()

