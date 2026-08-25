#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vlm_gap.config import (  # noqa: E402
    CAPTION_PROMPT,
    CATEGORY_LABELS,
    CLASSIFICATION_PROMPT,
    DEFAULT_IMAGE_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_RESULTS_DIR,
    INTERNVL_MODEL_NAME,
    QWEN_MODEL_NAME,
    QWEN_MAX_PIXELS,
    QWEN_MIN_PIXELS,
    YOLO_WORLD_MODEL_NAME,
)
from vlm_gap.data import load_manifest, validate_manifest  # noqa: E402
from vlm_gap.evaluation import evaluate_benchmark, write_benchmark_results  # noqa: E402
from vlm_gap.models import (  # noqa: E402
    BLIPCaptioner,
    CLIPClassifier,
    InternVisionLanguageModel,
    QwenVisionLanguageModel,
    YOLOWorldDetector,
    choose_device,
)


MODEL_FACTORIES = {
    "clip": CLIPClassifier,
    "blip": BLIPCaptioner,
    "qwen": QwenVisionLanguageModel,
    "internvl": InternVisionLanguageModel,
    "yolo_world": YOLOWorldDetector,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run selected baselines on the fixed Dollar Street subset."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_FACTORIES),
        default=list(MODEL_FACTORIES),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.25,
        help="YOLO-World confidence threshold (default: 0.25).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.yolo_confidence <= 1.0:
        parser.error("--yolo-confidence must be between 0 and 1")

    rows = load_manifest(args.manifest)
    errors = validate_manifest(rows)
    if errors:
        raise SystemExit("\n".join(errors))
    if args.limit is not None:
        rows = rows[: args.limit]

    device = args.device or choose_device()
    print(f"Using device: {device}")
    print(f"Loading models: {', '.join(args.models)}")
    models = {}
    for name in args.models:
        if name == "yolo_world":
            models[name] = YOLOWorldDetector(
                device=device,
                confidence=args.yolo_confidence,
            )
        else:
            models[name] = MODEL_FACTORIES[name](device=device)
    predictions = evaluate_benchmark(rows, args.image_dir, models)
    write_benchmark_results(predictions, args.output_dir)
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "models": args.models,
        "model_checkpoints": {
            "qwen": QWEN_MODEL_NAME,
            "internvl": INTERNVL_MODEL_NAME,
            "yolo_world": YOLO_WORLD_MODEL_NAME,
        },
        "categories": list(CATEGORY_LABELS),
        "classification_prompt": CLASSIFICATION_PROMPT.format(
            labels=", ".join(CATEGORY_LABELS)
        ),
        "caption_prompt": CAPTION_PROMPT,
        "device": device,
        "manifest": str(args.manifest.resolve()),
        "image_dir": str(args.image_dir.resolve()),
        "rows_evaluated": len(rows),
        "bootstrap_seed": 2026,
        "bootstrap_samples": 2000,
        "qwen_min_pixels": QWEN_MIN_PIXELS,
        "qwen_max_pixels": QWEN_MAX_PIXELS,
        "yolo_world_confidence_threshold": args.yolo_confidence,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Benchmark results written to {args.output_dir}")


if __name__ == "__main__":
    main()
