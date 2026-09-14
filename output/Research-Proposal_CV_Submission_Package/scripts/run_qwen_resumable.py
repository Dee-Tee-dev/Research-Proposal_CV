#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vlm_gap.config import (  # noqa: E402
    CAPTION_PROMPT,
    CATEGORY_LABELS,
    CLASSIFICATION_PROMPT,
    DEFAULT_IMAGE_DIR,
    DEFAULT_MANIFEST,
    QWEN_MAX_PIXELS,
    QWEN_MIN_PIXELS,
    QWEN_MODEL_NAME,
)
from vlm_gap.data import ManifestRow, load_manifest, validate_manifest  # noqa: E402
from vlm_gap.evaluation import write_benchmark_results  # noqa: E402
from vlm_gap.metrics import caption_mentions_object  # noqa: E402
from vlm_gap.models import QwenVisionLanguageModel  # noqa: E402


PREDICTION_COLUMNS = (
    "image_id",
    "image_name",
    "study_label",
    "income",
    "income_quartile",
    "region",
    "country_id",
    "country_name",
    "model",
    "accepted_caption_terms",
    "task",
    "raw_output",
    "predicted_label",
    "confidence",
    "correct_rank",
    "boxes",
    "metric_value",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_config(manifest: Path, image_dir: Path, device: str, rows: int) -> dict:
    return {
        "model": QWEN_MODEL_NAME,
        "device": device,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _file_sha256(manifest),
        "image_dir": str(image_dir.resolve()),
        "rows_evaluated": rows,
        "categories": list(CATEGORY_LABELS),
        "classification_prompt": CLASSIFICATION_PROMPT.format(
            labels=", ".join(CATEGORY_LABELS)
        ),
        "caption_prompt": CAPTION_PROMPT,
        "qwen_min_pixels": QWEN_MIN_PIXELS,
        "qwen_max_pixels": QWEN_MAX_PIXELS,
    }


def _validate_resume_config(path: Path, expected: dict) -> None:
    if not path.exists():
        path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != expected:
        differing = sorted(
            key for key in set(existing) | set(expected)
            if existing.get(key) != expected.get(key)
        )
        raise ValueError(
            "Cannot resume Qwen with a changed run configuration; "
            f"differing fields: {differing}"
        )


def _validate_checkpoint(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    missing = set(PREDICTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Checkpoint is missing columns: {sorted(missing)}")
    if set(frame["model"]) != {"qwen"}:
        raise ValueError("Checkpoint contains a model other than qwen")
    if frame.duplicated(["image_id", "model", "task"]).any():
        raise ValueError("Checkpoint contains duplicate image/model/task rows")
    tasks_by_image = frame.groupby("image_id")["task"].agg(set)
    incomplete = tasks_by_image[tasks_by_image != {"classification", "captioning"}]
    if not incomplete.empty:
        raise ValueError(
            "Checkpoint contains an incomplete image; expected both tasks for "
            f"{incomplete.index.tolist()[:5]}"
        )
    return set(tasks_by_image.index.astype(str))


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _records_for_image(
    row: ManifestRow,
    image,
    model: QwenVisionLanguageModel,
) -> list[dict[str, object]]:
    base = {
        "image_id": row.image_id,
        "image_name": row.image_name,
        "study_label": row.study_label,
        "income": row.income,
        "income_quartile": row.income_quartile,
        "region": row.region,
        "country_id": row.country_id,
        "country_name": row.country_name,
        "model": "qwen",
        "accepted_caption_terms": json.dumps(row.accepted_caption_terms),
    }
    prediction = model.classify(image)
    classification = base | {
        "task": "classification",
        "raw_output": prediction.raw_output,
        "predicted_label": prediction.label,
        "confidence": None,
        "correct_rank": None,
        "boxes": None,
        "metric_value": prediction.label == row.study_label,
    }
    caption = model.caption(image)
    captioning = base | {
        "task": "captioning",
        "raw_output": caption,
        "predicted_label": None,
        "confidence": None,
        "correct_rank": None,
        "boxes": None,
        "metric_value": caption_mentions_object(
            caption,
            row.accepted_caption_terms,
        ),
    }
    return [classification, captioning]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen with atomic per-image checkpoints and safe resume."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/divya/qwen_resumable_full"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    errors = validate_manifest(rows)
    if errors:
        raise SystemExit("\n".join(errors))
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = _run_config(args.manifest, args.image_dir, args.device, len(rows))
    config_path = args.output_dir / "resume_config.json"
    _validate_resume_config(config_path, config)

    checkpoint_path = args.output_dir / "qwen_checkpoint.csv"
    if checkpoint_path.exists():
        checkpoint = pd.read_csv(checkpoint_path)
    else:
        checkpoint = pd.DataFrame(columns=PREDICTION_COLUMNS)
    completed_ids = _validate_checkpoint(checkpoint)
    manifest_ids = {row.image_id for row in rows}
    unexpected_ids = completed_ids - manifest_ids
    if unexpected_ids:
        raise ValueError(
            f"Checkpoint contains IDs outside this run: {sorted(unexpected_ids)[:5]}"
        )

    remaining = [row for row in rows if row.image_id not in completed_ids]
    print(
        f"Qwen resume state: {len(completed_ids)}/{len(rows)} images complete; "
        f"{len(remaining)} remaining"
    )
    if remaining:
        model = QwenVisionLanguageModel(device=args.device)
        records = checkpoint.to_dict("records")
        for row in tqdm(remaining, desc="Running resumable Qwen"):
            image_path = row.image_path(args.image_dir)
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image: {image_path}")
            with Image.open(image_path) as loaded:
                image = loaded.convert("RGB")
                records.extend(_records_for_image(row, image, model))
            checkpoint = pd.DataFrame(records, columns=PREDICTION_COLUMNS)
            _validate_checkpoint(checkpoint)
            _atomic_write_csv(checkpoint, checkpoint_path)
            print(f"Checkpointed {row.image_id}: {len(checkpoint) // 2}/{len(rows)}")

    final = pd.read_csv(checkpoint_path)
    completed_ids = _validate_checkpoint(final)
    if len(completed_ids) != len(rows):
        raise RuntimeError(
            f"Qwen run is incomplete: {len(completed_ids)}/{len(rows)} images"
        )
    write_benchmark_results(final, args.output_dir)
    metadata = config | {
        "created_utc": datetime.now(UTC).isoformat(),
        "bootstrap_seed": 2026,
        "bootstrap_samples": 2000,
        "checkpointed_per_image": True,
        "completed_images": len(completed_ids),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Completed Qwen benchmark written to {args.output_dir}")


if __name__ == "__main__":
    main()
