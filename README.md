# Socioeconomic and Regional Performance Gaps in Vision-Language Models

This repository contains the code, proposal, and reproducible data split for a
Computer Vision course project. The study compares contrastive, generative,
and detection baselines on household-object images from Dollar Street.

The main comparison is across four income quartiles. Regional results are
secondary because the available images are not balanced equally across regions.

## Study design

- Models: CLIP, BLIP, Qwen2.5-VL-3B, InternVL3.5-2B, and YOLO-World
- Categories: roof, light source, stove, trash container, switch, footwear
- Sample: 168 unique images
- Balance: 7 images per category in each of 4 income quartiles
- CLIP metrics: top-1 accuracy and correct-label rank
- BLIP/Qwen/InternVL caption metric: accepted-term recall plus semantic review
- Qwen/InternVL classification metric: forced-choice top-1 accuracy
- YOLO-World metric: image-level correct-class detection rate
- BLIP intervention: the label-free prefix
  `the main household object in this image is`

The general-purpose VLMs use the same forced-choice instruction and
deterministic decoding. YOLO-World uses the same six text categories. Because
Dollar Street does not include bounding boxes, detections are evaluated at the
image level; mAP and IoU are deliberately not reported.

The category names used for evaluation are intentionally broader than some
ImageNet labels. For example, a correct caption containing "lantern" should not
be rejected simply because the source ImageNet class is called "table lamp".

## Repository layout

```text
proposal/          One-page research proposal
data/metadata/     Normalized metadata for the exposed 1,600-row test split
data/splits/       Fixed 168-image experimental manifest
data/audit/        Category-selection and data-quality evidence
data/images/       Downloaded image files (ignored by Git)
src/vlm_gap/       Reusable data, model, and metric code
scripts/           Download, validation, and evaluation commands
tests/             Fast tests that do not download models
results/           Generated predictions and summaries (ignored by Git)
docs/              Experiment log, method notes, and literature review
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

To run Qwen, InternVL, or YOLO-World, install the optional model packages:

```bash
python -m pip install -r requirements-models.txt
```

On Apple Silicon, PyTorch will use MPS when it is available. The code also
works on CUDA and CPU.

## Validate the fixed data split

```bash
python scripts/validate_manifest.py --check-images
python -m unittest discover -s tests
```

Expected result: 168 unique records and exactly 7 records in every
category-income-quartile cell.

## Download the selected images

The repository does not track image binaries or the original 11 GB archive.
Download only the 168 selected images:

```bash
python scripts/download_images.py
```

Use `--limit 2` for a quick connection test. Existing valid files are skipped,
so interrupted downloads can be resumed.

The images remain subject to the Dollar Street dataset licence and attribution
requirements.

## Run a small experiment first

```bash
python scripts/run_evaluation.py --limit 4
```

This legacy command runs the original CLIP/BLIP experiment. The expanded
benchmark can be checked with selected models before a full run:

```bash
python scripts/run_benchmark.py --models clip blip --limit 4
python scripts/run_benchmark.py --models yolo_world --limit 4
python scripts/run_benchmark.py --models qwen internvl --limit 1
```

YOLO-World uses a predeclared confidence threshold of 0.25. Optional threshold
sensitivity checks can be run with `--yolo-confidence 0.05` or `0.50`; they
must be reported separately from the primary result.

Then run the full fixed subset. A CUDA machine or Colab runtime is recommended
for Qwen and InternVL:

```bash
python scripts/run_benchmark.py \
  --models clip blip qwen internvl yolo_world
```

Outputs are written to:

- `results/benchmark_predictions.csv`
- `results/benchmark_by_income.csv`
- `results/benchmark_by_category.csv`
- `results/income_gap_estimates.csv`
- `results/failure_cases.csv`
- `results/run_metadata.json`

The first run of each model downloads its pretrained weights. Generated files
remain ignored until the full run and manual quality checks are complete.

Divya's separately executed CLIP/BLIP and Qwen predictions must be merged only
through the strict finalizer. It checks all five model-task conditions, exact
row counts, unique image IDs, and the balanced category-quartile design before
regenerating tables and figures:

```bash
python scripts/finalize_divya_results.py
```

## Current project status

- The fixed 168-image subset has been downloaded and validated locally.
- All 21 fast repository tests pass.
- A four-image end-to-end smoke test was completed on 2026-07-29.
- The expanded benchmark code now covers five model families and three tasks.
- Divya's full CLIP/BLIP evaluation and deterministic half-sample BLIP semantic
  review are complete. The full Qwen evaluation is in progress; its one-image
  compatibility check is not presented as a research result.

See [`docs/experiment_log.md`](docs/experiment_log.md) for the tested results
and their interpretation limits. See
[`docs/literature_review.md`](docs/literature_review.md) for the baseline
background and the difference from the closest Dollar Street study.

## Launch the early demo

```bash
python app.py
```

The demo supports an uploaded image and a selectable comparison of all five
baselines. Qwen, InternVL, and YOLO-World require the optional model packages.
Aggregate charts will be added only after the full evaluation has passed its
quality checks.

## Reproducibility notes

- The manifest stores the source row index, image ID, class, income quartile,
  country, region, and accepted caption terms.
- Multi-class source rows were excluded.
- Image selection is deterministic.
- Income analysis is balanced by design.
- Regional analysis must always report sample counts and should not be treated
  as a balanced causal comparison.
