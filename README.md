# Socioeconomic and Regional Performance Gaps in Vision-Language Models

This repository contains the code, proposal, and reproducible data split for a
Computer Vision course project. The study compares CLIP zero-shot
classification and BLIP image captioning on household-object images from Dollar
Street.

The main comparison is across four income quartiles. Regional results are
secondary because the available images are not balanced equally across regions.

## Study design

- Models: `openai/clip-vit-base-patch32` and
  `Salesforce/blip-image-captioning-base`
- Categories: roof, light source, stove, trash container, switch, footwear
- Sample: 168 unique images
- Balance: 7 images per category in each of 4 income quartiles
- CLIP metrics: top-1 accuracy and correct-label rank
- BLIP metrics: accepted-term recall and a later blinded manual review
- BLIP intervention: the label-free prefix
  `the main household object in this image is`

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
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
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

Then run the full fixed subset:

```bash
python scripts/run_evaluation.py
```

Outputs are written to:

- `results/predictions.csv`
- `results/summary_by_income.csv`
- `results/summary_by_category.csv`

The first model run downloads pretrained weights from Hugging Face.

## Launch the early demo

```bash
python app.py
```

The initial demo supports an uploaded image and displays the CLIP prediction,
an unprompted BLIP caption, and the prompted BLIP caption side by side.
Aggregate charts will be added after the full evaluation has been run.

## Reproducibility notes

- The manifest stores the source row index, image ID, class, income quartile,
  country, region, and accepted caption terms.
- Multi-class source rows were excluded.
- Image selection is deterministic.
- Income analysis is balanced by design.
- Regional analysis must always report sample counts and should not be treated
  as a balanced causal comparison.
