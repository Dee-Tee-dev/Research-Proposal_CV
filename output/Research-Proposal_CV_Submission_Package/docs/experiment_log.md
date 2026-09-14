# Experiment Log

## 2026-07-29 — Four-image pipeline smoke test

### Purpose

Confirm that the local image files, CLIP classification, BLIP caption
generation, metrics, and CSV output pipeline work together end to end.

### Command

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
python scripts/run_evaluation.py --limit 4 --device cpu
```

Anonymous Hugging Face access was used because the machine's cached token was
invalid. Both selected model repositories are public.

### Models

- `openai/clip-vit-base-patch32`
- `Salesforce/blip-image-captioning-base`

### Smoke-test result

| Measure | Result |
|---|---:|
| Images processed | 4 |
| CLIP top-1 accuracy | 3/4 (75%) |
| BLIP baseline accepted-term recall | 1/4 (25%) |
| BLIP prompted accepted-term recall | 1/4 (25%) |

The four rows selected by `--limit 4` were all roof images from income quartile
Q1. This run verifies the software pipeline only. It is not a balanced
experiment and must not be used to claim a socioeconomic performance gap or a
prompting improvement.

Generated CSV files are stored locally under `results/` and are intentionally
ignored by Git until the full evaluation and quality checks are complete.

