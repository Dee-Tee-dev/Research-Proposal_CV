# Divya Experiment Log

## 2026-08-25 — Full CLIP and BLIP evaluation

### Scope

This run covers only Divya's assigned CLIP and BLIP baselines. It does not
include InternVL, YOLO-World, the Gradio demo, or Riya's qualitative analysis.

### Configuration

- Images: 168 unique files from the fixed balanced manifest
- Device: CPU
- CLIP checkpoint: `openai/clip-vit-base-patch32`
- BLIP checkpoint: `Salesforce/blip-image-captioning-base`
- Bootstrap: 2,000 category-stratified resamples, seed 2026
- Completed rows: 168 CLIP classifications, 168 BLIP baseline captions, and
  168 BLIP prompted captions
- Duplicate image/model/task rows: 0

### Verified results

| Model condition | Correct/matched | Score |
|---|---:|---:|
| CLIP classification | 143/168 | 85.1% |
| BLIP baseline caption recall | 89/168 | 53.0% |
| BLIP prompted caption recall | 86/168 | 51.2% |

| Model condition | Q1 | Q2 | Q3 | Q4 | Q4 minus Q1 (95% CI) |
|---|---:|---:|---:|---:|---:|
| CLIP classification | 71.4% | 83.3% | 92.9% | 92.9% | 21.4 pp (7.1, 35.7) |
| BLIP baseline caption recall | 28.6% | 52.4% | 54.8% | 76.2% | 47.6 pp (31.0, 64.3) |
| BLIP prompted caption recall | 26.2% | 54.8% | 45.2% | 78.6% | 52.4 pp (38.1, 66.7) |

### Interpretation boundary

The results show a Q4-over-Q1 difference for these model conditions in the
selected sample. They do not establish income as the cause, do not make the
subset representative, and do not yet include the manual caption-quality
review. Qwen is also pending, so the Divya model comparison is not complete.

The prompt did not improve BLIP's overall accepted-term recall. It reduced the
score by 1.8 percentage points, with effects varying by object category. This
result should be reported as observed rather than rewritten as a successful
intervention.
