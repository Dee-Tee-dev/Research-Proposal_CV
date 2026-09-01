# Divya Experiment Log

## 2026-08-25 — Full CLIP and BLIP evaluation

### Scope

This run covers only Divya's assigned CLIP and BLIP baselines. It does not
include InternVL, YOLO-World, the Gradio demo, or Riya's qualitative analysis.

### Configuration

- Images: 168 unique files from the fixed balanced manifest
- Device: CPU
- Python 3.13.5; PyTorch 2.12.1; Transformers 5.13.0
- NumPy 2.3.3; pandas 2.3.3; Pillow 11.3.0
- CLIP checkpoint: `openai/clip-vit-base-patch32`
- BLIP checkpoint: `Salesforce/blip-image-captioning-base`
- CLIP preprocessing: resize shortest edge to 224, centre crop to 224 × 224,
  checkpoint channel normalisation
- BLIP preprocessing: resize to 384 × 384, checkpoint channel normalisation
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

## 2026-08-25 — Divya caption semantic audit

The deterministic Divya half contains 84 images and two BLIP captions per
image. All 168 captions were checked in context, and automatic misses were
visually compared with their source images.

| Condition | Automatic matches | Clear semantic matches | Uncertain | Semantic upper bound |
|---|---:|---:|---:|---:|
| BLIP baseline | 45/84 (53.6%) | 55/84 (65.5%) | 6 | 61/84 (72.6%) |
| BLIP prompted | 43/84 (51.2%) | 52/84 (61.9%) | 6 | 58/84 (69.0%) |

The audit found 19 clear automatic false negatives in total and no clear false
positives. The prompted condition remained below the baseline under the strict
semantic decision and the upper-bound treatment of uncertain cases. The audit
therefore supports the main interpretation while demonstrating that accepted-
term recall is a conservative measure of object mention quality.

## 2026-08-25 — Qwen compatibility check

`Qwen/Qwen2.5-VL-3B-Instruct` was loaded from its public checkpoint and tested
on one manifest image on CPU using the final classification and captioning
prompts. The run completed without an inference error and produced exactly one
classification row and one caption row. For the test roof image, the forced-
choice response was `roof`, and the caption described a corrugated metal roof;
both automatic scores were correct. This single-image check is a pipeline
validation only and is not included in the reported benchmark results.

The test took 712 seconds for both tasks. PyTorch was built with MPS support,
but MPS was not available in this execution environment, so the full run must
use CPU. The measured rate gives an approximate full-run duration of 33 hours.

## 2026-09-01 — Interrupted Qwen full run and recovery design

The first full Qwen attempt used the general benchmark runner and reached 53 of
168 images before the tracked Mac process disappeared without a normal exit.
There was no model exception in the captured output. Because that runner kept
predictions in memory until the end, no partial prediction rows were used or
reported.

The replacement Qwen runner writes both tasks atomically after every completed
image and validates the saved model, task, image, manifest, prompt, and
resolution configuration before resuming. To reduce CPU runtime consistently,
Qwen preprocessing now uses the checkpoint's documented 256-visual-token
setting (200,704 pixels) for every image. A changed run configuration cannot be
mixed with an existing checkpoint. Full results remain pending until all 168
images and 336 Qwen prediction rows pass the final integration checks.

### Recovery-run trial

The final resumable configuration was tested end to end on one image. It wrote
one classification row and one caption row, then produced the expected summary,
failure, metadata, and checkpoint files. The checkpoint and final prediction
files were identical, the image/task grain had no duplicate, and both outputs
were scored correctly. A second invocation detected the completed image and
finished without loading the model or repeating inference. The two-task trial
took 314.89 seconds on CPU, reducing the estimated uninterrupted full-run time
to about 15 hours.
