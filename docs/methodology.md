# Expanded Benchmark Methodology

## Research question

Do pretrained vision-language and open-vocabulary detection models show
different household-object performance across the four income quartiles in the
fixed Dollar Street subset? The analysis is descriptive and comparative. It
does not treat income as the proven cause of an error and does not assume that
a gap must exist.

## Data and controls

The experiment contains 168 unique images: six household-object categories,
four income quartiles, and seven images in every category-by-quartile cell.
Every model receives the same image set. Category balance is kept fixed so that
an overall quartile score cannot be driven by one category appearing more
often. The manifest records the original image ID, income, country, region,
category, and accepted caption terms.

The images cover 44 countries. Region counts are unequal, so regional results
are secondary and always include sample counts. The subset is a controlled
course-project sample, not a representative estimate of all households or
regions.

## Tasks and baselines

| Model | Task | Output used for scoring |
|---|---|---|
| CLIP ViT-B/32 | six-way zero-shot classification | highest-scoring text label |
| BLIP base | captioning | baseline and label-free prompted captions |
| Qwen2.5-VL-3B-Instruct | classification and captioning | deterministic generated answer |
| InternVL3.5-2B | classification and captioning | deterministic generated answer |
| YOLO-World small v2 | open-vocabulary detection | detections from the same six labels |

Qwen and InternVL receive the same classification instruction, which requires
exactly one of the six labels, and the same short-caption instruction. Neither
prompt includes the correct object, income, country, or region. YOLO-World is
given only the six category names. No model is fine-tuned on the selected
images.

## Metrics

- Classification: top-1 accuracy. CLIP correct-label rank is retained as a
  secondary diagnostic.
- Captioning: accepted-term recall, followed by blinded manual checking of
  automatic false positives and false negatives. Terms are defined before the
  full experiment and stored in the manifest.
- Detection: image-level hit rate, meaning that at least one detection has the
  correct study label. Confidence and predicted boxes are saved for failure
  analysis. The primary threshold is fixed at 0.25 before the full run; 0.05
  and 0.50 may be reported as clearly labelled sensitivity checks.

The detection result is not reported as mAP or IoU because the source split has
no ground-truth boxes. Caption recall is not presented as a complete measure of
caption quality; it tests whether the relevant object was named.

## Analysis plan

For each model and task, the report will show the score and sample count for
each income quartile and category. The main gap estimate is Q4 minus Q1, with a
95% category-stratified bootstrap interval using a fixed random seed. The
interval describes uncertainty in this selected sample; it does not make the
subset population-representative.

Failure analysis will include examples from each quartile and separate common
cases such as visually small objects, clutter, local object designs, ambiguous
labels, caption synonyms missed by the automatic metric, and detector
localization failures. Examples will be selected by a written rule, not only by
how surprising they look.

## Reproducibility and reporting limits

The split, prompts, model checkpoint names, thresholds, raw outputs, and random
seed are stored in the repository. Smoke tests are marked separately from the
full experiment. The final report will distinguish observed associations from
causal claims and will report unsuccessful runs or model-specific limitations
rather than silently excluding them.
