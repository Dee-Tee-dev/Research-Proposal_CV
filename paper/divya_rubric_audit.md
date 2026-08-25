# Divya Section Rubric Audit

## Introduction and Problem Motivation (6 points)

- [x] Defines the household-object recognition and captioning problem.
- [x] Explains the computer-vision challenge: cross-household visual variation.
- [x] States one explicit research question.
- [x] Explains why a controlled income-quartile evaluation is useful.
- [x] Avoids assuming a fairness gap before observing results.

## Methodology and Technical Correctness (Divya contribution)

- [x] Identifies the source split, eligible-record filters, and quartile cutoffs.
- [x] Reports 168 images, 44 countries, six categories, four quartiles, and
      seven images per category–quartile cell.
- [x] Records exact CLIP, BLIP, and Qwen checkpoints.
- [x] Records CLIP and BLIP preprocessing and Qwen resolution limits.
- [x] Provides exact prompts and deterministic generation settings.
- [x] States that no training or fine-tuning is performed.
- [x] Defines classification accuracy, caption recall, semantic audit, and
      Q4–Q1 bootstrap intervals.
- [x] Records software versions, random seed, and output validation rules.
- [ ] Add Qwen runtime metadata after the full run finishes.

## Results and Quantitative Analysis (Divya contribution)

- [x] Full CLIP and BLIP row counts validated.
- [x] Overall, income-quartile, and category results reported.
- [x] Q4–Q1 gaps include 95% category-stratified bootstrap intervals.
- [x] Prompt intervention is reported as unsuccessful rather than reframed.
- [x] Figures separate classification accuracy and caption recall.
- [x] Figure captions include task metric and sample count.
- [x] Deterministic half-sample caption audit is complete and reproducible.
- [x] Metric false negatives and ambiguous source-label cases are quantified.
- [ ] Add and validate full Qwen results.
- [ ] Regenerate combined Divya tables and figures with Qwen.

## Writing Quality and Organization

- [x] Sections follow Introduction → Dataset → Methodology → Results.
- [x] Acronyms and tasks are introduced before detailed metrics.
- [x] Results distinguish evidence from interpretation.
- [x] Limitations prevent causal or population-level overclaiming.
- [ ] Perform final compression and page-limit check after group integration.

Riya's related-work appendix, InternVL/YOLO-World results, qualitative failure
analysis, demo, and assigned writing are intentionally outside this audit.
