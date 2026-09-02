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
- [x] Records Qwen runtime metadata for the completed 168-image run.

## Results and Quantitative Analysis (Divya contribution)

- [x] Full CLIP and BLIP row counts validated.
- [x] Overall, income-quartile, and category results reported.
- [x] Q4–Q1 gaps include 95% category-stratified bootstrap intervals.
- [x] Prompt intervention is reported as unsuccessful rather than reframed.
- [x] Figures separate classification accuracy and caption recall.
- [x] Figure captions include task metric and sample count.
- [x] Deterministic half-sample caption audit is complete and reproducible.
- [x] Metric false negatives and ambiguous source-label cases are quantified.
- [x] Full Qwen classification and caption results validated for all 168 images.
- [x] Combined 840-row Divya tables and figures regenerated with Qwen.
- [x] Matched 84-image Qwen semantic audit completed and summarized.

## Writing Quality and Organization

- [x] Main-paper sections follow the required Introduction → Method → Results
      order, with Dataset inside Method.
- [x] Acronyms and tasks are introduced before detailed metrics.
- [x] Results distinguish evidence from interpretation.
- [x] Limitations prevent causal or population-level overclaiming.
- [x] Divya's main-paper contribution was compressed from 1,754 words
      and reduced to one primary figure to leave space within the two-page main
      paper for Riya's assigned analysis.
- [ ] Render the integrated conference-style PDF and confirm the two-page main
      paper plus one-page related-work appendix limit.

Riya's related-work appendix, InternVL/YOLO-World results, qualitative failure
analysis, demo, and assigned writing are intentionally outside this audit.
