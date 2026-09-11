# Final Short Paper Rubric Audit

## Evidence ownership and integration

| Area | Divya contribution | Riya contribution | Combined treatment |
|---|---|---|---|
| Models | CLIP, BLIP, Qwen | InternVL, YOLO-World | Eight complete model-task conditions on the same 168 images |
| Quantitative analysis | Tables, bootstrap gaps, CLIP/BLIP/Qwen checks | InternVL/YOLO outputs and combined evidence | One corrected 1,344-row benchmark with shared quartile and category analysis |
| Caption review | Deterministic 84-image Qwen audit and earlier BLIP checks | Full two-rater BLIP review and adjudication | Automatic caption recall is interpreted beside semantic review |
| Qualitative evidence | Model-specific failure interpretation | Demo evidence and detector examples | Failure cases are described in the Results section and preserved in the repository |
| Shared deliverable | Dataset, method, and quantitative writing | Literature/model background, limitations, demo, and new baselines | Two-page main paper plus one-page related-work appendix |

Neither personal branch was edited during integration. The combined paper was built on `codex/final-combined` from Riya's final evidence commit, which already contains Divya's completed commit.

## Rubric coverage

### Introduction and Problem Motivation 6 points

- [x] Defines the household-object recognition problem and why appearance variation is a computer-vision challenge.
- [x] States one explicit research question before experimental detail.
- [x] Explains the link to prior Dollar Street CLIP work.
- [x] States the significance of comparing multiple task and model families.
- [x] Avoids a predetermined fairness conclusion and causal wording.

### Methodology and Technical Correctness 12 points

- [x] Documents the source split, exclusion rule, quartile thresholds, 168-image subset, 44 countries, six categories, and 24 balanced cells.
- [x] Identifies all five model families, checkpoints or variants, prompts, preprocessing, deterministic decoding, and the no-fine-tuning design.
- [x] Defines top-1 accuracy, accepted-term recall, human semantic review, detection hit rate, and the category-stratified bootstrap.
- [x] Explains why YOLO mAP and IoU are not reported without ground-truth boxes.
- [x] Preserves the exact failed BLIP prompt intervention and the corrected proposal-prompt CLIP rerun.
- [x] Provides scripts, raw outputs, run metadata, a fixed seed, and a combined automated gate.

### Results and Analysis 12 points

- [x] Reports overall and quartile scores for all eight model-task conditions.
- [x] Reports Q4-Q1 intervals and distinguishes intervals that include zero.
- [x] Includes a clear table, a two-panel figure, category analysis, and concrete failure cases.
- [x] Compares automatic caption recall with the full two-rater BLIP review.
- [x] Reports agreement, Cohen's kappa, hallucination rates, and the different prompt conclusion under human review.
- [x] Treats YOLO-World as a weak image-level baseline without unsupported localization claims.

### Related Work and Literature Review 5 points

- [x] Covers Dollar Street, the closest socioeconomic CLIP study, and all five baseline families.
- [x] Explains how the dataset, task formulation, multi-model comparison, and manual review differ from prior work.
- [x] Includes a comparison table and seven consistently formatted references.

### Writing Quality and Organization 5 points

- [x] Follows Introduction, Methodology, Results, Discussion, then Related Work in the appendix.
- [x] Uses task-specific metric labels and sample counts in captions.
- [x] Uses two main-paper pages plus one related-work appendix page, with page numbers.
- [x] Word and PDF renders were visually inspected for clipping, overlap, table wrapping, and figure readability.
- [x] Names both authors and keeps the main limitations beside the affected conclusions.

## Validation status

**Ready within reviewed scope.** The 1,344 saved prediction rows contain 168 unique images in each of eight model-task conditions, no duplicate image-model-task keys, and no missing metrics. Every condition retains seven images in each of the 24 category-quartile cells. The 336-row adjudicated BLIP review is complete. All 42 repository unit tests and the combined paper gate pass.

The image binaries are intentionally ignored by Git and are not present in this integration checkout, so the final check could not reopen all 168 source images. This does not invalidate the saved full-run predictions, metadata, human-review sheets, or demo screenshots, but anyone rerunning the benchmark must download the images using the repository script first.
