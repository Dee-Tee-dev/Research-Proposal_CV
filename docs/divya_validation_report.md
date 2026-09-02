# Divya Evidence Validation Report

## Overall assessment

**Divya's CLIP, BLIP, and Qwen analysis is ready to share with the stated
caveats.** The manifest, full predictions, calculations, figures, and both
caption-audit summaries agree. The completed Qwen benchmark, rather than the
earlier one-image compatibility check, is used in the paper.

Final validation was performed on 2026-09-03 against the fixed manifest, saved
CLIP/BLIP and Qwen predictions, regenerated bootstrap output, completed Divya
caption-review files, rendered figures, and current paper text.

## Dataset and grain checks

- Intended image grain: one row per selected Dollar Street image in the
  manifest; one row per image, model condition, and task in predictions.
- Manifest: 168 rows, 168 unique image IDs, zero exact duplicates, and no nulls
  in the required ID, filename, category, income, quartile, region, country ID,
  or country-name fields.
- Balance: 42 images in each income quartile, 28 in each object category, and
  exactly seven in all 24 category–quartile cells.
- Coverage: 44 countries; regional counts are Asia 64, Americas 44, Africa 39,
  and Europe 21.
- Combined prediction file: 840 rows—168 rows in each of CLIP classification,
  BLIP baseline captioning, BLIP prompted captioning, Qwen classification, and
  Qwen captioning. Each condition contains the same 168 manifest IDs, with no
  unexpected IDs, missing IDs, duplicate
  image/model/task rows, or missing metric values.

## Calculation spot-checks

| Claim | Independent check | Status |
|---|---|---|
| CLIP overall accuracy | 143/168 = 85.1% | Verified |
| BLIP baseline recall | 89/168 = 53.0% | Verified |
| BLIP prompted recall | 86/168 = 51.2% | Verified |
| Qwen classification accuracy | 145/168 = 86.3% | Verified |
| Qwen caption recall | 106/168 = 63.1% | Verified |
| Prompt overall change | 51.2% − 53.0% = −1.8 percentage points | Verified |
| CLIP Q4–Q1 gap | 39/42 − 30/42 = 21.4 points | Verified |
| BLIP baseline Q4–Q1 gap | 32/42 − 12/42 = 47.6 points | Verified |
| BLIP prompted Q4–Q1 gap | 33/42 − 11/42 = 52.4 points | Verified |
| Qwen classification Q4–Q1 gap | 37/42 − 36/42 = 2.4 points | Verified |
| Qwen caption Q4–Q1 gap | 34/42 − 15/42 = 45.2 points | Verified |
| Bootstrap intervals | Recomputed with 2,000 category-stratified samples and seed 2026 | Verified to floating-point precision |
| Caption semantic audit | Baseline 55 yes and 6 uncertain; prompted 52 yes and 6 uncertain | Verified |
| Qwen semantic audit | 59 yes, 11 uncertain, 15 false negatives, 1 false positive | Verified |

All quartile and category percentages in `paper/divya_sections.md` reconcile to
the raw Boolean prediction rows. The reported CLIP confusion counts also sum to
28 in every true-category row. The BLIP prompt-delta figure agrees with the
category numerators.

## Method and narrative review

- The research question, selected population, exclusions, balancing procedure,
  model checkpoints, prompts, preprocessing, metrics, bootstrap design, and
  reproducibility settings are stated.
- The paper correctly describes observed associations rather than claiming
  that income caused model errors.
- The failed prompt intervention is reported directly and is not reframed as a
  success.
- Classification accuracy and caption accepted-term recall are kept separate
  in the figures and text.
- The caption audits demonstrate that accepted-term recall is conservative: 19
  clear BLIP matches and 15 clear Qwen matches were missed by the automatic
  vocabulary. The paper reports strict semantic results and upper bounds that
  include uncertain decisions.

## Required caveats

- The balanced 168-image subset is deliberately controlled for category but is
  not population-representative.
- Region is unbalanced and may be used only as a secondary descriptive split.
- Q4–Q1 differences are associations within the selected images, not causal
  effects of income.
- Caption accepted-term recall measures mention of the target object, not full
  caption fluency, completeness, or factual accuracy.
- Divya's caption audit uses one deterministic half-sample and one reviewer; it
  must not be described as blinded or as an inter-rater reliability study.

## Final gate

`scripts/finalize_divya_results.py` passed all exact-row, unique-ID,
shared-image, and balance checks before the Qwen values were inserted. Combined
tables, confidence intervals, and figures were regenerated; both primary plots
were visually checked, and their classification and captioning axes use the
correct metric labels. `scripts/check_divya_paper.py` is the final automated
gate for Divya's submission-ready sections. Rendering the integrated paper is a
shared final-report step because Riya's assigned sections are intentionally not
modified here.
