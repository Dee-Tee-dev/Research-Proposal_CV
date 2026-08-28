# Divya Evidence Validation Report

## Overall assessment

**Completed CLIP/BLIP analysis: ready to share with stated caveats.** The
manifest, predictions, calculations, figures, and caption-audit counts agree.
**Final Divya section: not yet submission-ready** because the full 168-image
Qwen run and its final integration are still pending. The one-image Qwen check
is correctly excluded from the research results.

Validation was performed on 2026-08-28 against the fixed manifest, the saved
2026-08-25 CLIP/BLIP predictions, the saved bootstrap output, the completed
Divya caption-review file, and the current paper text.

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
- Prediction file: 504 rows—168 CLIP classifications, 168 BLIP baseline
  captions, and 168 BLIP prompted captions. Each condition contains the same
  168 manifest IDs, with no unexpected IDs, missing IDs, duplicate
  image/model/task rows, or missing metric values.

## Calculation spot-checks

| Claim | Independent check | Status |
|---|---|---|
| CLIP overall accuracy | 143/168 = 85.1% | Verified |
| BLIP baseline recall | 89/168 = 53.0% | Verified |
| BLIP prompted recall | 86/168 = 51.2% | Verified |
| Prompt overall change | 51.2% − 53.0% = −1.8 percentage points | Verified |
| CLIP Q4–Q1 gap | 39/42 − 30/42 = 21.4 points | Verified |
| BLIP baseline Q4–Q1 gap | 32/42 − 12/42 = 47.6 points | Verified |
| BLIP prompted Q4–Q1 gap | 33/42 − 11/42 = 52.4 points | Verified |
| Bootstrap intervals | Recomputed with 2,000 category-stratified samples and seed 2026 | Verified to floating-point precision |
| Caption semantic audit | Baseline 55 yes and 6 uncertain; prompted 52 yes and 6 uncertain | Verified |

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
- The caption audit demonstrates that accepted-term recall is conservative: 19
  clear semantic matches were missed by the automatic vocabulary. The paper
  reports both the strict semantic result and an upper bound including uncertain
  decisions.

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

## Submission blocker and final gate

The only current blocker within Divya's assigned work is the full Qwen result.
After it completes, `scripts/finalize_divya_results.py` must pass all exact-row,
unique-ID, shared-image, and balance checks before Qwen values are inserted in
the paper. The combined tables, confidence intervals, and figures must then be
regenerated and visually checked. No smoke-test number may be substituted if
the full run fails or remains incomplete.
