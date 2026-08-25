# Divya's Short-Paper Sections

> Scope: Introduction, Dataset, Methodology, and Quantitative Results for CLIP,
> BLIP, and Qwen. Bracketed result fields must only be replaced after the full
> 168-image experiment has completed and been checked.

## 1. Introduction and Problem Motivation

Vision-language models are often evaluated on standard benchmark images, but
the same household object can look very different across homes. A stove may be
a modern built-in appliance, a portable gas burner, or a simple solid-fuel
setup. Similar visual variation occurs for roofs, light sources, switches,
footwear, and waste containers. Models trained mainly on common web images may
therefore recognise familiar versions of an object more reliably than versions
that appear less often in their training data.

This project examines whether such differences are visible across household
income groups in Dollar Street. The main research question is: **How does the
performance of pretrained vision-language models on household-object
recognition and captioning vary across four income quartiles when object
category frequency is held constant?** We treat this as an empirical question;
the experiment does not assume beforehand that lower-income images must produce
worse results.

We construct a fixed, balanced subset and compare models representing different
vision-language approaches. Divya's quantitative evaluation covers CLIP
zero-shot classification, BLIP captioning with and without a label-free prompt,
and Qwen2.5-VL forced-choice classification and captioning. The broader group
benchmark also contains baselines assigned to the other project member. Our
contribution is not a newly trained model. It is a controlled and reproducible
comparison showing whether any observed income-related pattern is consistent
across tasks and model designs.

## 2. Dataset

We use images and metadata from the Dollar Street test split. The fixed study
subset contains 168 unique images from 44 countries. It covers six household
object categories: roof, light source, stove, trash container, switch, and
footwear. Income values are divided into four quartiles, Q1 to Q4. Each
quartile contains 42 images, and every category–quartile combination contains
exactly seven images. Each category therefore contributes 28 images overall.

The balanced design prevents one income quartile from receiving a higher score
simply because it contains more examples of an easier category. The manifest
stores the original image identifier, source row, object label, income,
quartile, country, region, and accepted caption terms. Multi-class source rows
were excluded to reduce label ambiguity. All 168 selected files were checked
for uniqueness and readability before evaluation.

The subset should not be treated as representative of the world population.
Although it spans 44 countries, regional counts are unequal: 64 images are
from Asia, 44 from the Americas, 39 from Africa, and 21 from Europe. For this
reason, income quartile is the primary comparison and region is only a
secondary descriptive variable.

## 3. Methodology

### 3.1 Models and tasks

CLIP (`openai/clip-vit-base-patch32`) is used for six-way zero-shot
classification. Each image is compared with the six text prompts using the
template `a photo of a household {label}`. The label with the highest
probability is the top-1 prediction, and the rank of the correct label is also
saved.

BLIP (`Salesforce/blip-image-captioning-base`) generates two captions per
image. The baseline caption is unprompted. The second uses the prefix `the main
household object in this image is`. This prompt does not reveal the correct
category, income, country, or region. Comparing the two outputs tests whether a
small, training-free prompt helps the model mention the main object.

Qwen2.5-VL-3B-Instruct is evaluated on both classification and captioning. For
classification, it receives the image and the same fixed list of six candidate
labels and must return one label only. For captioning, it receives a short
instruction to describe the main household object without guessing location or
income. Greedy decoding is used for reproducibility. No model is trained or
fine-tuned on the study images.

### 3.2 Evaluation

Classification performance is measured using top-1 accuracy. CLIP's
correct-label rank is retained as a secondary diagnostic. Captioning is scored
using accepted-term recall: a caption is counted as a match when it contains a
predefined term or synonym associated with the study category. For example,
`lantern` can count for light source. Whole-term matching is used to avoid
partial-word errors. Because this automatic measure does not capture full
caption quality, half of the caption outputs are assigned to Divya for manual
review and the other half to the second reviewer.

Scores are calculated for every model and task by income quartile and object
category. The main gap is Q4 minus Q1. A category-stratified bootstrap with
2,000 resamples and random seed 2026 is used to obtain a 95% interval for this
gap. The interval describes uncertainty within the selected sample; it does
not make the sample population-representative. Raw outputs, prompts, checkpoint
names, predictions, and run metadata are saved for reproducibility.

## 4. Quantitative Results

> **Do not submit this section with placeholders.** Populate it from
> `results/divya/analysis/` after validating the full run.

Table 1 compares the primary score of each Divya baseline across all 168
images. CLIP achieved **[CLIP overall accuracy]** top-1 accuracy. Qwen achieved
**[Qwen overall accuracy]** on the same classification task. For captioning,
BLIP's accepted-term recall changed from **[BLIP baseline recall]** without the
prompt to **[BLIP prompted recall]** with the label-free prompt, while Qwen
reached **[Qwen caption recall]**.

Across income quartiles, **[state the observed direction without causal
language]**. The Q4–Q1 difference was **[gap]** for CLIP and **[gap]** for Qwen
classification. Their 95% bootstrap intervals were **[interval]** and
**[interval]**, respectively. An interval containing zero is reported as an
uncertain difference in this sample, not as proof that the groups perform
identically.

Category-level results show that **[best-supported category finding]**. The
largest variation occurred for **[category]**, whereas **[category]** was more
consistent across quartiles. These results are important because an aggregate
income score can hide category-specific behaviour even in a balanced design.
Figure 1 shows scores by income quartile, and Figure 2 presents the category
breakdown.

The automatic caption metric was checked against Divya's assigned manual-review
half. **[number]** automatic matches were judged incorrect, and **[number]**
automatic misses still described the correct object using wording absent from
the accepted-term list. These cases are considered when interpreting caption
recall rather than silently changing the term list after seeing the results.

### Quantitative-results checklist

- [ ] Confirm all 168 images were evaluated by CLIP, BLIP, and Qwen.
- [ ] Confirm expected row counts: CLIP 168, Qwen classification 168, Qwen
      captioning 168, BLIP baseline 168, BLIP prompted 168.
- [ ] Fill every bracketed value from saved CSV files.
- [ ] Refer to every included table and figure in the text.
- [ ] Report sample counts with quartile and category scores.
- [ ] Describe associations, not causes or universal fairness conclusions.
- [ ] Record the completed manual-review counts.
