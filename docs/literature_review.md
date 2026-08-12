# Literature Review and Positioning

## Socioeconomic evaluation of vision-language models

Nwatu, Ignat, and Mihalcea (2023) evaluated CLIP on Dollar Street and reported
performance differences associated with household income. Their work is the
closest prior study because it uses the same source dataset and treats income as
an important evaluation variable. Our project does not try to reproduce every
task in that paper. It uses a smaller, fixed, category-balanced subset and asks
whether the pattern is consistent across different model families and tasks.
This is important because a result observed for CLIP similarity scores may not
hold for instruction-following VLMs or an open-vocabulary detector.

## Baseline model families

CLIP learns aligned image and text representations through contrastive
pretraining and supports zero-shot classification by comparing an image with
text prompts (Radford et al., 2021). It is the simplest classification baseline
and connects the study directly to the closest prior work.

BLIP combines vision-language understanding and generation and improves its
training data through synthetic caption generation and filtering (Li et al.,
2022). We use its pretrained captioner without fine-tuning. The comparison
includes an unprompted caption and one label-free object-focused prompt, which
tests whether a small prompting change improves object mention recall.

Qwen2.5-VL and InternVL3.5 represent newer general-purpose multimodal models.
Unlike CLIP, they can follow a natural-language instruction and generate a
response. We use compact public checkpoints, Qwen2.5-VL-3B-Instruct and
InternVL3.5-2B, so both can be tested with the same forced-choice classification
instruction and the same short caption instruction. Decoding is deterministic,
and the models receive no income, country, region, or correct-label metadata.

YOLO-World is an open-vocabulary detector that accepts user-defined category
names (Cheng et al., 2024). It therefore provides a model family that is
different from both contrastive classification and caption generation. Dollar
Street supplies image-level object labels but not bounding-box annotations, so
we report image-level detection hit rate and qualitative boxes. We do not
report mAP or IoU, because those metrics would require ground-truth boxes.

## Difference from prior studies

The main contribution is a controlled comparison rather than a new model. All
baselines use the same 168 images, the same six object categories, and the same
four income groups. The subset contains seven images in each
category-by-income cell, which prevents category frequency from explaining an
income-level difference. Results will be reported by income group and category,
with sample counts, failure examples, and limits on interpretation. Region is a
secondary analysis because the subset is not region-balanced. The study tests
for observed performance differences; it does not assume in advance that every
model will show the same direction or size of gap.

## References

- Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., & Shan, Y. (2024).
  *YOLO-World: Real-Time Open-Vocabulary Object Detection*. CVPR, 16901–16911.
- Li, J., Li, D., Xiong, C., & Hoi, S. (2022). *BLIP: Bootstrapping
  Language-Image Pre-training for Unified Vision-Language Understanding and
  Generation*. ICML, 12888–12900.
- Nwatu, J., Ignat, O., & Mihalcea, R. (2023). *Bridging the Digital Divide:
  Performance Variation across Socio-Economic Factors in Vision-Language
  Models*. EMNLP, 10686–10702.
- Qwen Team. (2025). *Qwen2.5-VL Technical Report*. arXiv:2502.13923.
- Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural
  Language Supervision*. ICML, 8748–8763.
- Wang, W., et al. (2025). *InternVL3.5: Advancing Open-Source Multimodal Models
  in Versatility, Reasoning, and Efficiency*. arXiv:2508.18265.
