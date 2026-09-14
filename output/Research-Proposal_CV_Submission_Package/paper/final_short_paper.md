# Income-Related Performance Gaps in Vision-Language Models

**Diya Tiwari and Riya Katikar**

## Abstract

We study whether pretrained vision-language models perform differently on pictures of household objects from four income groups in Dollar Street. The benchmark contains 168 images from six categories, with seven images in every category-income cell. We compare CLIP, BLIP, Qwen2.5-VL, InternVL3.5, and YOLO-World without fine-tuning. CLIP classification and every automatic caption measure scored higher on the highest-income quartile than the lowest, while Qwen and InternVL classification changed little. A two-rater review also showed that vocabulary matching understated BLIP caption quality. The measured gap depends on the model, task, and evaluation metric.

## 1 Introduction

The same household object can look very different across homes. For example, a stove may be built into a kitchen, placed on a table, or use solid fuel. Models trained largely on web image-text pairs may be more familiar with some of these forms than others. Earlier Dollar Street work found income-related differences in CLIP alignment and retrieval [4]. We ask: **How does performance on six household-object categories vary across four income quartiles when each category is equally represented?** Income is used to group images; it is not treated as the cause of an error.

## 2 Method

### Dataset

We used the public 1,600-row Dollar Street test split [6]. Income records monthly consumption per adult equivalent in PPP-adjusted US dollars. After removing 45 multi-label rows, we formed quartiles from 1,555 eligible records. The evaluation subset contains 168 images from 44 countries and six categories: roof, light source, stove, trash container, switch, and footwear. Each of the 24 category-quartile cells has seven images. This controls category frequency, although the subset is not representative of the whole population and regional counts remain unequal.

### Models and tasks

CLIP ViT-B/32 [5] performed six-way zero-shot classification with `a photo of a {label}`. BLIP-base [3] generated an unprompted caption and a caption beginning with the label-free prefix `the main household object in this image is`. Qwen2.5-VL-3B-Instruct [1] and InternVL3.5-2B [7] used the same forced-choice classification instruction and the same short, location-neutral caption instruction. YOLO-World-small-v2 [2] received the six labels with a confidence threshold of 0.25. Since Dollar Street has no bounding boxes, YOLO-World is reported as an image-level correct-class detection rate; mAP and IoU cannot be calculated.

CLIP used its 224-pixel checkpoint processor and BLIP its 384-pixel processor. Qwen used a 256-token visual budget, InternVL used 448 by 448 RGB inputs with ImageNet normalization, and YOLO-World used the Ultralytics prediction pipeline. Generation was deterministic. The manifest, image identifiers, prompts, outputs, and run settings are included in the repository.

### Evaluation

Classification uses top-1 accuracy. Caption recall checks for a predefined category term or synonym with whole-term matching. Two raters independently reviewed all 336 BLIP captions for object correctness, non-informativeness, and hallucination, and then resolved disagreements. We report raw agreement and Cohen's kappa. A separate one-rater check used a fixed 84-image sample for Qwen captions. The main gap is Q4 minus Q1, with a 95% category-stratified bootstrap interval from 2,000 resamples using seed 2026. We verified all 1,344 prediction rows for coverage, duplicate keys, and missing metrics.

## 3 Results and analysis

| Model and task | Q1 | Q2 | Q3 | Q4 | Overall |
|---|---:|---:|---:|---:|---:|
| CLIP classification | 76.2 | 81.0 | 95.2 | 92.9 | **86.3** |
| Qwen classification | 85.7 | 88.1 | 83.3 | 88.1 | **86.3** |
| InternVL classification | 88.1 | 92.9 | 78.6 | 88.1 | **86.9** |
| YOLO-World detection | 11.9 | 14.3 | 14.3 | 21.4 | **15.5** |
| BLIP baseline caption recall | 28.6 | 52.4 | 54.8 | 76.2 | **53.0** |
| BLIP prompted caption recall | 26.2 | 54.8 | 45.2 | 78.6 | **51.2** |
| Qwen caption recall | 35.7 | 64.3 | 71.4 | 81.0 | **63.1** |
| InternVL caption recall | 42.9 | 59.5 | 66.7 | 64.3 | **58.3** |

*Table 1. Scores (%) by income quartile; each quartile contains 42 images.*

All three classifiers had similar overall accuracy, but their quartile results differed. CLIP's Q4-Q1 gap was 16.7 percentage points (95% CI 2.4 to 28.6). The intervals for Qwen's 2.4-point gap (-7.1 to 14.3) and InternVL's zero-point gap (-9.5 to 9.5) included zero. The CLIP pattern did not appear in both instruction-following classifiers.

Automatic caption recall was higher in Q4 than Q1 for every captioning condition. The gaps were 47.6 points for unprompted BLIP, 52.4 for prompted BLIP, 45.2 for Qwen, and 21.4 for InternVL. Manual review gave a different view of BLIP. Object correctness was 80.4% without the prompt and 83.9% with it, compared with automatic recall of 53.0% and 51.2%. The prompt reduced automatic recall by 1.8 points, but improved human-rated correctness by 3.6 points and reduced hallucinations from 16.7% to 12.5%. Object-correct agreement was 89.3% (kappa 0.65), and hallucination agreement was 94.0% (kappa 0.77). In the separate Qwen sample, automatic recall was 57.1% and clear semantic recall was 70.2%; 11 of 84 captions remained uncertain. The term list missed reasonable descriptions such as *bucket* or *basket* for a trash container.

The category results also differed by model. CLIP was weakest on light sources (60.7%), while Qwen and InternVL were weakest on roofs (39.3% and 50.0%). YOLO-World found 57.1% of footwear and 25.0% of stoves, but none of the roofs, light sources, or switches. In one failure case, CLIP labelled a small red fuel can as a stove while BLIP described the visible can. In another, a roof occupied little of the frame and was classified as a light source. These examples explain why we report category results and manually review captions.

## 4 Discussion and limitations

The results do not support one common income pattern across all models. CLIP and the automatic caption measures favored Q4, whereas Qwen and InternVL classification were nearly flat. The BLIP comparison also depended on the metric: term recall missed valid wording that the raters accepted.

The study covers 168 selected images and six categories. Balancing removes category-frequency differences but does not control country, region, image quality, scene complexity, or variation within a category. Quartiles are defined within the available test split, and the confidence intervals describe this benchmark only. Models never received income information, so the results show associations in performance and do not establish that income caused the errors. The Gradio interface demonstrates the five models on uploaded images; it is not a second evaluation dataset.

## 5 Related work

Dollar Street provides household-object images with country, region, and income metadata [6]. Nwatu et al. evaluated CLIP alignment and prompt-based retrieval across income groups [4]. We use a category-balanced subset, test six-way recognition and captioning, and compare five model families on the same images. CLIP supplies a direct comparison with earlier work [5], while BLIP adds caption generation [3]. Qwen2.5-VL and InternVL3.5 add compact instruction-following VLMs [1,7], and YOLO-World adds open-vocabulary detection [2]. Our experiment found that a gap observed with one model or automatic metric did not always carry over to another.

## References

1. Bai, S., et al. (2025). Qwen2.5-VL Technical Report. arXiv:2502.13923.
2. Cheng, T., et al. (2024). YOLO-World: Real-Time Open-Vocabulary Object Detection. CVPR, 16901-16911.
3. Li, J., Li, D., Xiong, C., and Hoi, S. (2022). BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. ICML, 12888-12900.
4. Nwatu, J., Ignat, O., and Mihalcea, R. (2023). Bridging the Digital Divide: Performance Variation across Socio-Economic Factors in Vision-Language Models. EMNLP, 10686-10702.
5. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML, 8748-8763.
6. Rojas, W. A. G., et al. (2022). The Dollar Street Dataset: Images Representing the Geographic and Socioeconomic Diversity of the World. NeurIPS Datasets and Benchmarks.
7. Wang, W., et al. (2025). InternVL3.5: Advancing Open-Source Multimodal Models. arXiv:2508.18265.
