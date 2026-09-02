# Divya Quantitative Results

Generated only after the complete five-condition validation passed.

| Model and task | Q1 | Q2 | Q3 | Q4 | Overall | Q4–Q1 gap (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| CLIP classification accuracy | 71.4% | 83.3% | 92.9% | 92.9% | 143/168 (85.1%) | 21.4 pp [7.1, 33.4] |
| BLIP baseline caption recall | 28.6% | 52.4% | 54.8% | 76.2% | 89/168 (53.0%) | 47.6 pp [31.0, 64.3] |
| BLIP prompted caption recall | 26.2% | 54.8% | 45.2% | 78.6% | 86/168 (51.2%) | 52.4 pp [38.1, 66.7] |
| Qwen classification accuracy | 85.7% | 88.1% | 83.3% | 88.1% | 145/168 (86.3%) | 2.4 pp [-7.1, 14.3] |
| Qwen caption recall | 35.7% | 64.3% | 71.4% | 81.0% | 106/168 (63.1%) | 45.2 pp [28.6, 61.9] |

All quartile cells contain 42 images. Intervals use 2,000 category-stratified bootstrap resamples with seed 2026. These are associations within the selected balanced subset, not causal effects.
