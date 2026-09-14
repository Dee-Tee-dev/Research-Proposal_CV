from __future__ import annotations

import gradio as gr
import pandas as pd
from PIL import Image

from vlm_gap.config import OBJECT_PROMPT
from vlm_gap.models import (
    BLIPCaptioner,
    CLIPClassifier,
    InternVisionLanguageModel,
    QwenVisionLanguageModel,
    YOLOWorldDetector,
)


MODEL_FACTORIES = {
    "CLIP": CLIPClassifier,
    "BLIP": BLIPCaptioner,
    "Qwen2.5-VL-3B": QwenVisionLanguageModel,
    "InternVL3.5-2B": InternVisionLanguageModel,
    "YOLO-World": YOLOWorldDetector,
}
model_cache: dict[str, object] = {}


def get_model(name: str):
    if name not in model_cache:
        model_cache[name] = MODEL_FACTORIES[name]()
    return model_cache[name]


def analyse(image: Image.Image, selected_models: list[str]):
    if image is None:
        raise gr.Error("Please upload an image.")
    if not selected_models:
        raise gr.Error("Select at least one model.")

    image = image.convert("RGB")
    rows: list[dict[str, object]] = []
    annotated = None
    for name in selected_models:
        try:
            model = get_model(name)
            if isinstance(model, CLIPClassifier):
                prediction = model.classify(image)
                rows.append({
                    "Model": name,
                    "Task": "classification",
                    "Output": prediction.label,
                    "Details": f"confidence={prediction.confidence:.3f}",
                })
            elif isinstance(model, BLIPCaptioner):
                rows.extend([
                    {
                        "Model": name,
                        "Task": "captioning (baseline)",
                        "Output": model.caption(image),
                        "Details": "unprompted",
                    },
                    {
                        "Model": name,
                        "Task": "captioning (prompted)",
                        "Output": model.caption(image, prompt=OBJECT_PROMPT),
                        "Details": "label-free object prompt",
                    },
                ])
            elif isinstance(
                model,
                (QwenVisionLanguageModel, InternVisionLanguageModel),
            ):
                prediction = model.classify(image)
                rows.extend([
                    {
                        "Model": name,
                        "Task": "classification",
                        "Output": prediction.label or "unparsed",
                        "Details": prediction.raw_output,
                    },
                    {
                        "Model": name,
                        "Task": "captioning",
                        "Output": model.caption(image),
                        "Details": "deterministic decoding",
                    },
                ])
            elif isinstance(model, YOLOWorldDetector):
                detection = model.detect(image, include_annotation=True)
                annotated = detection.annotated_image
                rows.append({
                    "Model": name,
                    "Task": "detection",
                    "Output": detection.label or "no detection",
                    "Details": (
                        f"top confidence={detection.confidence:.3f}"
                        if detection.confidence is not None
                        else "no box above threshold"
                    ),
                })
        except Exception as error:  # Demo should explain optional setup failures.
            rows.append({
                "Model": name,
                "Task": "load/run error",
                "Output": type(error).__name__,
                "Details": str(error),
            })

    frame = pd.DataFrame(rows, columns=["Model", "Task", "Output", "Details"])
    return frame, annotated, (
        f"Completed {len(selected_models)} model run(s). Models are cached after "
        "their first load; later comparisons are faster."
    )


with gr.Blocks(
    title="VLM Income-Gap Benchmark",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    css="""
    .hero {background: linear-gradient(120deg,#123d62,#217eaf); color: white;
           border-radius: 14px; padding: 18px 22px; margin-bottom: 12px;}
    .note {color: #536273; font-size: 0.92rem;}
    """,
) as demo:
    gr.Markdown(
        "<div class='hero'><h1>Household-object VLM benchmark</h1>"
        "<p>Run live comparisons across classification, captioning, and "
        "open-vocabulary detection.</p></div>"
        "<p class='note'>The six candidate labels are fixed. No income, country, "
        "or location is provided to any model. This demo runs inference on the "
        "uploaded image; the paper's quartile results are precomputed separately.</p>"
    )
    with gr.Row():
        image_input = gr.Image(type="pil", sources=["upload", "clipboard"], label="Upload an image")
        model_input = gr.CheckboxGroup(
            choices=list(MODEL_FACTORIES),
            value=["CLIP", "BLIP"],
            label="Models to compare (select one or more)",
        )
    gr.Markdown("**Tip:** CLIP and BLIP are the lightest starting choices. Qwen, InternVL, and YOLO-World may take longer on first use while checkpoints load.", elem_classes=["note"])
    with gr.Row():
        run_button = gr.Button("Run live comparison", variant="primary")
    comparison = gr.Dataframe(
        headers=["Model", "Task", "Output", "Details"],
        interactive=False,
        label="Model outputs",
    )
    clear_button = gr.ClearButton([image_input, comparison], value="Clear")
    annotated_image = gr.Image(label="YOLO-World detections")
    status = gr.Markdown("Upload an image and select at least one model.", elem_classes=["note"])
    run_button.click(
        analyse,
        inputs=[image_input, model_input],
        outputs=[comparison, annotated_image, status],
    )


if __name__ == "__main__":
    demo.launch()
