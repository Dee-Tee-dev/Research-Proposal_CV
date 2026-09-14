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

    return pd.DataFrame(rows), annotated


with gr.Blocks(title="VLM Income-Gap Benchmark") as demo:
    gr.Markdown(
        "# Household-object benchmark\n"
        "Compare classification, captioning, and open-vocabulary detection. "
        "The six candidate labels are fixed and no income or location is "
        "provided to any model."
    )
    with gr.Row():
        image_input = gr.Image(type="pil", label="Household-object image")
        model_input = gr.CheckboxGroup(
            choices=list(MODEL_FACTORIES),
            value=["CLIP", "BLIP"],
            label="Models to compare",
        )
    run_button = gr.Button("Run comparison", variant="primary")
    comparison = gr.Dataframe(
        headers=["Model", "Task", "Output", "Details"],
        interactive=False,
        label="Model outputs",
    )
    annotated_image = gr.Image(label="YOLO-World detections")
    run_button.click(
        analyse,
        inputs=[image_input, model_input],
        outputs=[comparison, annotated_image],
    )


if __name__ == "__main__":
    demo.launch()
