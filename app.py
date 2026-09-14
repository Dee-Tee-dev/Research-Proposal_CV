from __future__ import annotations

import sys
from pathlib import Path

# Allow `python app.py` to work directly from a fresh checkout without
# requiring a separate editable-install step.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gradio as gr
import pandas as pd
from PIL import Image

try:
    import spaces
except ImportError:  # Local runs do not need the ZeroGPU helper package.
    class _LocalSpaces:
        @staticmethod
        def GPU(function):
            return function
    spaces = _LocalSpaces()

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


@spaces.GPU
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
            detail = str(error)
            if isinstance(error, OSError) and ("Hugging Face" in detail or "valid repository" in detail):
                detail = (
                    "Checkpoint download failed. Connect to the internet and "
                    "retry; public Hugging Face models download automatically "
                    "on first use. Original error: " + detail
                )
            rows.append({
                "Model": name,
                "Task": "load/run error",
                "Output": type(error).__name__,
                "Details": detail,
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
    .gradio-container {max-width: 1120px !important; padding-top: 18px !important;}
    .hero {background: linear-gradient(120deg,#123b5d 0%,#176b87 55%,#2a9d8f 100%);
           color: white; border-radius: 16px; padding: 20px 26px; margin-bottom: 10px;
           box-shadow: 0 6px 18px rgba(18,59,93,.18);}
    .hero h1 {margin: 0 0 4px 0; font-size: 2rem;}
    .hero p {margin: 0; opacity: .92;}
    .note {color: #536273; font-size: 0.9rem; margin: 6px 0 12px 0;}
    .panel {border: 1px solid #dce7ec; border-radius: 12px; padding: 10px;
            background: #fbfdfe;}
    footer {display: none !important;}
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
    with gr.Row(equal_height=True):
        with gr.Column(scale=5, elem_classes=["panel"]):
            image_input = gr.Image(type="pil", sources=["upload", "clipboard"], label="Upload an image")
        with gr.Column(scale=4, elem_classes=["panel"]):
            model_input = gr.CheckboxGroup(
                choices=list(MODEL_FACTORIES),
                value=["CLIP", "BLIP"],
                label="Models to compare",
            )
    gr.Markdown("CLIP and BLIP load fastest. Larger models may take longer on their first run.", elem_classes=["note"])
    with gr.Row():
        run_button = gr.Button("Run live comparison", variant="primary")
    comparison = gr.Dataframe(
        headers=["Model", "Task", "Output", "Details"],
        interactive=False,
        label="Model outputs",
    )
    annotated_image = gr.Image(label="YOLO-World detections", show_label=True)
    with gr.Row():
        clear_button = gr.ClearButton([image_input, comparison, annotated_image], value="Clear")
    status = gr.Markdown("Upload an image and select at least one model.", elem_classes=["note"])
    run_button.click(
        analyse,
        inputs=[image_input, model_input],
        outputs=[comparison, annotated_image, status],
    )


if __name__ == "__main__":
    demo.launch()
