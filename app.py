from __future__ import annotations

import gradio as gr
from PIL import Image

from vlm_gap.config import OBJECT_PROMPT
from vlm_gap.models import BLIPCaptioner, CLIPClassifier


clip_model = None
blip_model = None


def get_models():
    global clip_model, blip_model
    if clip_model is None:
        clip_model = CLIPClassifier()
    if blip_model is None:
        blip_model = BLIPCaptioner()
    return clip_model, blip_model


def analyse(image: Image.Image):
    if image is None:
        raise gr.Error("Please upload an image.")
    clip, blip = get_models()
    prediction = clip.classify(image.convert("RGB"))
    baseline = blip.caption(image.convert("RGB"))
    prompted = blip.caption(image.convert("RGB"), prompt=OBJECT_PROMPT)
    scores = {label: round(score, 4) for label, score in prediction.scores.items()}
    return prediction.label, scores, baseline, prompted


demo = gr.Interface(
    fn=analyse,
    inputs=gr.Image(type="pil", label="Household-object image"),
    outputs=[
        gr.Textbox(label="CLIP top prediction"),
        gr.Label(label="CLIP category scores"),
        gr.Textbox(label="BLIP caption"),
        gr.Textbox(label="BLIP caption with object-focused prompt"),
    ],
    title="Household Objects Across Income Contexts",
    description=(
        "Compare CLIP classification with unprompted and object-focused BLIP "
        "captions. The prompt does not contain the correct category label."
    ),
)


if __name__ == "__main__":
    demo.launch()
