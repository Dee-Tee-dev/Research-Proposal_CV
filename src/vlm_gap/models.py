from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import (
    BLIP_MODEL_NAME,
    CATEGORY_LABELS,
    CLIP_MODEL_NAME,
    CLIP_PROMPT_TEMPLATE,
)


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ClipPrediction:
    label: str
    confidence: float
    correct_rank: int
    scores: dict[str, float]


class CLIPClassifier:
    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str | None = None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or choose_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def classify(self, image, correct_label: str | None = None) -> ClipPrediction:
        prompts = [CLIP_PROMPT_TEMPLATE.format(label) for label in CATEGORY_LABELS]
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits_per_image[0]
        probabilities = logits.softmax(dim=-1).detach().cpu().tolist()
        ranked = sorted(
            zip(CATEGORY_LABELS, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        rank = 0
        if correct_label is not None:
            rank = next(
                index
                for index, (label, _) in enumerate(ranked, start=1)
                if label == correct_label
            )
        return ClipPrediction(
            label=ranked[0][0],
            confidence=float(ranked[0][1]),
            correct_rank=rank,
            scores={label: float(score) for label, score in ranked},
        )


class BLIPCaptioner:
    def __init__(self, model_name: str = BLIP_MODEL_NAME, device: str | None = None):
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.device = device or choose_device()
        self.model = (
            BlipForConditionalGeneration.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )
        self.processor = BlipProcessor.from_pretrained(model_name)

    def caption(self, image, prompt: str | None = None) -> str:
        if prompt:
            inputs = self.processor(image, prompt, return_tensors="pt")
        else:
            inputs = self.processor(image, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=30)
        return self.processor.decode(output[0], skip_special_tokens=True).strip()
