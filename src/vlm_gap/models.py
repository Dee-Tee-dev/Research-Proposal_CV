from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

import torch

from .config import (
    BLIP_MODEL_NAME,
    CAPTION_PROMPT,
    CATEGORY_LABELS,
    CLASSIFICATION_PROMPT,
    CLIP_MODEL_NAME,
    CLIP_PROMPT_TEMPLATE,
    INTERNVL_MODEL_NAME,
    QWEN_MAX_PIXELS,
    QWEN_MIN_PIXELS,
    QWEN_MODEL_NAME,
    YOLO_WORLD_MODEL_NAME,
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


@dataclass
class GenerativePrediction:
    raw_output: str
    label: str | None


@dataclass
class DetectionPrediction:
    label: str | None
    confidence: float | None
    detected_labels: tuple[str, ...]
    boxes: tuple[dict[str, object], ...]
    annotated_image: object | None = None


def parse_category_response(text: str) -> str | None:
    """Extract one of the fixed study labels without accepting partial words."""
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    exact = re.sub(r"[^a-z0-9 ]+", "", normalized).strip()
    if exact in CATEGORY_LABELS:
        return exact

    matches = []
    for label in CATEGORY_LABELS:
        pattern = rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])"
        match = re.search(pattern, normalized)
        if match:
            matches.append((match.start(), label))
    return min(matches)[1] if matches else None


def _classification_prompt() -> str:
    return CLASSIFICATION_PROMPT.format(labels=", ".join(CATEGORY_LABELS))


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


class QwenVisionLanguageModel:
    """Qwen2.5-VL adapter for forced-choice classification and captioning."""

    def __init__(self, model_name: str = QWEN_MODEL_NAME, device: str | None = None):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device or choose_device()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=QWEN_MIN_PIXELS,
            max_pixels=QWEN_MAX_PIXELS,
        )

    def _generate(self, image, prompt: str, max_new_tokens: int) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as error:
            raise RuntimeError(
                "Qwen support needs the optional qwen-vl-utils package. "
                "Install requirements-models.txt."
            ) from error

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            output[len(input_ids):]
            for input_ids, output in zip(inputs.input_ids, generated)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def classify(self, image) -> GenerativePrediction:
        output = self._generate(image, _classification_prompt(), max_new_tokens=12)
        return GenerativePrediction(output, parse_category_response(output))

    def caption(self, image) -> str:
        return self._generate(image, CAPTION_PROMPT, max_new_tokens=40)


class InternVisionLanguageModel:
    """InternVL adapter using the model's documented ``chat`` interface."""

    def __init__(
        self,
        model_name: str = INTERNVL_MODEL_NAME,
        device: str | None = None,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or choose_device()
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )

    def _pixel_values(self, image):
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        transform = transforms.Compose([
            transforms.Resize(
                (448, 448),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        dtype = next(self.model.parameters()).dtype
        return transform(image.convert("RGB")).unsqueeze(0).to(
            device=self.device,
            dtype=dtype,
        )

    def _generate(self, image, prompt: str, max_new_tokens: int) -> str:
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                self._pixel_values(image),
                prompt,
                generation_config,
            )
        return str(response).strip()

    def classify(self, image) -> GenerativePrediction:
        output = self._generate(image, _classification_prompt(), max_new_tokens=12)
        return GenerativePrediction(output, parse_category_response(output))

    def caption(self, image) -> str:
        return self._generate(image, CAPTION_PROMPT, max_new_tokens=40)


class YOLOWorldDetector:
    """Open-vocabulary detector evaluated as an image-level baseline."""

    def __init__(
        self,
        model_name: str = YOLO_WORLD_MODEL_NAME,
        device: str | None = None,
        confidence: float = 0.25,
    ):
        try:
            from ultralytics import YOLOWorld
        except ImportError as error:
            raise RuntimeError(
                "YOLO-World support needs ultralytics. "
                "Install requirements-models.txt."
            ) from error

        self.device = device or choose_device()
        self.confidence = confidence
        self.model = YOLOWorld(model_name)
        self.model.set_classes(list(CATEGORY_LABELS))

    def detect(self, image, include_annotation: bool = False) -> DetectionPrediction:
        result = self.model.predict(
            source=np.asarray(image.convert("RGB")),
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        boxes = []
        for class_id, confidence, coordinates in zip(
            result.boxes.cls.detach().cpu().tolist(),
            result.boxes.conf.detach().cpu().tolist(),
            result.boxes.xyxy.detach().cpu().tolist(),
        ):
            label = str(result.names[int(class_id)])
            boxes.append({
                "label": label,
                "confidence": float(confidence),
                "xyxy": [float(value) for value in coordinates],
            })
        boxes.sort(key=lambda item: float(item["confidence"]), reverse=True)
        top = boxes[0] if boxes else None
        annotated = result.plot()[:, :, ::-1].copy() if include_annotation else None
        return DetectionPrediction(
            label=str(top["label"]) if top else None,
            confidence=float(top["confidence"]) if top else None,
            detected_labels=tuple(dict.fromkeys(str(box["label"]) for box in boxes)),
            boxes=tuple(boxes),
            annotated_image=annotated,
        )
