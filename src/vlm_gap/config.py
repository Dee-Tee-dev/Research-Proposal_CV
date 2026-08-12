from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "splits" / "balanced_subset_manifest.csv"
DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "images"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
INTERNVL_MODEL_NAME = "OpenGVLab/InternVL3_5-2B"
YOLO_WORLD_MODEL_NAME = "yolov8s-worldv2.pt"

CATEGORY_LABELS = (
    "roof",
    "light source",
    "stove",
    "trash container",
    "switch",
    "footwear",
)

CLIP_PROMPT_TEMPLATE = "a photo of a household {}"
OBJECT_PROMPT = "the main household object in this image is"

CLASSIFICATION_PROMPT = """Look at the image and identify the main household object.
Choose exactly one label from: {labels}.
Answer with the label only."""

CAPTION_PROMPT = (
    "Describe the main household object in this image in one short sentence. "
    "Do not guess the country, region, or income level."
)
