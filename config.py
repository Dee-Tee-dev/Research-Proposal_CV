from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "splits" / "balanced_subset_manifest.csv"
DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "images"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

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

