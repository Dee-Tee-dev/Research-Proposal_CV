from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    ROOT / "results/demo_evidence/01_strong_success_clip_blip.png",
    ROOT / "results/demo_evidence/02_strong_success_yolo.png",
    ROOT / "results/demo_evidence/03_cultural_form_failure.png",
]
LABELS = [
    "A  CLIP and BLIP success",
    "B  YOLO-World localization",
    "C  Cultural-form failure",
]
OUTPUT = ROOT / "paper/assets/combined/qualitative_examples.png"


def cover(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def main() -> None:
    panel_width, panel_height, label_height, gap = 760, 530, 52, 14
    canvas = Image.new(
        "RGB",
        (panel_width * 3 + gap * 2, panel_height + label_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial Bold.ttf", 27)
    except OSError:
        font = ImageFont.load_default()
    for index, (path, label) in enumerate(zip(INPUTS, LABELS)):
        image = cover(Image.open(path).convert("RGB"), panel_width, panel_height)
        x = index * (panel_width + gap)
        canvas.paste(image, (x, label_height))
        draw.rectangle((x, 0, x + panel_width, label_height), fill="#17365D")
        draw.text((x + 16, 12), label, fill="white", font=font)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, quality=94)
    print(OUTPUT)


if __name__ == "__main__":
    main()
