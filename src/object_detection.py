import io
import json
from pathlib import Path

import httpx
import torch
from PIL import Image, ImageDraw
from transformers import pipeline

pipe = pipeline(
    task="zero-shot-object-detection",
    model="IDEA-Research/grounding-dino-base",
    torch_dtype=torch.float32,
    use_fast=True,
    device=torch.device("xpu" if torch.xpu.is_available() else "cpu"),
)

labels = ["a cat.", "a controller.", "a furniture."]
image_bytes = httpx.get("http://images.cocodataset.org/val2017/000000039769.jpg").read()
image = Image.open(io.BytesIO(image_bytes))

predictions = pipe(image, labels)

draw = ImageDraw.Draw(image)
for prediction in predictions:
    if float(prediction["score"]) < 0.2:
        continue

    box = prediction["box"]
    label = prediction["label"]
    score = prediction["score"]

    xmin, ymin, xmax, ymax = box.values()
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{label}: {round(score, 2)}", fill="white")

Path("./out/object_detection.json").write_text(json.dumps(predictions, indent=2))
image.save("./out/object_detection.jpg")
