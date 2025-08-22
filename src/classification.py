import json
from pathlib import Path

import torch
from transformers import pipeline

clip = pipeline(
    task="zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    torch_dtype=torch.bfloat16,
    use_fast=True,
    device=torch.device("xpu" if torch.xpu.is_available() else "cpu"),
)

labels = ["a photo of a cat", "a photo of cats and controllers", "a photo of a car"]

predictions = clip(
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    candidate_labels=labels,
)

data = json.dumps(predictions, indent=2)
Path("./out/classification.json").write_text(data)

print(data)
