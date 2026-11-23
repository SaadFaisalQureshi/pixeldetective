"""Shared preprocessing and scoring for the notebook and the web app."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import ImageOps
from transformers import AutoImageProcessor, AutoModel

MODELS = {
    "efficientnet": ("EfficientNet-B0", "google/efficientnet-b0",
                     "3d8d75d0812b65064f0151b456f869c6344bf20c"),
    "vit": ("ViT-Tiny", "WinKawaks/vit-tiny-patch16-224",
            "77d1485af66b34d4ed0fe95dbb0c60c7496f950b"),
    "dinov2": ("DINOv2 Base", "facebook/dinov2-base",
               "f9e44c814b77203eaa57a6bdbbd535f21ede1415"),
    "convnext": ("ConvNeXt V2 Tiny", "facebook/convnextv2-tiny-22k-224",
                 "9cba4896e97bb86b1eb609e482a2149d84f345bc"),
}
IMAGE_SIZES = dict.fromkeys(MODELS, 128)
IMAGE_SIZES["dinov2"] = 126
LABELS = ["Authentic", "Manipulated"]


def load_encoder(key, device):
    _, repo, revision = MODELS[key]
    processor = AutoImageProcessor.from_pretrained(repo, revision=revision, use_fast=False)
    options = {"add_pooling_layer": False} if key == "vit" else {}
    model = AutoModel.from_pretrained(repo, revision=revision, **options).to(device).eval()
    return processor, model


@torch.inference_mode()
def embeddings(key, encoder, images, image_size=None):
    processor, model = encoder
    images = [ImageOps.exif_transpose(image).convert("RGB") for image in images]
    side = image_size if image_size is not None else IMAGE_SIZES[key]
    size = {"height": side, "width": side}
    crop = {"crop_size": size} if key == "efficientnet" else {}
    if key == "dinov2":
        crop = {"do_center_crop": False}
    if key == "convnext":
        size = {"shortest_edge": side}
    inputs = processor(images=images, size=size, return_tensors="pt", **crop).to(model.device)
    options = {"interpolate_pos_encoding": True} if key == "vit" else {}
    output = model(**inputs, **options)
    if key in {"vit", "dinov2"}:
        features = output.last_hidden_state[:, 0]
    else:
        features = output.pooler_output.flatten(1)
    return features.cpu().float().numpy()


def load_head(path, key):
    head = json.loads(Path(path).read_text())
    if head["source"] != list(MODELS[key][1:]) or head["labels"] != LABELS or head["image_size"] != IMAGE_SIZES[key]:
        raise ValueError(f"Checkpoint metadata does not match {key}.")
    return head


def probabilities(features, head):
    logits = features @ np.asarray(head["weights"]) + head["bias"]
    manipulated = np.exp(-np.logaddexp(0, -logits))
    return np.column_stack((1 - manipulated, manipulated))
