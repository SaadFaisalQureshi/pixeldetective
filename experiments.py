"""Run robustness, higher-resolution, and score-map evaluation experiments."""
import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image, ImageOps

from modeling import IMAGE_SIZES, MODELS, embeddings, load_encoder, load_head, probabilities
from score_map import occlusion_scores
from train import evaluate, extract, fit_head

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
TRANSFORMS = ("Original", "JPEG Q65", "Half size")


def read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def transform(image, name):
    if name == "Half size":
        return image.resize(tuple(max(1, n // 2) for n in image.size), Image.Resampling.LANCZOS)
    if name == "JPEG Q65":
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=65)
        with Image.open(buffer) as decoded:
            return decoded.copy()
    return image


def robustness(args, device):
    source = args.external_manifest or ARTIFACTS / "split.csv"
    rows = read_rows(source)
    if args.external_manifest is None:
        rows = [row for row in rows if row["split"] == "test"]
    labels = np.asarray([int(row["label"]) for row in rows])
    if set(labels) != {0, 1}:
        raise ValueError("Provide both authentic (0) and manipulated (1) images.")
    name = "External labeled set" if args.external_manifest else "CASIA v2 held-out split"
    result = {"dataset": name, "sample_count": len(rows), "models": {}}
    for key in args.models:
        encoder = load_encoder(key, device)
        head = load_head(ARTIFACTS / f"{key}_head.json", key)
        preds = {name: [] for name in TRANSFORMS}
        for start in range(0, len(rows), args.batch_size):
            images = []
            for row in rows[start:start + args.batch_size]:
                with Image.open(args.data / row["path"]) as image:
                    images.append(ImageOps.exif_transpose(image).convert("RGB"))
            for variant in TRANSFORMS:
                features = embeddings(key, encoder, [transform(image, variant) for image in images])
                preds[variant].extend(probabilities(features, head).argmax(axis=1))
        result["models"][key] = {name: evaluate(labels, pred) for name, pred in preds.items()}
        del encoder
    return result


def resolution(args, device):
    key = "convnext"
    rows = read_rows(ARTIFACTS / "split.csv")
    labels = np.asarray([int(row["label"]) for row in rows])
    train, val, test = (np.flatnonzero([row["split"] == split for row in rows])
                        for split in ("train", "validation", "test"))
    saved = json.loads((ARTIFACTS / "metrics.json").read_text())["models"][key]
    result = {
        "dataset": "CASIA v2 held-out split",
        "128": {name: saved[name] for name in ("validation_macro_f1", "test")},
    }
    features = extract(args.data, rows, key, args.batch_size, device, image_size=224)
    head, score, c_value, _, _ = fit_head(features, labels, train, val, key, 224)
    result["224"] = {
        "validation_macro_f1": score, "selected_C": c_value,
        "test": evaluate(labels[test], probabilities(features[test], head).argmax(axis=1)),
    }
    encoder = load_encoder(key, device)
    with Image.open(args.data / rows[test[0]]["path"]) as example:
        for side in (IMAGE_SIZES[key], 224):
            embeddings(key, encoder, [example], image_size=side)
            if device == "cuda":
                torch.cuda.synchronize()
            samples = []
            for _ in range(10):
                start = time.perf_counter()
                embeddings(key, encoder, [example], image_size=side)
                if device == "cuda":
                    torch.cuda.synchronize()
                samples.append(1000 * (time.perf_counter() - start))
            result[str(side)]["median_embedding_ms"] = float(np.median(samples))
    result["device"] = device
    return result


def localization(args, device):
    rows = read_rows(args.mask_manifest)
    if not rows:
        raise ValueError("Provide at least one manipulated image and its mask.")
    batches = []
    encoder = load_encoder("convnext", device)
    head = load_head(ARTIFACTS / "convnext_head.json", "convnext")
    for row in rows:
        with Image.open(args.data / row["path"]) as image, Image.open(args.data / row["mask"]) as mask:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if mask.size != image.size:
                raise ValueError(f"Mask dimensions differ from image: {row['path']}")
            mask = np.asarray(mask.convert("L")) >= 128
            if not mask.any():
                raise ValueError(f"Mask has no edited pixels: {row['mask']}")
            preview = image.copy()
            preview.thumbnail((900, 480))

            def predict(images):
                return probabilities(embeddings("convnext", encoder, images), head)[:, 1]

            grid = occlusion_scores(preview, predict)
            heatmap = Image.fromarray(grid).resize(image.size, Image.Resampling.NEAREST)
            hot = np.asarray(heatmap) >= 0.10
            iou = (hot & mask).sum() / (hot | mask).sum()
            batches.append(float(iou))
    return {"method": "Occlusion score sensitivity", "threshold_score_drop": 0.10,
            "sample_count": len(rows), "mean_iou": float(np.mean(batches))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["robustness", "resolution", "localization"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path)
    parser.add_argument("--mask-manifest", type=Path)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=["convnext", "vit"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.experiment == "localization" and args.mask_manifest is None):
        parser.error("Use a positive batch size and supply --mask-manifest for localization.")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = {"robustness": robustness, "resolution": resolution, "localization": localization}
    result = run[args.experiment](args, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(args.output)
