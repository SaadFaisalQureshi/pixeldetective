"""Fit and compare logistic heads on frozen Hugging Face image encoders."""
import argparse
from collections import Counter
import csv
from contextlib import ExitStack
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare import prepare
from modeling import IMAGE_SIZES, LABELS, MODELS, embeddings, load_encoder, probabilities

SEED = 42
CANDIDATES = (0.01, 0.1, 1.0, 10.0)


def evaluate(y, predictions):
    return {
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro"),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y, predictions, labels=[0, 1], target_names=LABELS, output_dict=True, zero_division=0
        ),
    }


def extract(root, rows, key, batch_size, device):
    encoder = load_encoder(key, device)
    batches = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        with ExitStack() as stack:
            images = [stack.enter_context(Image.open(root / row["path"])) for row in batch]
            batches.append(embeddings(key, encoder, images))
        if start == 0 or (start // batch_size + 1) % 50 == 0:
            print(f"{key}: {min(start + batch_size, len(rows))}/{len(rows)} images", flush=True)
    return np.concatenate(batches)


def run(root, output, batch_size):
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = output / "split.csv"
    prepare(root, manifest)
    with manifest.open(newline="") as file:
        rows = list(csv.DictReader(file))
    groups = {}
    counts = {}
    for row in rows:
        if groups.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError("A source group crosses split boundaries.")
        label_counts = counts.setdefault(row["split"], Counter())
        label_counts[int(row["label"])] += 1
    splits = np.array([row["split"] for row in rows])
    train, val, test = (np.flatnonzero(splits == name) for name in ("train", "validation", "test"))
    jpeg = np.asarray([Path(rows[i]["path"]).suffix.lower() in {".jpg", ".jpeg"} for i in test])
    with (output / "unreadable.csv").open(newline="") as file:
        unreadable_count = sum(1 for _ in csv.DictReader(file))
    audit = {"class_counts": counts, "known_source_groups": len(groups),
             "largest_group": max(Counter(row["group"] for row in rows).values()),
             "images_without_parsed_source_id": sum(not row["source_ids"] for row in rows),
             "unreadable_images": unreadable_count,
             "files_by_class_and_extension": {
                 str(label): dict(Counter(Path(row["path"]).suffix.lower()
                                          for row in rows if int(row["label"]) == label))
                 for label in (0, 1)
             }}
    print(json.dumps(audit, indent=2), flush=True)
    y = np.asarray([int(row["label"]) for row in rows])
    metrics = {"seed": SEED, "audit": audit, "models": {}}
    test_predictions = {}
    for key, (name, repo, revision) in MODELS.items():
        x = extract(root, rows, key, batch_size, device)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        best = None
        scores = {}
        for c_value in CANDIDATES:
            candidate = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, max_iter=2000))
            candidate.fit(x[train], y[train])
            score = f1_score(y[val], candidate.predict(x[val]), average="macro")
            scores[c_value] = score
            if best is None or score > best[0]:
                best = (score, c_value, candidate)
        score, c_value, pipeline = best
        scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
        weights = classifier.coef_[0] / scaler.scale_
        head = {"source": [repo, revision], "labels": LABELS,
                "weights": weights.tolist(), "image_size": IMAGE_SIZES[key],
                "bias": float(classifier.intercept_[0] - scaler.mean_ @ weights)}
        # Evaluate the exact exported scoring function used in Gradio.
        prediction = probabilities(x[test], head).argmax(axis=1)
        np.testing.assert_array_equal(prediction, pipeline.predict(x[test]))
        test_predictions[key] = prediction
        (output / f"{key}_head.json").write_text(json.dumps(head))
        test_report = evaluate(y[test], prediction)
        if len(np.unique(y[test][jpeg])) == 2:
            test_report["jpeg_only"] = evaluate(y[test][jpeg], prediction[jpeg])
        metrics["models"][key] = {"display_name": name, "source": repo, "revision": revision,
                                   "selected_C": c_value, "validation_macro_f1": score,
                                   "validation_grid": scores, "test": test_report}
    selected = max(MODELS, key=lambda key: metrics["models"][key]["validation_macro_f1"])
    majority = np.bincount(y[train]).argmax()
    metrics["selected_by_validation"] = selected
    metrics["majority_baseline_test"] = evaluate(y[test], np.full(len(test), majority))
    metrics["held_out_disagreements_vs_vit"] = {
        key: int(np.sum(predictions != test_predictions["vit"]))
        for key, predictions in test_predictions.items() if key != "vit"
    }
    with (output / "test_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "actual", *MODELS])
        for i, index in enumerate(test):
            writer.writerow([rows[index]["path"], int(y[index]),
                             *(int(prediction[i]) for prediction in test_predictions.values())])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", default=Path("artifacts"), type=Path)
    parser.add_argument("--batch-size", default=32, type=int)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    run(args.data, args.output, args.batch_size)
