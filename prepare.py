"""Audit CASIA and create a reproducible group-aware split."""
import argparse
import csv
import hashlib
import math
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABELS = ("Authentic", "Tampered")


def assign_splits(rows, groups):
    # A connected source group larger than the target hold-out cannot fit into it.
    # Keep it intact in training and stratify the remaining data by group.
    sizes = Counter(groups)
    oversized = {group for group, size in sizes.items() if size > len(rows) / 5}
    candidates = [index for index, group in enumerate(groups) if group not in oversized]
    print("Oversized training groups:", sorted(sizes[group] for group in oversized))
    for index, row in enumerate(rows):
        row.update(group=groups[index], split="train")
    candidate_labels = [int(rows[index]["label"]) for index in candidates]
    candidate_groups = [groups[index] for index in candidates]
    fold_count = max(3, min(5, math.ceil(5 * len(candidates) / len(rows))))
    splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=42)
    for split, (_, indices) in zip(("test", "validation"),
                                   splitter.split(candidates, candidate_labels, candidate_groups)):
        for index in indices:
            rows[candidates[index]]["split"] = split


def prepare(root, output):
    root = Path(root).resolve()
    rows = []
    parents = []
    owners = {}
    rejected = []
    formats = Counter()

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        prefix = path.stem.lower().split("_")[0]
        if prefix not in ("au", "tp") or path.parent.name.lower() != prefix:
            continue
        label = int(prefix == "tp")
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                # A coarse fingerprint also links some visually identical re-encodes.
                thumbnail = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
                pixels = thumbnail.tobytes()
                average = sum(pixels) / len(pixels)
                signature = tuple(pixel > average for pixel in pixels)
        except (OSError, ValueError) as error:
            rejected.append((str(path.relative_to(root)), str(error)))
            continue
        if label:
            sources = [key for key in re.findall(r"(?<![a-z0-9])[a-z]{3}\d{5}(?![a-z0-9])",
                                                path.stem.lower()) if key != "xxx00000"]
        else:
            match = re.match(r"au_([a-z]{3})_(\d{5})(?:_|$)", path.stem.lower())
            sources = ["".join(match.groups())] if match else []
        formats[label, path.suffix.lower()] += 1
        index = len(rows)
        rows.append({"path": path.relative_to(root).as_posix(), "label": label,
                     "source_ids": " ".join(sources)})
        parents.append(index)
        keys = [("pixels", digest), *(("source", source) for source in sources)]
        if len(set(signature)) == 2:
            keys.append(("thumbnail", signature))
        for key in keys:
            previous = owners.setdefault(key, index)
            parents[find(index)] = find(previous)

    print("Readable:", Counter(row["label"] for row in rows))
    print("Unreadable:", len(rejected))
    print("File formats by class:", formats)
    print("Without parsed source ID:", sum(not row["source_ids"] for row in rows))
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Expected both Au_ and Tp_ images in the extracted CASIA release.")
    groups = [find(index) for index in range(len(rows))]
    print("Largest source groups:", Counter(groups).most_common(5))
    assign_splits(rows, groups)
    counts = Counter((row["split"], row["label"]) for row in rows)
    if len(counts) != 6:
        raise ValueError(f"At least one split lacks a class: {counts}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with output.with_name("unreadable.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "error"])
        writer.writerows(rejected)
    print("Split counts:", counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output", default="artifacts/split.csv")
    args = parser.parse_args()
    prepare(args.root, args.output)
