"""Measure how covering parts of a photo changes its manipulation score."""
import numpy as np
from PIL import Image


def occlusion_scores(image, predict, grid_size=5):
    # The app passes a bounded preview to keep the occlusion batch small.
    columns, rows = min(image.width, grid_size), min(image.height, grid_size)
    x_edges = np.linspace(0, image.width, columns + 1, dtype=int)
    y_edges = np.linspace(0, image.height, rows + 1, dtype=int)
    reference = float(predict([image])[0])
    values = []
    for y in range(rows):
        row = []
        for x in range(columns):
            covered = image.copy()
            covered.paste((127, 127, 127), (x_edges[x], y_edges[y], x_edges[x + 1], y_edges[y + 1]))
            row.append(covered)
        values.extend(reference - predict(row))
    return np.asarray(values, dtype=np.float32).reshape(rows, columns)


def overlay(image, grid):
    strength = Image.fromarray(np.uint8(np.clip(grid, 0, 1) * 255))
    alpha = strength.resize(image.size, Image.Resampling.NEAREST).point(lambda value: min(180, 4 * value))
    return Image.composite(Image.new("RGB", image.size, "#e84c39"), image, alpha)
