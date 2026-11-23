"""Measure how covering parts of a photo changes its manipulation score."""
import numpy as np
from PIL import Image, ImageDraw


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
    """Render signed score changes using a relative color scale."""
    rendered = image.convert("RGBA")
    colors = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(colors)
    rows, columns = grid.shape
    peak = float(np.abs(grid).max())
    if peak < 0.001:
        return image.copy()
    x_edges = np.linspace(0, image.width, columns + 1, dtype=int)
    y_edges = np.linspace(0, image.height, rows + 1, dtype=int)
    for y, row in enumerate(grid):
        for x, change in enumerate(row):
            if abs(change) < 0.001:
                continue
            color = (234, 62, 42) if change > 0 else (32, 106, 242)
            opacity = round(190 * abs(change) / peak)
            draw.rectangle((x_edges[x], y_edges[y], x_edges[x + 1] - 1, y_edges[y + 1] - 1),
                           fill=(*color, opacity))
    return Image.alpha_composite(rendered, colors).convert("RGB")
