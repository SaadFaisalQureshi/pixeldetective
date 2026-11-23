"""Measure how covering parts of a photo changes its manipulation score."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
    markers = []
    for y, row in enumerate(grid):
        for x, change in enumerate(row):
            if abs(change) < 0.001:
                continue
            color = (194, 61, 0) if change > 0 else (0, 68, 168)
            opacity = round(190 * abs(change) / peak)
            draw.rectangle((x_edges[x], y_edges[y], x_edges[x + 1] - 1, y_edges[y + 1] - 1),
                           fill=(*color, opacity))
            width, height = x_edges[x + 1] - x_edges[x], y_edges[y + 1] - y_edges[y]
            if min(width, height) >= 20:
                markers.append((int(x_edges[x] + width // 2), int(y_edges[y] + height // 2),
                                "-" if change > 0 else "+"))
    result = Image.alpha_composite(rendered, colors).convert("RGB")
    labels = ImageDraw.Draw(result)
    font_size = max(12, min(32, min(image.width // columns, image.height // rows) // 3))
    font = ImageFont.load_default(size=font_size)
    for x, y, sign in markers:
        radius = font_size // 2 + 2
        labels.ellipse((x - radius, y - radius, x + radius, y + radius),
                       fill="#111827", outline="white", width=2)
        labels.text((x, y), sign, fill="white", font=font, anchor="mm", stroke_width=1)
    return result
