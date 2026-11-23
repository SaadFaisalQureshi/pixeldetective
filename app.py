"""PixelDetective Gradio web interface."""
import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch

from modeling import LABELS, MODELS, embeddings, load_encoder, load_head, probabilities

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(torch.get_num_threads(), 4))


@lru_cache(maxsize=2)
def encoder_for(key):
    return load_encoder(key, DEVICE)


def selected_model():
    path = ARTIFACTS / "metrics.json"
    if path.is_file():
        return json.loads(path.read_text())["selected_by_validation"]
    return "vit"


def analyze(image, key=None):
    if image is None:
        raise gr.Error("Upload an image to analyze.")
    if image.width * image.height > 20_000_000:
        raise gr.Error("Choose an image with 20 megapixels or fewer.")
    key = key or selected_model()
    if key not in MODELS:
        raise gr.Error("Choose a model from the list.")
    path = ARTIFACTS / f"{key}_head.json"
    if not path.is_file():
        raise gr.Error("This model's trained classifier is missing.")
    head = load_head(path, key)
    features = embeddings(key, encoder_for(key), [image])
    return dict(zip(LABELS, probabilities(features, head)[0].tolist()))


def benchmark_table():
    path = ARTIFACTS / "metrics.json"
    if not path.is_file():
        return "Evaluation results appear here after running the notebook."
    report = json.loads(path.read_text())
    selected = report["models"][report["selected_by_validation"]]["display_name"]
    lines = ["**CASIA v2 held-out test results**", "",
             "| Model | Accuracy | Macro F1 | Manipulated recall | JPEG-only macro F1 |", "|---|---:|---:|---:|---:|"]
    for model in report["models"].values():
        result = model["test"]
        jpeg = result.get("jpeg_only", {}).get("macro_f1")
        jpeg_display = f"{jpeg:.3f}" if jpeg is not None else "—"
        recall = result["classification_report"]["Manipulated"]["recall"]
        lines.append(f"| {model['display_name']} | {result['accuracy']:.2%} | {result['macro_f1']:.3f} | {recall:.1%} | {jpeg_display} |")
    lines.extend(["", f"Model selected on validation macro F1: **{selected}**"])
    return "\n".join(lines)


with gr.Blocks(title="PixelDetective", theme=gr.themes.Soft(primary_hue="teal"),
               css=".gradio-container {max-width: 1050px !important; margin: auto !important;}") as demo:
    gr.Markdown("# PixelDetective\nUpload a photograph and check it for signs of manipulation.")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", image_mode="RGB", sources=["upload"], label="Photograph")
            model_choice = gr.Dropdown(
                choices=[(name, key) for key, (name, _, _) in MODELS.items()],
                value=selected_model(), label="Model",
            )
            button = gr.Button("Analyze photograph", variant="primary")
        with gr.Column():
            result = gr.Label(label="Prediction scores", num_top_classes=2)
    gr.Markdown("Scores reflect model estimates. Performance can change on images outside CASIA v2.")
    with gr.Accordion("Held-out evaluation", open=False):
        gr.Markdown(benchmark_table())
    button.click(analyze, inputs=[image, model_choice], outputs=result, api_name="analyze")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(max_file_size="12mb")
