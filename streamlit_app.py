from hashlib import sha256
from html import escape
from io import BytesIO
import json
from pathlib import Path
from threading import Lock

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from modeling import LABELS, MODELS, embeddings, load_encoder, load_head, probabilities
from score_map import occlusion_scores, overlay

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
MAX_PIXELS = 20_000_000
st.set_page_config(page_title="PixelDetective", layout="wide")


st.markdown("""
<style>
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] button,
[data-testid="stSelectbox"] [data-baseweb="select"] {
    font-size: 1.15rem;
}
[data-testid="stCaptionContainer"] p { color: #283548; font-size: 1rem; }
[data-testid="stAlert"] { border: 2px solid #34435A; }
[data-testid="stAlert"] p { font-weight: 600; }
.projection-table { overflow-x: auto; }
.projection-table table { width: 100%; border-collapse: collapse; font-size: 1.1rem; }
.projection-table th, .projection-table td {
    border: 1px solid #637188;
    color: #111827;
    padding: .65rem .75rem;
    text-align: left;
}
.projection-table th { background: #122441; color: #FFFFFF; }
.projection-table tr:nth-child(even) td { background: #EEF2F8; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def inference_lock():
    torch.set_num_threads(2)
    return Lock()


@st.cache_resource(max_entries=1, show_spinner=False)
def get_encoder(model_key):
    return load_encoder(model_key, "cpu")


def predict(image, key):
    head = load_head(ARTIFACTS / f"{key}_head.json", key)
    with inference_lock():
        scores = probabilities(embeddings(key, get_encoder(key), [image]), head)[0]
    return {
        "Model": MODELS[key][0],
        "Prediction": LABELS[int(scores.argmax())],
        **{label: float(score) for label, score in zip(LABELS, scores)},
    }


def sensitivity_preview(image, key):
    head = load_head(ARTIFACTS / f"{key}_head.json", key)
    with inference_lock():
        encoder = get_encoder(key)

        def score(images):
            return probabilities(embeddings(key, encoder, images), head)[:, 1]

        grid = occlusion_scores(image, score)
    return overlay(image, grid), grid


def percent_table(rows, columns=()):
    if not rows:
        return
    headings = list(dict.fromkeys(heading for row in rows for heading in row))
    header = "".join(f'<th scope="col">{escape(name)}</th>' for name in headings)
    body = []
    for row in rows:
        cells = []
        for heading in headings:
            value = row.get(heading)
            if value is None:
                value = "—"
            elif heading in columns:
                value = f"{value:.1%}"
            cells.append(f"<td>{escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        f'<div class="projection-table"><table><thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def show_benchmark(report):
    st.subheader("CASIA v2 held-out results")
    models = report["models"]
    selected = report["selected_by_validation"]
    st.caption(f"{models[selected]['display_name']} is the default because it scored highest on validation macro F1.")
    rows = []
    for key, entry in models.items():
        test = entry["test"]
        rows.append({
            "Model": entry["display_name"],
            "Default": "Yes" if key == selected else "—",
            "Validation macro F1": round(entry["validation_macro_f1"], 3),
            "Test accuracy": test["accuracy"],
            "Test macro F1": round(test["macro_f1"], 3),
            "Manipulated recall": test["classification_report"]["Manipulated"]["recall"],
            "JPEG-only macro F1": round(test["jpeg_only"]["macro_f1"], 3) if test.get("jpeg_only") else None,
        })
    baseline = report["majority_baseline_test"]
    rows.append({
        "Model": "Majority baseline", "Default": "—", "Test accuracy": baseline["accuracy"],
        "Test macro F1": round(baseline["macro_f1"], 3),
        "Manipulated recall": baseline["classification_report"]["Manipulated"]["recall"],
    })
    percent_table(rows, ["Test accuracy", "Manipulated recall"])
    matrix_key = st.selectbox("Confusion matrix", list(models), index=list(models).index(selected),
                              format_func=lambda key: models[key]["display_name"])
    matrix = models[matrix_key]["test"]["confusion_matrix"]
    percent_table([{"Actual": label, **{f"Predicted {name.lower()}": count
              for name, count in zip(LABELS, row)}} for label, row in zip(LABELS, matrix)])
    st.caption("Macro F1 gives both classes equal weight. Manipulated recall measures how many edited photos a model catches.")


report = json.loads((ARTIFACTS / "metrics.json").read_text())
default = report["selected_by_validation"]
st.title("PixelDetective")
st.caption("Image tampering analysis across four models")
analysis, benchmark = st.tabs(["Analyze photo", "Model benchmark"])

with analysis:
    controls, results = st.columns([1, 2], gap="large")
    with controls:
        mode = st.radio("Analysis mode", ["Single model", "Compare models"], horizontal=True)
        keys = list(MODELS)
        if mode == "Single model":
            keys = [st.selectbox("Model", keys, index=keys.index(default),
                                 format_func=lambda key: MODELS[key][0])]
        photo = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        map_requested = mode == "Single model" and st.checkbox("Show score sensitivity map")
        launch = st.button("Analyze photo", type="primary", disabled=photo is None)

    with results:
        if photo is None:
            st.info("Upload a JPG or PNG to start.")
        else:
            data = photo.getvalue()
            signature = (sha256(data).hexdigest(), tuple(keys), map_requested)
            try:
                with Image.open(BytesIO(data)) as source:
                    if source.width * source.height > MAX_PIXELS:
                        raise ValueError("Choose an image with 20 megapixels or fewer.")
                    image = ImageOps.exif_transpose(source).convert("RGB")
                preview = image.copy()
                preview.thumbnail((900, 480))
                st.image(preview, caption=f"{image.width} × {image.height} pixels")
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as error:
                st.error(str(error) if isinstance(error, ValueError) else "Choose a readable JPG or PNG.")
            else:
                if launch:
                    rows, failed = [], []
                    score_image = None
                    map_failed = False
                    with st.status("Analyzing photo…", expanded=False) as status:
                        for key in keys:
                            st.write(MODELS[key][0])
                            try:
                                rows.append(predict(image, key))
                            except (OSError, RuntimeError, ValueError):
                                failed.append(MODELS[key][0])
                        if map_requested and rows:
                            try:
                                score_image = sensitivity_preview(preview, keys[0])
                            except (OSError, RuntimeError, ValueError):
                                map_failed = True
                        status.update(label="Analysis complete" if not failed else "Analysis incomplete",
                                      state="complete" if not failed else "error")
                    st.session_state["analysis_result_v2"] = signature, rows, failed, score_image, map_failed
                previous = st.session_state.get("analysis_result_v2")
                if previous and previous[0] == signature:
                    _, rows, failed, score_image, map_failed = previous
                    if failed:
                        st.error("Could not run " + ", ".join(failed) + ". Try Analyze photo again.")
                    if rows:
                        labels = {row["Prediction"] for row in rows}
                        if len(labels) > 1:
                            st.warning("Models disagree on this photo.")
                        elif not failed:
                            prefix = "All models predict" if len(rows) > 1 else "Prediction"
                            st.subheader(f"{prefix} · {rows[0]['Prediction']}")
                        percent_table(rows, LABELS)
                        st.caption("Scores are uncalibrated model estimates. Agreement can still be wrong.")
                    if map_failed:
                        st.error("The score sensitivity map could not be generated.")
                    if score_image is not None:
                        st.subheader("Score sensitivity map")
                        rendered, grid = score_image
                        largest_drop = max(0.0, float(grid.max()))
                        largest_rise = max(0.0, -float(grid.min()))
                        if max(largest_drop, largest_rise) < 0.001:
                            st.info("Covering any tile changed the manipulation score by less than 0.1 percentage points.")
                        else:
                            st.image(rendered)
                            st.caption(
                                f"− Score fell (orange, up to {100 * largest_drop:.2f} pp). "
                                f"+ Score rose (blue, up to {100 * largest_rise:.2f} pp). "
                                "Color intensity is relative to this photo. The map shows score sensitivity."
                            )

with benchmark:
    show_benchmark(report)
    for filename in ("robustness_casia.json", "robustness_external.json"):
        path = ARTIFACTS / filename
        if path.is_file():
            experiment = json.loads(path.read_text())
            st.subheader(f"Robustness · {experiment['dataset']}")
            st.caption(f"{experiment['sample_count']:,} labeled images per variant")
            percent_table([
                {"Model": MODELS[key][0], "Variant": name, "Accuracy": result["accuracy"],
                 "Macro F1": round(result["macro_f1"], 3),
                 "Manipulated recall": result["classification_report"]["Manipulated"]["recall"]}
                for key, variants in experiment["models"].items() for name, result in variants.items()
            ], ["Accuracy", "Manipulated recall"])

    resolution = ARTIFACTS / "resolution.json"
    if resolution.is_file():
        experiment = json.loads(resolution.read_text())
        st.subheader("ConvNeXt input resolution")
        st.caption(f"{experiment['dataset']} · Embedding time on {experiment['device']}")
        percent_table([
            {"Input": f"{side} × {side}", "Validation macro F1": round(experiment[side]["validation_macro_f1"], 3),
             "Accuracy": experiment[side]["test"]["accuracy"],
             "Test macro F1": round(experiment[side]["test"]["macro_f1"], 3),
             "Median embedding time (ms)": round(experiment[side]["median_embedding_ms"], 1)}
            for side in ("128", "224")
        ], ["Accuracy"])

    localization = ARTIFACTS / "localization.json"
    if localization.is_file():
        experiment = json.loads(localization.read_text())
        st.subheader("Score map overlap with edit masks")
        st.metric("Mean IoU", f"{experiment['mean_iou']:.3f}")
        st.caption(f"{experiment['sample_count']:,} labeled masks · Score-drop threshold {experiment['threshold_score_drop']:.2f}")
