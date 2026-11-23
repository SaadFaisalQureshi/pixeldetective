from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from modeling import LABELS, MODELS, embeddings, load_encoder, load_head, probabilities

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

st.set_page_config(page_title="PixelDetective")
st.title("PixelDetective")

key = st.selectbox(
    "Model",
    options=list(MODELS),
    index=list(MODELS).index("convnext"),
    format_func=lambda name: MODELS[name][0],
)
photo = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])


@st.cache_resource(max_entries=1, show_spinner=False)
def get_encoder(model_key):
    torch.set_num_threads(2)
    return load_encoder(model_key, "cpu")


if photo is not None and st.button("Analyze photo", type="primary"):
    try:
        with Image.open(photo) as image:
            if image.width * image.height > 20_000_000:
                st.error("Choose an image with 20 megapixels or fewer.")
                st.stop()

            with st.spinner("Loading model and analyzing photo…"):
                head = load_head(ARTIFACTS / f"{key}_head.json", key)
                features = embeddings(key, get_encoder(key), [image])
                scores = probabilities(features, head)[0]

        st.subheader(f"Prediction: {LABELS[int(scores.argmax())]}")
        for label, score in zip(LABELS, scores):
            st.write(f"{label}: {score:.1%}")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error("The image or model files could not be read. Check the app logs if this persists.")