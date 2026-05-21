"""
Streamlit UI for the HAM10000 binary classifier from cv_skinCancer.ipynb.

Save weights after training:
    model.save('skin_cancer_model.keras')

Run:
    streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
import streamlit as st
from PIL import Image

from cv_features import (
    analyze_image_quality,
    blend_heatmap,
    edge_visualization,
    grad_cam,
    preprocess_numpy_rgb,
    rgb_histogram_figure,
)
from model_utils import (
    LABELS,
    load_trained_model,
    preprocess_pil,
    predict_proba_malignant,
)


@st.cache_resource
def cached_model():
    return load_trained_model()


st.set_page_config(page_title="Skin lesion classifier", page_icon="🔬", layout="wide")

st.title("Skin lesion classifier (demo)")
st.caption(
    "MobileNetV2 binary model: Benign (Nevus) vs Malignant (Melanoma). "
    "Not a medical device."
)

try:
    model = cached_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

uploaded = st.file_uploader("Upload a skin lesion image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is None:
    st.info("Upload an image to run inference and computer vision analysis.")
    st.stop()

image = Image.open(uploaded)
batch = preprocess_pil(image)
p_mal = predict_proba_malignant(model, batch)

tab_pred, tab_cv = st.tabs(["Prediction", "Computer vision"])

with tab_pred:
    st.image(image, caption="Original upload", use_container_width=True)
    p_ben = 1.0 - p_mal

    st.subheader("Predicted probabilities")
    st.write(f"**{LABELS[1]}** — {100 * p_mal:.1f}%")
    st.progress(min(max(p_mal, 0.0), 1.0))
    st.write(f"**{LABELS[0]}** — {100 * p_ben:.1f}%")
    st.progress(min(max(p_ben, 0.0), 1.0))

    if p_mal >= 0.5:
        st.warning(f"Model threshold (0.5): **{LABELS[1]}**")
    else:
        st.success(f"Model threshold (0.5): **{LABELS[0]}**")

with tab_cv:
    st.subheader("Image quality")
    q = analyze_image_quality(image)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resolution", f"{q.width} × {q.height}")
    c2.metric("Megapixels", f"{q.megapixels:.2f}")
    c3.metric("Laplacian variance (sharpness)", f"{q.blur_variance:.0f}")
    c4.metric("Mean luminance", f"{q.mean_luminance:.0f}")

    if min(q.width, q.height) < 128:
        st.warning(
            "Very small image: model input is upsampled to 224×224; "
            "fine detail may be unreliable."
        )
    if q.blur_variance < 30:
        st.warning(
            "Low Laplacian variance often indicates blur or heavy compression. "
            "Interpret predictions cautiously."
        )

    st.subheader("Preprocessing (matches training)")
    prep = preprocess_numpy_rgb(image)
    prep_vis = Image.fromarray(np.clip(prep * 255.0, 0, 255).astype(np.uint8))
    pe1, pe2 = st.columns(2)
    with pe1:
        st.caption("224 × 224, RGB, scaled ÷255 (same as validation generator)")
        st.image(prep_vis, use_container_width=True)

    st.subheader("Edges (Canny)")
    ec1, ec2 = st.columns([2, 1])
    with ec2:
        lo = st.slider("Canny low", 10, 200, 80, key="canny_lo")
        hi = st.slider("Canny high", 20, 250, 160, key="canny_hi")
    edges_img = edge_visualization(image, low=lo, high=max(hi, lo + 1))
    with ec1:
        st.caption("Edges emphasized in green on the 224 px resize")
        st.image(edges_img, use_container_width=True)

    st.subheader("Color distribution")
    fig_hist = rgb_histogram_figure(image)
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    st.subheader("Grad-CAM (class-discriminative regions)")
    st.caption(
        "Approximate saliency on the last convolutional layer for the malignant "
        "(positive class) logit — useful for debugging, not clinical explanation."
    )
    try:
        heatmap = grad_cam(model, batch)
        overlay = blend_heatmap(image, heatmap, alpha=0.42)
        og1, og2 = st.columns(2)
        with og1:
            st.caption("Heatmap only (224 px)")
            hm_rgb = colormaps["jet"](heatmap)[..., :3]
            hm_img = Image.fromarray(np.clip(hm_rgb * 255, 0, 255).astype(np.uint8))
            st.image(hm_img, use_container_width=True)
        with og2:
            st.caption("Overlay on resized RGB")
            st.image(overlay, use_container_width=True)
    except Exception as exc:
        st.error(f"Grad-CAM failed: {exc}")

st.markdown(
    "**Disclaimer:** For coursework or research only. "
    "Do not use for diagnosis; see a qualified clinician."
)
