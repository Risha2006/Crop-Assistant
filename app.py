# streamlit_app.py
import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib

# TF imports lazy (avoid import cost if you only use crop rec)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------- CONFIG: tweak if your paths differ ----------
DISEASE_MODEL_PATH = r"C:\Users\Risha H M\AI_Crop_System\crop_disease_detection\notebooks\crop_disease_model.h5"
DISEASE_TRAIN_DIR  = r"C:\Users\Risha H M\AI_Crop_System\crop_disease_detection\data\split\train"  # to derive class names

CROP_MODEL_PATH    = r"C:\Users\Risha H M\AI_Crop_System\crop_recommendation\notebooks\crop_recommendation.pkl"
CROP_META_PATH     = r"C:\Users\Risha H M\AI_Crop_System\crop_recommendation\notebooks\crop_recommendation_metadata.json"

IMG_SIZE = (224, 224)  # must match your training
TOP_K = 3
# --------------------------------------------------------

st.set_page_config(page_title="AI Crop Assistant", page_icon="🌾", layout="wide")

# ---------- Helper: derive disease classes in same order as training ----------
def load_disease_classes(train_dir: str):
    if not os.path.isdir(train_dir):
        return []
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes = sorted(classes)  # Keras flow_from_directory uses sorted order
    return classes

# ---------- Cache: load models once ----------
@st.cache_resource
def load_disease_model_and_classes():
    classes = load_disease_classes(DISEASE_TRAIN_DIR)
    if not os.path.exists(DISEASE_MODEL_PATH):
        return None, classes
    model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    return model, classes

@st.cache_resource
def load_crop_model_and_meta():
    if not os.path.exists(CROP_MODEL_PATH) or not os.path.exists(CROP_META_PATH):
        return None, None
    model = joblib.load(CROP_MODEL_PATH)
    with open(CROP_META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta

disease_model, disease_classes = load_disease_model_and_classes()
crop_model, crop_meta = load_crop_model_and_meta()

# ---------- UI layout ----------
st.title("🌾 AI-Powered Crop Assistant")
st.caption("Leaf disease diagnosis + data-driven crop recommendation")

tab1, tab2 = st.tabs(["🩺 Disease Diagnosis", "🧪 Crop Recommendation"])

# =========================
# TAB 1: Disease Diagnosis
# =========================
with tab1:
    st.subheader("Upload a leaf image")
    colL, colR = st.columns([1,1])

    with colL:
        img_file = st.file_uploader("Choose an image (leaf close-up)", type=["jpg", "jpeg", "png"])
        info = st.empty()

        if not os.path.exists(DISEASE_MODEL_PATH):
            st.warning(f"Model not found at: `{DISEASE_MODEL_PATH}`. Place your .h5 file there.")
        if not os.path.isdir(DISEASE_TRAIN_DIR):
            st.warning(f"Training folder not found: `{DISEASE_TRAIN_DIR}` (used to infer class names).")
        elif len(disease_classes) == 0:
            st.warning("No class folders found in training directory. Class names may be unavailable.")

    with colR:
        if img_file is not None and disease_model is not None:
            # Read & preprocess
            image = Image.open(img_file).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)

            img = image.resize(IMG_SIZE)
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            # you trained with rescale=1./255; preprocess_input is fine on [0,255] too,
            # but since you used rescale, we do manual /255 here:
            arr = arr / 255.0

            with st.spinner("Running inference..."):
                preds = disease_model.predict(arr)
                preds = np.squeeze(preds)  # shape: (num_classes,)

            # Top-K
            top_idx = np.argsort(preds)[::-1][:TOP_K]
            if disease_classes and len(disease_classes) == preds.shape[0]:
                top_labels = [disease_classes[i] for i in top_idx]
            else:
                # Fallback labels if classes not available
                top_labels = [f"class_{i}" for i in top_idx]

            st.markdown("### Results")
            for rank, (idx, label) in enumerate(zip(top_idx, top_labels), start=1):
                st.write(f"**{rank}. {label}** — {preds[idx]*100:.2f}%")

            # Best prediction highlight
            best_label = top_labels[0]
            best_conf = preds[top_idx[0]] * 100
            st.success(f"Predicted: **{best_label}** ({best_conf:.2f}% confidence)")

            # Optional: simple advisory text stub
            with st.expander("Advisory (template)"):
                st.write("• Confirm visually with multiple leaves.")
                st.write("• Remove heavily infected leaves.")
                st.write("• Consider recommended fungicide/biocontrol based on local guidelines.")
                st.write("• Ensure proper field sanitation and crop rotation.")

# =========================
# TAB 2: Crop Recommendation
# =========================
with tab2:
    st.subheader("Enter soil & weather values")
    if crop_model is None or crop_meta is None:
        st.warning("Crop recommendation model/metadata not found. "
                   f"Expected `{CROP_MODEL_PATH}` and `{CROP_META_PATH}`.")
    else:
        feats = crop_meta.get("features", ["N","P","K","temperature","humidity","ph","rainfall"])
        classes = crop_meta.get("classes", [])

        # Two-column input form
        c1, c2 = st.columns(2)
        with c1:
            N  = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0, value=90.0)
            P  = st.number_input("Phosphorus (P)", min_value=0.0, step=1.0, value=42.0)
            K  = st.number_input("Potassium (K)", min_value=0.0, step=1.0, value=43.0)
            temp = st.number_input("Temperature (°C)", min_value=0.0, step=0.5, value=20.0)
        with c2:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.5, value=80.0)
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1, value=6.5)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0, value=200.0)

        btn = st.button("Recommend Crop")

        if btn:
            x = np.array([[N, P, K, temp, humidity, ph, rainfall]], dtype=float)
            with st.spinner("Scoring..."):
                pred = crop_model.predict(x)[0]
                proba = crop_model.predict_proba(x)[0]
                top_idx = np.argsort(proba)[::-1][:TOP_K]
                top3 = [(crop_model.classes_[i], float(proba[i])) for i in top_idx]

            st.success(f"Recommended crop: **{pred}**")
            st.markdown("**Top-3 options:**")
            for i, (cls, p) in enumerate(top3,  start=1):
                st.write(f"{i}. {cls} — {p*100:.2f}%")

            with st.expander("Debug info"):
                st.json({
                    "features_order": feats,
                    "input": [N, P, K, temp, humidity, ph, rainfall],
                    "best_params": crop_meta.get("best_params", {})
                })

# Footer
st.caption("Built with TensorFlow + scikit-learn + Streamlit.")
