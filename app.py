import os, sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from feature_extraction import extract_features, preprocess_for_display, FEATURE_NAMES

st.set_page_config(page_title="Parkinson Spiral Analysis", page_icon="🌀", layout="wide")

MODEL_DIR="models"

@st.cache_resource
def load_models():
    out={}
    for name in ["svm","random_forest","xgboost"]:
        p=os.path.join(MODEL_DIR,f"{name}.joblib")
        if os.path.exists(p):
            out[name]=joblib.load(p)
    return out

st.title("🌀 Parkinson's Spiral Drawing Analysis")
st.caption("Research prototype — not a medical diagnosis.")

with st.sidebar:
    st.header("Model")
    models=load_models()
    if not models:
        st.error("No trained model found. Run: python src/train.py")
        st.stop()
    model_name=st.selectbox("Choose model",list(models.keys()))
    st.info("Use a clear spiral drawing on a plain background. Keep the entire drawing visible.")

uploaded=st.file_uploader("Upload a spiral drawing",type=["png","jpg","jpeg","bmp","webp"])

if uploaded:
    pil=Image.open(uploaded).convert("RGB")
    img=np.array(pil)
    bgr=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    col1,col2=st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(pil,use_container_width=True)

    try:
        feat=extract_features(bgr)
        mask=preprocess_for_display(bgr)
        with col2:
            st.subheader("Processed drawing")
            st.image(mask,use_container_width=True)

        model=models[model_name]
        pred=int(model.predict(feat.reshape(1,-1))[0])
        prob=float(model.predict_proba(feat.reshape(1,-1))[0,1])

        st.divider()
        if pred==1:
            st.warning(f"Model output: Parkinson class — estimated model probability {prob*100:.1f}%")
        else:
            st.success(f"Model output: Control class — estimated Parkinson-class probability {prob*100:.1f}%")

        st.warning(
            "This result cannot diagnose Parkinson's disease. Image quality, drawing method, "
            "dataset bias and person-to-person differences can substantially affect the prediction."
        )

        with st.expander("Extracted features"):
            st.dataframe(pd.DataFrame({"feature":FEATURE_NAMES,"value":feat}),use_container_width=True)

    except Exception as e:
        st.error(f"Could not analyze this image: {e}")

st.divider()
st.subheader("Optional: draw directly")
st.write("For direct drawing, install streamlit-drawable-canvas. The upload workflow above is the recommended starting point.")
