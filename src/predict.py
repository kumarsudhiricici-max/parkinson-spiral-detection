import os, cv2, joblib
import numpy as np
from .feature_extraction import extract_features, FEATURE_NAMES

def load_model(model_path):
    return joblib.load(model_path)

def predict_image(image, model):
    feat = extract_features(image).reshape(1,-1)
    pred = int(model.predict(feat)[0])
    prob = float(model.predict_proba(feat)[0,1])
    return pred, prob, feat[0]
