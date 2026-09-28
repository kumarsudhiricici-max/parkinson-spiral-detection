# 🌀 Parkinson Spiral Detection

A research prototype that analyzes spiral handwriting images using computer vision and machine learning.

## Features

- Spiral image preprocessing with OpenCV
- Handwriting/spiral geometry features
- SVM classifier
- Random Forest classifier
- XGBoost classifier
- Streamlit web interface
- Feature visualization
- Probability output

## Project flow

```text
Spiral image
     ↓
OpenCV preprocessing
     ↓
Feature extraction
     ↓
SVM / Random Forest / XGBoost
     ↓
Class prediction
```

## Dataset

Place images into:

```text
data/raw/control/
data/raw/parkinson/
```

The folder determines the class.

**Do not upload identifiable patient data to a public GitHub repository.**

## Local setup

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

## Streamlit deployment

See `DEPLOY_STREAMLIT.md`.

## Medical disclaimer

This is an experimental research/educational project. It is not a medical device and does not diagnose Parkinson's disease. Model performance depends heavily on dataset quality, image quality, population, and subject-level validation.

For research evaluation, use subject-level train/test separation when multiple drawings belong to the same participant.
