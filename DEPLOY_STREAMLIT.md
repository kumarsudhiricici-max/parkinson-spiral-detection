# Deploy on Streamlit Community Cloud from mobile

## 1. Create the GitHub repository

Create a public GitHub repository named:

`parkinson-spiral-detection`

Upload the contents of this project.

Do not upload patient/private datasets.

## 2. Train the model

Training requires your spiral dataset and is normally done on a computer or cloud notebook.

After training, the generated model files are:

- `models/svm.joblib`
- `models/random_forest.joblib`
- `models/xgboost.joblib`

For the current app to make predictions, at least one trained model must be available in `models/`.

## 3. Deploy

Open Streamlit Community Cloud and connect your GitHub account.

Choose:

- Repository: `YOUR_USERNAME/parkinson-spiral-detection`
- Branch: `main`
- Main file: `app.py`

Then deploy.

## Important

The application is a research prototype. It must not be presented as a medical diagnostic device. A prediction from a spiral image does not establish that someone has Parkinson's disease.
