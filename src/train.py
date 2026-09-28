import argparse, os, glob
import cv2
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from feature_extraction import extract_features, FEATURE_NAMES

EXTS = ("*.png","*.jpg","*.jpeg","*.bmp","*.webp")

def collect_images(root):
    rows = []
    for folder, label in [("control",0),("parkinson",1)]:
        paths=[]
        for e in EXTS:
            paths += glob.glob(os.path.join(root, folder, e))
        for p in sorted(paths):
            rows.append((p,label))
    return rows

def build_dataset(root):
    rows = []
    errors = []
    for path, label in collect_images(root):
        try:
            img = cv2.imread(path)
            feat = extract_features(img)
            rows.append([path, label, *feat])
        except Exception as e:
            errors.append((path, str(e)))
    df = pd.DataFrame(rows, columns=["path","label",*FEATURE_NAMES])
    return df, errors

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="data/processed/features.csv")
    args=ap.parse_args()

    os.makedirs(args.models, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df, errors = build_dataset(args.data)
    if len(df) < 10 or df["label"].nunique() < 2:
        raise SystemExit("Need at least 10 valid images and both control/parkinson classes.")

    df.to_csv(args.out,index=False)
    if errors:
        print("Skipped files:")
        for x in errors: print(x)

    X=df[FEATURE_NAMES].values
    y=df.label.values

    Xtr,Xte,ytr,yte=train_test_split(
        X,y,test_size=0.20,stratify=y,random_state=42
    )

    models={
        "svm": Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
            ("model",SVC(kernel="rbf",probability=True,class_weight="balanced",random_state=42))
        ]),
        "random_forest": Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("model",RandomForestClassifier(
                n_estimators=400,class_weight="balanced",random_state=42,n_jobs=-1
            ))
        ]),
        "xgboost": Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("model",XGBClassifier(
                n_estimators=300,max_depth=4,learning_rate=0.04,
                subsample=0.85,colsample_bytree=0.85,
                eval_metric="logloss",random_state=42,n_jobs=4
            ))
        ])
    }

    metrics=[]
    for name,model in models.items():
        model.fit(Xtr,ytr)
        pred=model.predict(Xte)
        prob=model.predict_proba(Xte)[:,1]
        row={
            "model":name,
            "accuracy":accuracy_score(yte,pred),
            "precision":precision_score(yte,pred,zero_division=0),
            "recall":recall_score(yte,pred,zero_division=0),
            "f1":f1_score(yte,pred,zero_division=0),
            "roc_auc":roc_auc_score(yte,prob) if len(np.unique(yte))==2 else np.nan
        }
        metrics.append(row)
        print("\n",name)
        print(classification_report(yte,pred,target_names=["Control","Parkinson"],zero_division=0))
        joblib.dump(model,os.path.join(args.models,f"{name}.joblib"))

    mdf=pd.DataFrame(metrics)
    mdf.to_csv(os.path.join(args.models,"metrics.csv"),index=False)
    joblib.dump(FEATURE_NAMES,os.path.join(args.models,"feature_names.joblib"))
    print("\nSaved models to:",args.models)
    print(mdf.to_string(index=False))

if __name__=="__main__":
    main()
