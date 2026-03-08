"""
classifier.py

Trains a Voting Ensemble (SVM + Random Forest + XGBoost) with SMOTE
oversampling on filtered_static_features.tsv.

Pipeline:
  1. Load data, drop ID columns, separate features (X) / label (y)
  2. Impute missing values with column mean
  3. Standard-scale features
  4. SMOTE oversampling on training data to balance classes
  5. 80/20 stratified train/test split
  6. Train Voting Ensemble (soft voting)
  7. Evaluate on held-out test set
  8. Print accuracy, precision, recall, F1, AUC-ROC
"""

import os
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── 1. Load data ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "b2ai-voice", "features")
data_path = os.path.join(FEATURES_DIR, "filtered_static_features.tsv")

df = pd.read_csv(data_path, sep="\t")
print(f"Loaded {data_path}")
print(f"  Shape: {df.shape}")

# ── 2. Separate features and label ───────────────────────────────────────────
ID_COLS = ["participant_id", "session_id"]
cols_to_drop = [c for c in ID_COLS if c in df.columns]

X = df.drop(columns=cols_to_drop + ["label"])
y = df["label"]

print(f"  Features: {X.shape[1]}")
print(f"  Samples:  {len(y)}  (label=1: {y.sum()}, label=0: {(y == 0).sum()})")

# ── 3. Train / test split (80/20, stratified) ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {len(y_train)}  |  Test: {len(y_test)}")

# ── 4. Build Voting Ensemble pipeline ───────────────────────────────────────
ensemble = VotingClassifier(
    estimators=[
        ("svm", SVC(C=10, gamma="scale", random_state=42, probability=True)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)),
        ("xgb", XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=8,
            random_state=42, eval_metric="logloss", verbosity=0
        )),
    ],
    voting="soft",
)

pipe = ImbPipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("classifier", ensemble),
])

# ── 5. Train ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Training: Voting Ensemble (SVM + RF + XGBoost)")
print(f"{'='*60}")

pipe.fit(X_train, y_train)

# ── 6. Cross-validation score ───────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
print(f"\n  5-fold CV F1: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std():.4f})")

# ── 7. Evaluate on held-out test set ────────────────────────────────────────
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print(f"\n  Classification report (held-out test set):")
print(classification_report(y_test, y_pred, target_names=["Control", "Benign"]))

print(f"{'='*60}")
print(f"  FINAL RESULTS — Voting Ensemble (SVM + RF + XGBoost)")
print(f"{'='*60}")
print(f"  Accuracy:   {acc:.4f}")
print(f"  Precision:  {prec:.4f}")
print(f"  Recall:     {rec:.4f}")
print(f"  F1 Score:   {f1:.4f}")
print(f"  AUC-ROC:    {auc:.4f}")
print(f"  CV F1:      {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std():.4f})")
