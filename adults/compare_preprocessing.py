import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, classification_report
)
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "features/filtered_static_features.tsv"

# ── Load Data ──
df = pd.read_csv(DATA_PATH, sep='\t')
meta_cols = ['participant_id', 'session_id', 'task_name', 'transcription', 'label']
feature_cols = [c for c in df.columns if c not in meta_cols]
y = df['label']
groups = df['participant_id']
imbalance_ratio = len(y[y == 0]) / max(len(y[y == 1]), 1)

print(f"Dataset: {len(df)} samples, {len(feature_cols)} features, {df['participant_id'].nunique()} participants")
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1\n")


# ── Preprocessing Strategies ──

def preprocess_minimal(df, feature_cols):
    """Strategy A: Minimal - median impute missing values, let CatBoost handle the rest."""
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X, "Minimal (median impute only)"


def preprocess_log_transform(df, feature_cols):
    """Strategy B: Log-transform heavily skewed features + jitter/shimmer."""
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Log-transform jitter/shimmer columns
    jitter_shimmer = [c for c in X.columns if any(k in c.lower() for k in ['jitter', 'shimmer'])]
    for col in jitter_shimmer:
        X[col] = np.log10(X[col].clip(lower=1e-6))

    # Log-transform highly skewed stddevNorm columns (skew > 10)
    skewed_cols = [c for c in X.columns if 'stddevNorm' in c]
    for col in skewed_cols:
        vals = X[col].dropna()
        if vals.skew() > 10:
            X[col] = np.log1p(X[col].clip(lower=0))

    X = X.fillna(X.median(numeric_only=True))
    return X, "Log-transform (skewed + jitter/shimmer)"


def preprocess_aggressive(df, feature_cols):
    """Strategy C: Log-transform + winsorize outliers + drop high-missingness columns."""
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns with >15% missing
    missing_pct = X.isnull().mean()
    drop_cols = missing_pct[missing_pct > 0.15].index.tolist()
    X = X.drop(columns=drop_cols)

    # Log-transform jitter/shimmer
    jitter_shimmer = [c for c in X.columns if any(k in c.lower() for k in ['jitter', 'shimmer'])]
    for col in jitter_shimmer:
        X[col] = np.log10(X[col].clip(lower=1e-6))

    # Log-transform highly skewed stddevNorm columns
    skewed_cols = [c for c in X.columns if 'stddevNorm' in c]
    for col in skewed_cols:
        vals = X[col].dropna()
        if vals.skew() > 10:
            X[col] = np.log1p(X[col].clip(lower=0))

    # Winsorize to 1st/99th percentile
    for col in X.select_dtypes(include='number').columns:
        p01, p99 = X[col].quantile(0.01), X[col].quantile(0.99)
        X[col] = X[col].clip(lower=p01, upper=p99)

    X = X.fillna(X.median(numeric_only=True))
    return X, f"Aggressive (log + winsorize + dropped {len(drop_cols)} cols)"


# ── Evaluation ──

def evaluate_pipeline(X, y, groups, strategy_name):
    gkf = GroupKFold(n_splits=5)
    metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values
        groups_train = groups.iloc[train_idx].values

        # SMOTE on training data only (respecting fold boundary)
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

        # CatBoost with class weights on top of SMOTE
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            auto_class_weights='Balanced',
            eval_metric='AUC',
            random_seed=42,
            verbose=0,
        )
        model.fit(X_train_sm, y_train_sm)

        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs > 0.5).astype(int)

        metrics.append({
            'fold': fold + 1,
            'auc_roc': roc_auc_score(y_val, val_probs),
            'avg_precision': average_precision_score(y_val, val_probs),
            'precision': precision_score(y_val, val_preds),
            'recall': recall_score(y_val, val_preds),
            'f1': f1_score(y_val, val_preds),
        })

    results = pd.DataFrame(metrics)
    return results


# ── Run All Three ──

strategies = [preprocess_minimal, preprocess_log_transform, preprocess_aggressive]
all_results = {}

for strategy_fn in strategies:
    X, name = strategy_fn(df, feature_cols)
    print(f"{'=' * 70}")
    print(f"Strategy: {name}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"{'=' * 70}")

    results = evaluate_pipeline(X, y, groups, name)

    for _, row in results.iterrows():
        print(f"  Fold {int(row['fold'])}: AUC={row['auc_roc']:.4f}  "
              f"Prec={row['precision']:.4f}  Rec={row['recall']:.4f}  F1={row['f1']:.4f}")

    mean = results.mean(numeric_only=True)
    std = results.std(numeric_only=True)
    print(f"\n  MEAN:   AUC={mean['auc_roc']:.4f}  Prec={mean['precision']:.4f}  "
          f"Rec={mean['recall']:.4f}  F1={mean['f1']:.4f}  AvgPrec={mean['avg_precision']:.4f}")
    print(f"  STD:    AUC={std['auc_roc']:.4f}  Prec={std['precision']:.4f}  "
          f"Rec={std['recall']:.4f}  F1={std['f1']:.4f}  AvgPrec={std['avg_precision']:.4f}")
    print()

    all_results[name] = mean

# ── Summary Comparison ──
print(f"\n{'=' * 70}")
print("SUMMARY COMPARISON")
print(f"{'=' * 70}")
summary = pd.DataFrame(all_results).T
summary = summary[['auc_roc', 'avg_precision', 'precision', 'recall', 'f1']]
summary.columns = ['AUC-ROC', 'Avg Precision', 'Precision', 'Recall', 'F1']
print(summary.round(4).to_string())
print()
best_prec = summary['Precision'].idxmax()
best_auc = summary['AUC-ROC'].idxmax()
print(f"Best Precision:  {best_prec}")
print(f"Best AUC-ROC:    {best_auc}")
