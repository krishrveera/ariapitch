import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "features/filtered_static_features.tsv"
MODEL_PATH = "benign_lesion_model.joblib"

# ── 1. Load Data ──
df = pd.read_csv(DATA_PATH, sep='\t')
meta_cols = ['participant_id', 'session_id', 'task_name', 'transcription', 'label']
feature_cols = [c for c in df.columns if c not in meta_cols]

def get_task_category(task):
    for prefix in ['cape-v', 'diadochokinesis', 'free-speech', 'harvard-sentences',
                    'maximum-phonation', 'prolonged-vowel', 'rainbow-passage',
                    'respiration-and-cough', 'caterpillar', 'loudness', 'picture-description',
                    'story-recall', 'glides', 'animal-fluency', 'open-response',
                    'breath-sounds', 'cinderella', 'productive-vocab', 'random-item',
                    'voluntary-cough', 'word-color-stroop']:
        if task.startswith(prefix):
            return prefix
    return task

# ── 2. Filter to high-signal tasks ──
df['task_category'] = df['task_name'].apply(get_task_category)
HIGH_SIGNAL_TASKS = ['productive-vocab', 'harvard-sentences', 'breath-sounds', 'animal-fluency',
                     'voluntary-cough', 'maximum-phonation', 'prolonged-vowel', 'rainbow-passage',
                     'open-response', 'free-speech', 'caterpillar', 'picture-description',
                     'glides', 'cape-v', 'story-recall']
df_hs = df[df['task_category'].isin(HIGH_SIGNAL_TASKS)].copy()

print(f"Loaded {len(df)} total recordings, filtered to {len(df_hs)} high-signal task recordings")
print(f"Participants: {df_hs['participant_id'].nunique()}")

# ── 3. Aggregate per-participant mean+std ──
agg_mean = df_hs.groupby('participant_id')[feature_cols].mean()
agg_std = df_hs.groupby('participant_id')[feature_cols].std().fillna(0)
agg_mean.columns = [c + '_mean' for c in feature_cols]
agg_std.columns = [c + '_std' for c in feature_cols]
agg = pd.concat([agg_mean, agg_std], axis=1)
agg['label'] = df_hs.groupby('participant_id')['label'].first()
feat_all = [c for c in agg.columns if c != 'label']
X_all = agg[feat_all].replace([np.inf, -np.inf], np.nan).fillna(agg[feat_all].median(numeric_only=True))
y = agg['label']
groups = agg.index

print(f"Aggregated to {len(X_all)} participants, {X_all.shape[1]} features (mean+std)")
print(f"Label distribution: {y.value_counts().to_dict()}")

# ── 4. Feature selection: get top 60 by importance ──
smote_fs = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=1.0)
X_rs, y_rs = smote_fs.fit_resample(X_all.values, y.values)
n_min = int(sum(y_rs == 1))
under_fs = RandomUnderSampler(sampling_strategy={0: int(n_min * 0.6), 1: n_min}, random_state=42)
X_rs, y_rs = under_fs.fit_resample(X_rs, y_rs)

imp_model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                class_weights={0: 1, 1: 3}, eval_metric='AUC',
                                random_seed=42, verbose=0)
imp_model.fit(X_rs, y_rs)
importances = imp_model.feature_importances_
ranked_idx = np.argsort(importances)[::-1]
top60_feats = [feat_all[i] for i in ranked_idx[:60]]
X = X_all[top60_feats]

print(f"Selected top 60 features by importance")

# ── 5. Cross-validation ──
print(f"\n{'=' * 60}")
print("5-fold GroupKFold cross-validation")
print(f"{'=' * 60}")

gkf = GroupKFold(n_splits=5)
oof_probs = np.zeros(len(X))
oof_true = np.zeros(len(X))

for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
    Xtr, Xva = X.iloc[tr].values, X.iloc[va].values
    ytr, yva = y.iloc[tr].values, y.iloc[va].values

    # SMOTE to 1:1, then undersample majority to 60% of minority
    smote = SMOTE(random_state=42, k_neighbors=min(5, sum(ytr == 1) - 1), sampling_strategy=1.0)
    Xtr_rs, ytr_rs = smote.fit_resample(Xtr, ytr)
    n_min = int(sum(ytr_rs == 1))
    under = RandomUnderSampler(sampling_strategy={0: int(n_min * 0.6), 1: n_min}, random_state=42)
    Xtr_rs, ytr_rs = under.fit_resample(Xtr_rs, ytr_rs)

    model = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        class_weights={0: 1, 1: 3}, eval_metric='AUC',
        random_seed=42, verbose=0)
    model.fit(Xtr_rs, ytr_rs)
    oof_probs[va] = model.predict_proba(Xva)[:, 1]
    oof_true[va] = yva

    preds_fold = (oof_probs[va] > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yva, preds_fold).ravel()
    print(f"  Fold {fold+1}: TP={tp} FN={fn} FP={fp} TN={tn}")

# ── 6. Results ──
preds = (oof_probs > 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(oof_true, preds).ravel()
rec = tp / (tp + fn)
prec = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
auc = roc_auc_score(oof_true, oof_probs)

print(f"\n{'=' * 60}")
print("FINAL RESULTS")
print(f"{'=' * 60}")
print(f"                    Predicted No    Predicted Yes")
print(f"  Actual Lesion      FN = {fn:<5}      TP = {tp:<5}")
print(f"  Actual No Lesion   TN = {tn:<5}      FP = {fp:<5}")
print(f"\n  Recall:    {rec:.3f}  ({tp}/{tp+fn})")
print(f"  Precision: {prec:.3f}  ({tp}/{tp+fp})")
print(f"  F1:        {f1:.3f}")
print(f"  AUC-ROC:   {auc:.4f}")

# ── 7. Train final model on all data ──
print(f"\nTraining final model on all {len(X)} participants...")
smote_final = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=1.0)
X_final_rs, y_final_rs = smote_final.fit_resample(X.values, y.values)
n_min = int(sum(y_final_rs == 1))
under_final = RandomUnderSampler(sampling_strategy={0: int(n_min * 0.6), 1: n_min}, random_state=42)
X_final_rs, y_final_rs = under_final.fit_resample(X_final_rs, y_final_rs)

final_model = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.05,
    class_weights={0: 1, 1: 3}, eval_metric='AUC',
    random_seed=42, verbose=0)
final_model.fit(X_final_rs, y_final_rs)

# ── 8. Save ──
artifact = {
    'model': final_model,
    'feature_cols': feature_cols,
    'top60_feats': top60_feats,
    'high_signal_tasks': HIGH_SIGNAL_TASKS,
    'threshold': 0.30,
    'task_prefixes': ['cape-v', 'diadochokinesis', 'free-speech', 'harvard-sentences',
                      'maximum-phonation', 'prolonged-vowel', 'rainbow-passage',
                      'respiration-and-cough', 'caterpillar', 'loudness', 'picture-description',
                      'story-recall', 'glides', 'animal-fluency', 'open-response',
                      'breath-sounds', 'cinderella', 'productive-vocab', 'random-item',
                      'voluntary-cough', 'word-color-stroop'],
}
joblib.dump(artifact, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ── 9. Top features ──
final_imp = final_model.feature_importances_
top_idx = np.argsort(final_imp)[::-1][:15]
print("\nTop 15 features:")
for i, idx in enumerate(top_idx):
    print(f"  {i+1:2d}. {top60_feats[idx]:55s} {final_imp[idx]:.2f}")
