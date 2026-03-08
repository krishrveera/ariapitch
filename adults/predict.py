import pandas as pd
import numpy as np
import joblib
import sys

MODEL_PATH = "benign_lesion_model.joblib"

def predict_benign_lesion(input_path):
    """
    Given a TSV of voice features (raw recordings), aggregates per participant
    and returns the probability of benign lesion.
    """
    artifact = joblib.load(MODEL_PATH)
    model = artifact['model']
    feature_cols = artifact['feature_cols']
    top60_feats = artifact['top60_feats']
    high_signal_tasks = artifact['high_signal_tasks']
    task_prefixes = artifact['task_prefixes']

    def get_task_category(task):
        for prefix in task_prefixes:
            if task.startswith(prefix):
                return prefix
        return task

    df = pd.read_csv(input_path, sep='\t')

    # Filter to high-signal tasks
    df['task_category'] = df['task_name'].apply(get_task_category)
    df_hs = df[df['task_category'].isin(high_signal_tasks)].copy()

    if len(df_hs) == 0:
        print("Warning: no high-signal task recordings found. Using all recordings.")
        df_hs = df.copy()

    # Aggregate per participant: mean + std
    agg_mean = df_hs.groupby('participant_id')[feature_cols].mean()
    agg_std = df_hs.groupby('participant_id')[feature_cols].std().fillna(0)
    agg_mean.columns = [c + '_mean' for c in feature_cols]
    agg_std.columns = [c + '_std' for c in feature_cols]
    agg = pd.concat([agg_mean, agg_std], axis=1)

    # Ensure all expected features are present
    for c in top60_feats:
        if c not in agg.columns:
            agg[c] = 0

    X = agg[top60_feats].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    probabilities = model.predict_proba(X.values)[:, 1]
    threshold = artifact.get('threshold', 0.30)

    results = pd.DataFrame({
        'participant_id': agg.index,
        'benign_lesion_probability': probabilities,
        'flagged': (probabilities >= threshold).astype(int),
        'risk_level': pd.cut(probabilities, bins=[0, 0.3, 0.6, 1.0],
                             labels=['Low', 'Medium', 'High']),
    })

    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_features.tsv>")
        print("  Aggregates per participant and returns benign lesion probability.")
        sys.exit(1)

    results = predict_benign_lesion(sys.argv[1])
    print(results.to_string(index=False))
    print(f"\nProcessed {len(results)} participants")
    print(f"Mean probability: {results['benign_lesion_probability'].mean():.4f}")
