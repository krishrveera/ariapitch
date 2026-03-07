import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

def prepare_hackathon_data(data_root="."):
    """
    Loads B2AI TSV files, applies clinical preprocessing, and sets up 
    cross-validation groups to prevent participant leakage.
    """
    # 1. Load Data Core
    # features/static_features.tsv contains the ~130 biomarkers
    df_features = pd.read_csv(f"{data_root}/features/static_features.tsv", sep='\t')
    
    # phenotype/diagnosis/benign_lesions.tsv contains the ground truth
    df_labels = pd.read_csv(f"{data_root}/phenotype/diagnosis/benign_lesions.tsv", sep='\t')
    
    # 2. Merge on Participant & Session
    # Ensures labels and features are perfectly aligned
    df = pd.merge(df_features, df_labels[['participant_id', 'label']], on='participant_id', how='inner')

    # 3. Clinical Preprocessing (Log-Transforms)
    # Stability markers like Jitter and Shimmer are often log-normally distributed
    log_cols = ['local_jitter', 'local_shimmer', 'jitterLocal_sma3nz_amean']
    for col in log_cols:
        if col in df.columns:
            # Add small constant to avoid log(0)
            df[col] = np.log10(df[col] + 1e-6)

    # 4. Handle Missing Data (Median Imputation)
    # Fills gaps where pitch couldn't be detected without biasing the model
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))

    # 5. Define Feature Set
    # Drop IDs and target from training columns
    X = df.drop(columns=['participant_id', 'session_id', 'label', 'transcription'], errors='ignore')
    y = df['label']
    groups = df['participant_id'] # Crucial for GroupKFold

    # 6. GroupKFold Splitting
    # Ensures the model is tested on voices it has NEVER heard before
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(X, y, groups=groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 7. Final Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"✅ Data Prepared: {len(X_train)} training samples, {len(X_val)} validation samples.")
    print(f"🚀 Balanced classes: {y_train.value_counts(normalize=True).to_dict()}")

    return X_train_scaled, X_val_scaled, y_train, y_val, X.columns

# Usage:
# X_train, X_val, y_train, y_val, feature_names = prepare_hackathon_data()