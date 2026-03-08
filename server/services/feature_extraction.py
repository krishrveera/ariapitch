"""
Feature Extraction Service
Extracts comprehensive B2AI-Voice features using SenseLab.
"""
import numpy as np
import torch
import pandas as pd


def extract_features(wav_path: str, task_type: str) -> dict:
    """
    Extract B2AI-aligned features from processed audio using SenseLab.

    Returns a comprehensive feature dictionary including:
    - Praat features (pitch, jitter, shimmer, HNR)
    - MFCCs
    - OpenSMILE features
    - Spectrograms
    - PPGs (Phonetic Posteriorgrams)
    - And more

    Args:
        wav_path: Path to preprocessed 16kHz WAV file
        task_type: Type of vocal task

    Returns:
        Dictionary containing all extracted features
    """
    try:
        from senselab.audio.data_structures import Audio
        from senselab.audio.tasks.features_extraction import extract_features_from_audios
    except ImportError:
        # Fallback to basic Praat extraction if SenseLab not available
        return _extract_basic_praat_features(wav_path, task_type)

    # Load audio using SenseLab
    audio = Audio(filepath=wav_path)

    # Extract all features using B2AI pipeline
    features_list = extract_features_from_audios([audio])
    raw_features = features_list[0] if features_list else {}

    # Remove unwanted torchaudio features
    EXCLUDED_FEATURES = ['mel_filter_bank', 'mel_spectrogram', 'mfcc', 'pitch', 'spectrogram']
    if 'torchaudio' in raw_features:
        for feature_name in EXCLUDED_FEATURES:
            raw_features['torchaudio'].pop(feature_name, None)

    # Flatten and serialize the B2AI features for JSON response
    flattened = _flatten_b2ai_features(raw_features)

    return flattened


def _flatten_b2ai_features(features_dict: dict, prefix: str = "") -> dict:
    """
    Recursively flatten nested B2AI feature structure into a flat dictionary
    suitable for JSON serialization and ML model input.

    Handles:
    - Nested dictionaries (recursive)
    - NumPy arrays (convert to lists with statistics)
    - PyTorch tensors (convert to lists with statistics)
    - Pandas DataFrames (extract summary statistics)
    - Scalars (keep as-is)
    - Strings (keep as-is)
    """
    flattened = {}

    for key, value in features_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key

        # Handle nested dictionaries
        if isinstance(value, dict):
            nested = _flatten_b2ai_features(value, full_key)
            flattened.update(nested)
            continue

        # Handle PyTorch tensors
        if isinstance(value, torch.Tensor):
            val_np = value.squeeze().detach().cpu().numpy()

            # 2D arrays (spectrograms, MFCCs, PPGs)
            if val_np.ndim == 2:
                flattened[f"{full_key}.shape"] = list(val_np.shape)
                flattened[f"{full_key}.mean"] = float(np.mean(val_np))
                flattened[f"{full_key}.std"] = float(np.std(val_np))
                flattened[f"{full_key}.min"] = float(np.min(val_np))
                flattened[f"{full_key}.max"] = float(np.max(val_np))

                # Store mean across time for each coefficient/channel
                if val_np.shape[0] < 100:  # If first dim is small (e.g., MFCC coefficients)
                    mean_across_time = np.mean(val_np, axis=1)
                    for i, val in enumerate(mean_across_time):
                        flattened[f"{full_key}.coef_{i}"] = round(float(val), 4)

            # 1D arrays (pitch contour, loudness, etc.)
            elif val_np.ndim == 1:
                flattened[f"{full_key}.length"] = len(val_np)
                flattened[f"{full_key}.mean"] = float(np.mean(val_np))
                flattened[f"{full_key}.std"] = float(np.std(val_np))
                flattened[f"{full_key}.min"] = float(np.min(val_np))
                flattened[f"{full_key}.max"] = float(np.max(val_np))
                flattened[f"{full_key}.median"] = float(np.median(val_np))

            # Scalars
            else:
                flattened[full_key] = float(val_np)

            continue

        # Handle NumPy arrays
        if isinstance(value, np.ndarray):
            val_np = np.squeeze(value)

            if val_np.ndim == 2:
                flattened[f"{full_key}.shape"] = list(val_np.shape)
                flattened[f"{full_key}.mean"] = float(np.mean(val_np))
                flattened[f"{full_key}.std"] = float(np.std(val_np))
            elif val_np.ndim == 1:
                flattened[f"{full_key}.mean"] = float(np.mean(val_np))
                flattened[f"{full_key}.std"] = float(np.std(val_np))
                flattened[f"{full_key}.min"] = float(np.min(val_np))
                flattened[f"{full_key}.max"] = float(np.max(val_np))
            else:
                flattened[full_key] = float(val_np)

            continue

        # Handle Pandas DataFrames (OpenSMILE features)
        if isinstance(value, pd.DataFrame):
            flattened[f"{full_key}.shape"] = list(value.shape)

            # Extract key statistics from each column
            for col in value.columns:
                col_clean = str(col).replace(" ", "_").replace("[", "").replace("]", "")
                if value[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    flattened[f"{full_key}.{col_clean}.mean"] = float(value[col].mean())
                    flattened[f"{full_key}.{col_clean}.std"] = float(value[col].std())

            continue

        # Handle strings (transcriptions)
        if isinstance(value, str):
            flattened[f"{full_key}.text"] = value
            flattened[f"{full_key}.length"] = len(value)
            continue

        # Handle scalars (floats, ints, bools)
        if isinstance(value, (int, float, bool, np.integer, np.floating)):
            flattened[full_key] = float(value) if isinstance(value, (float, np.floating)) else value
            continue

        # Skip unhandled types
        flattened[f"{full_key}.type"] = str(type(value).__name__)

    return flattened


def _extract_basic_praat_features(wav_path: str, task_type: str = None) -> dict:
    """
    Fallback feature extraction using basic Praat/Parselmouth.
    Used when SenseLab is not available.
    """
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        raise ImportError("Neither SenseLab nor Parselmouth is available for feature extraction")

    sound = parselmouth.Sound(wav_path)
    features = {}

    # F0 (Pitch)
    try:
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        f0_values = pitch.selected_array["frequency"]
        f0_voiced = f0_values[f0_values > 0]

        if len(f0_voiced) > 0:
            features["praat.pitch.f0_mean_hz"] = round(float(np.mean(f0_voiced)), 2)
            features["praat.pitch.f0_std_hz"] = round(float(np.std(f0_voiced)), 2)
            features["praat.pitch.f0_min_hz"] = round(float(np.min(f0_voiced)), 2)
            features["praat.pitch.f0_max_hz"] = round(float(np.max(f0_voiced)), 2)
        else:
            features["praat.pitch.f0_mean_hz"] = 0.0
            features["praat.pitch.f0_std_hz"] = 0.0
            features["praat.pitch.f0_min_hz"] = 0.0
            features["praat.pitch.f0_max_hz"] = 0.0
    except Exception:
        features["praat.pitch.f0_mean_hz"] = 0.0
        features["praat.pitch.f0_std_hz"] = 0.0

    # Jitter & Shimmer
    try:
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)

        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)

        shimmer_local = call(
            [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        shimmer_apq3 = call(
            [sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        features["praat.jitter.local_percent"] = round(jitter_local * 100, 4)
        features["praat.jitter.rap_percent"] = round(jitter_rap * 100, 4)
        features["praat.shimmer.local_percent"] = round(shimmer_local * 100, 4)
        features["praat.shimmer.apq3_percent"] = round(shimmer_apq3 * 100, 4)
    except Exception:
        features["praat.jitter.local_percent"] = 0.0
        features["praat.shimmer.local_percent"] = 0.0

    # HNR
    try:
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        features["praat.hnr.mean_db"] = round(hnr, 2)
    except Exception:
        features["praat.hnr.mean_db"] = 0.0

    # Formants
    try:
        formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        f1 = call(formant, "Get mean", 1, 0, 0, "hertz")
        f2 = call(formant, "Get mean", 2, 0, 0, "hertz")
        f3 = call(formant, "Get mean", 3, 0, 0, "hertz")

        features["praat.formants.f1_hz"] = round(f1, 1)
        features["praat.formants.f2_hz"] = round(f2, 1)
        features["praat.formants.f3_hz"] = round(f3, 1)
    except Exception:
        features["praat.formants.f1_hz"] = 0.0
        features["praat.formants.f2_hz"] = 0.0
        features["praat.formants.f3_hz"] = 0.0

    # CPP
    try:
        power_cepstrogram = call(sound, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        cpps = call(power_cepstrogram, "Get CPPS", False, 0.02, 0.0005, 60, 330, 0.05,
                   "Parabolic", 0.001, 0, "Exponential decay", "Robust slow")
        features["praat.cpp.mean_db"] = round(cpps, 2)
    except Exception:
        features["praat.cpp.mean_db"] = 0.0

    features["_extraction_method"] = "praat_fallback"
    return features
