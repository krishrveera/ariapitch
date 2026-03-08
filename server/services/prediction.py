"""
ML Prediction Service
Placeholder for machine learning model integration.
"""
from typing import List


# Model state
_model = None
_model_version = "b2ai-voice-v0.1.0-placeholder"


def load_model():
    """Load the ML model into memory. Called once at server startup."""
    global _model
    # TODO: Replace with actual model loading
    # Example: _model = torch.load("models/b2ai_v1.pt")
    _model = None  # Placeholder


def get_model_status() -> dict:
    """Get model status for health check endpoint."""
    return {
        "version": "1.0.0",
        "model_loaded": _model is not None,
        "model_version": _model_version,
        "gpu_available": False
    }


# Condition configurations
# Maps to both B2AI feature paths and Praat fallback paths
CONDITION_CONFIGS = {
    "parkinsons": {
        "display_name": "Parkinson's Disease",
        "features": {
            # B2AI paths (preferred)
            "jitter": ["praat.jitter.local_percent", "praat.jitter.local.mean"],
            "shimmer": ["praat.shimmer.local_percent", "praat.shimmer.local.mean"],
            "hnr": ["praat.hnr.mean_db", "praat.harmonicity.mean"],
            "f0_std": ["praat.pitch.f0_std_hz", "praat.pitch.std"],
        },
        "reference_thresholds": {
            "jitter_local_percent": {"normal_max": 1.04, "source": "Teixeira et al., 2013"},
            "shimmer_local_percent": {"normal_max": 3.81, "source": "Teixeira et al., 2013"},
            "hnr_db": {"normal_min": 20.0, "source": "Boersma, 1993"},
        }
    },
    "vocal_fold_paralysis": {
        "display_name": "Vocal Fold Paralysis",
        "features": {
            "cpp": ["praat.cpp.mean_db", "praat.cepstrum.cpp.mean"],
            "hnr": ["praat.hnr.mean_db", "praat.harmonicity.mean"],
            "shimmer": ["praat.shimmer.local_percent", "praat.shimmer.local.mean"],
            "f0_std": ["praat.pitch.f0_std_hz", "praat.pitch.std"],
        },
        "reference_thresholds": {
            "cpp_db": {"normal_min": 8.0, "source": "Heman-Ackah et al., 2003"},
        }
    },
    "depression": {
        "display_name": "Depression",
        "features": {
            "f0_mean": ["praat.pitch.f0_mean_hz", "praat.pitch.mean"],
            "f0_std": ["praat.pitch.f0_std_hz", "praat.pitch.std"],
            "jitter": ["praat.jitter.local_percent", "praat.jitter.local.mean"],
        },
        "reference_thresholds": {}
    },
    "copd": {
        "display_name": "COPD",
        "features": {
            "cpp": ["praat.cpp.mean_db", "praat.cepstrum.cpp.mean"],
            "hnr": ["praat.hnr.mean_db", "praat.harmonicity.mean"],
            "shimmer": ["praat.shimmer.local_percent", "praat.shimmer.local.mean"],
        },
        "reference_thresholds": {}
    },
}


# Task-to-condition mapping
TASK_CONDITIONS = {
    "sustained_vowel": ["parkinsons", "vocal_fold_paralysis"],
    "free_speech": ["depression", "parkinsons"],
    "reading_passage": ["vocal_fold_paralysis", "parkinsons"],
    "cough": ["copd"],
}


def _get_feature_value(features: dict, feature_paths: list) -> float:
    """
    Extract feature value from B2AI feature dict using multiple possible paths.

    Args:
        features: Flattened feature dictionary
        feature_paths: List of possible paths to try (in order of preference)

    Returns:
        Feature value, or 0.0 if not found
    """
    for path in feature_paths:
        if path in features:
            return float(features[path])
    return 0.0


def predict(features: dict, task_type: str) -> List[dict]:
    """
    Run prediction for all conditions relevant to the given task.

    Returns a list of prediction dicts, one per condition.

    TODO: Replace placeholder logic with actual model inference.
    """
    conditions = TASK_CONDITIONS.get(task_type, [])
    predictions = []

    for condition_id in conditions:
        config = CONDITION_CONFIGS.get(condition_id)
        if config is None:
            continue

        # Extract feature values (works with both B2AI and Praat fallback)
        feature_values = {}
        for feature_name, feature_paths in config["features"].items():
            feature_values[feature_name] = _get_feature_value(features, feature_paths)

        # PLACEHOLDER: return 0% risk until model is ready
        # When your model is ready, replace this with:
        #   feature_vector = list(feature_values.values())
        #   risk = _model.predict_proba([feature_vector])[0][1] * 100
        #   confidence = _model.confidence([feature_vector])
        risk_percent = 0.0
        confidence = 0.0

        # Severity tier based on risk percentage
        if risk_percent < 15:
            severity = "low"
        elif risk_percent < 40:
            severity = "moderate"
        elif risk_percent < 70:
            severity = "elevated"
        else:
            severity = "high"

        predictions.append({
            "condition": condition_id,
            "condition_name": config["display_name"],
            "risk_percent": round(risk_percent, 1),
            "confidence": round(confidence, 2),
            "severity_tier": severity,
            "features_used": list(feature_values.keys()),
            "feature_values": feature_values,
            "reference_thresholds": config["reference_thresholds"]
        })

    return predictions
