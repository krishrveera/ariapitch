"""
API Routes
Route definitions for the Flask server.
"""
import uuid
import numpy as np
from flask import Blueprint, request
from api.response import success_response, error_response, start_timer
from services.quality_gate import run_quality_gate
from services.preprocessing import run_preprocessing
from services.feature_extraction import extract_features
from services.prediction import predict, get_model_status
from services.explanation import generate_explanation, generate_quality_failure_suggestion
from services.task_definitions import get_all_tasks
from utils.audio_io import save_upload_to_temp, cleanup_temp


api_bp = Blueprint("api", __name__, url_prefix="/api/v1")


@api_bp.route("/analyze", methods=["POST"])
def analyze():
    """Main analysis endpoint - runs full pipeline."""
    request_id = str(uuid.uuid4())
    start_timer(request_id)

    # Validate request
    if "audio" not in request.files:
        return error_response(
            error_type="validation_error",
            message="No audio file provided.",
            code=400,
            suggestion="Include an audio file in the 'audio' field of the multipart form.",
            request_id=request_id
        )

    audio_file = request.files["audio"]
    device_id = request.form.get("device_id", "unknown")
    task_type = request.form.get("task_type", "sustained_vowel")
    silence_sec = float(request.form.get("silence_duration_sec", 3.0))

    valid_tasks = ["sustained_vowel", "free_speech", "reading_passage", "cough"]
    if task_type not in valid_tasks:
        return error_response(
            error_type="validation_error",
            message=f"Invalid task_type '{task_type}'.",
            code=400,
            suggestion=f"Must be one of: {', '.join(valid_tasks)}",
            request_id=request_id
        )

    # Save to temp file
    try:
        temp_path = save_upload_to_temp(audio_file)
    except ValueError as e:
        return error_response(
            error_type="validation_error",
            message=str(e),
            code=400,
            request_id=request_id
        )

    processed_path = None

    try:
        # Quality Gate
        gate_result = run_quality_gate(temp_path, device_id, task_type, silence_sec)

        if not gate_result["passed"]:
            suggestion = generate_quality_failure_suggestion(gate_result)
            return error_response(
                error_type="quality_gate_failure",
                message="Recording quality is too low for accurate analysis.",
                code=422,
                failed_checks=gate_result["failed_checks"],
                passed_checks=gate_result["passed_checks"],
                warnings=gate_result["warnings"],
                suggestion=suggestion,
                request_id=request_id
            )

        # Preprocessing
        processed_path, preproc_info = run_preprocessing(
            temp_path, device_id, silence_sec, gate_result
        )

        # Feature Extraction
        features = extract_features(processed_path, task_type)

        # ML Prediction
        predictions = predict(features, task_type)

        # LLM Explanation
        explanation = generate_explanation(features, predictions, task_type)

        # Assemble response
        data = {
            "quality": gate_result["summary"],
            "preprocessing": preproc_info,
            "features": features,
            "predictions": predictions,
            "explanation": explanation
        }

        return success_response(
            data=data,
            message="Analysis complete.",
            code=200,
            request_id=request_id
        )

    except Exception as e:
        return error_response(
            error_type="processing_error",
            message=f"An error occurred during processing: {str(e)}",
            code=500,
            request_id=request_id
        )

    finally:
        cleanup_temp(temp_path, processed_path)


@api_bp.route("/validate", methods=["POST"])
def validate():
    """Quick quality check without full analysis."""
    request_id = str(uuid.uuid4())
    start_timer(request_id)

    if "audio" not in request.files:
        return error_response(
            error_type="validation_error",
            message="No audio file provided.",
            code=400,
            request_id=request_id
        )

    audio_file = request.files["audio"]
    device_id = request.form.get("device_id", "unknown")
    task_type = request.form.get("task_type", "sustained_vowel")
    silence_sec = float(request.form.get("silence_duration_sec", 3.0))

    try:
        temp_path = save_upload_to_temp(audio_file)
    except ValueError as e:
        return error_response(
            error_type="validation_error",
            message=str(e),
            code=400,
            request_id=request_id
        )

    try:
        gate_result = run_quality_gate(temp_path, device_id, task_type, silence_sec)

        if not gate_result["passed"]:
            suggestion = generate_quality_failure_suggestion(gate_result)
            return error_response(
                error_type="quality_gate_failure",
                message="Recording quality is too low for accurate analysis.",
                code=422,
                failed_checks=gate_result["failed_checks"],
                passed_checks=gate_result["passed_checks"],
                warnings=gate_result["warnings"],
                suggestion=suggestion,
                request_id=request_id
            )

        return success_response(
            data={"quality": gate_result["summary"]},
            message="Recording quality is acceptable.",
            code=200,
            request_id=request_id
        )

    except Exception as e:
        return error_response(
            error_type="processing_error",
            message=f"An error occurred: {str(e)}",
            code=500,
            request_id=request_id
        )

    finally:
        cleanup_temp(temp_path)


@api_bp.route("/health", methods=["GET"])
def health():
    """Server health check."""
    request_id = str(uuid.uuid4())
    start_timer(request_id)

    model_status = get_model_status()
    return success_response(
        data=model_status,
        message="Server is healthy.",
        request_id=request_id
    )


@api_bp.route("/tasks", methods=["GET"])
def tasks():
    """Get available vocal tasks."""
    request_id = str(uuid.uuid4())
    start_timer(request_id)

    return success_response(
        data={"tasks": get_all_tasks()},
        message="Available tasks retrieved.",
        request_id=request_id
    )


@api_bp.route("/demo/analyze", methods=["POST"])
def demo_analyze():
    """
    Demo endpoint with full visualizations.

    Returns everything from /analyze plus:
    - Base64-encoded PNG visualizations
    - Quality gate plots
    - Preprocessing comparisons
    - Feature visualizations
    - Raw B2AI feature structure (not flattened)
    """
    request_id = str(uuid.uuid4())
    start_timer(request_id)

    # Validate request
    if "audio" not in request.files:
        return error_response(
            error_type="validation_error",
            message="No audio file provided.",
            code=400,
            suggestion="Include an audio file in the 'audio' field of the multipart form.",
            request_id=request_id
        )

    audio_file = request.files["audio"]
    device_id = request.form.get("device_id", "unknown")
    task_type = request.form.get("task_type", "sustained_vowel")
    silence_sec = float(request.form.get("silence_duration_sec", 3.0))

    valid_tasks = ["sustained_vowel", "free_speech", "reading_passage", "cough"]
    if task_type not in valid_tasks:
        return error_response(
            error_type="validation_error",
            message=f"Invalid task_type '{task_type}'.",
            code=400,
            suggestion=f"Must be one of: {', '.join(valid_tasks)}",
            request_id=request_id
        )

    # Save to temp file
    try:
        temp_path = save_upload_to_temp(audio_file)
    except ValueError as e:
        return error_response(
            error_type="validation_error",
            message=str(e),
            code=400,
            request_id=request_id
        )

    processed_path = None

    try:
        # Quality Gate
        gate_result = run_quality_gate(temp_path, device_id, task_type, silence_sec)

        if not gate_result["passed"]:
            # Generate visualizations even for failed quality gate
            from services.visualization import generate_all_visualizations

            visualizations = generate_all_visualizations(
                temp_path, gate_result, {}
            )

            suggestion = generate_quality_failure_suggestion(gate_result)
            return error_response(
                error_type="quality_gate_failure",
                message="Recording quality is too low for accurate analysis.",
                code=422,
                failed_checks=gate_result["failed_checks"],
                passed_checks=gate_result["passed_checks"],
                warnings=gate_result["warnings"],
                suggestion=suggestion,
                request_id=request_id
            )

        # Preprocessing
        processed_path, preproc_info = run_preprocessing(
            temp_path, device_id, silence_sec, gate_result
        )

        # Feature Extraction (get RAW features for visualization)
        try:
            from senselab.audio.data_structures import Audio
            from senselab.audio.tasks.features_extraction import extract_features_from_audios

            audio_obj = Audio(filepath=processed_path)
            raw_features_list = extract_features_from_audios([audio_obj])
            raw_features = raw_features_list[0] if raw_features_list else {}

            # Also get flattened features for ML
            from services.feature_extraction import extract_features, _flatten_b2ai_features
            features_flattened = _flatten_b2ai_features(raw_features)

        except ImportError:
            # Fallback to basic features
            from services.feature_extraction import extract_features
            features_flattened = extract_features(processed_path, task_type)
            raw_features = {}

        # ML Prediction
        predictions = predict(features_flattened, task_type)

        # LLM Explanation
        explanation = generate_explanation(features_flattened, predictions, task_type)

        # Generate Visualizations
        from services.visualization import (
            generate_all_visualizations,
            generate_b2ai_feature_visualizations
        )

        visualizations = generate_all_visualizations(
            temp_path, gate_result, preproc_info
        )

        # Add B2AI feature visualizations
        if raw_features:
            feature_viz = generate_b2ai_feature_visualizations(raw_features)
            visualizations.update(feature_viz)

        # Assemble response
        data = {
            "quality": gate_result["summary"],
            "preprocessing": preproc_info,
            "features": features_flattened,
            "features_raw": _serialize_raw_features(raw_features),  # For inspection
            "predictions": predictions,
            "explanation": explanation,
            "visualizations": visualizations  # Base64-encoded PNGs
        }

        return success_response(
            data=data,
            message="Demo analysis complete with visualizations.",
            code=200,
            request_id=request_id
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return error_response(
            error_type="processing_error",
            message=f"An error occurred during processing: {str(e)}",
            code=500,
            request_id=request_id
        )

    finally:
        cleanup_temp(temp_path, processed_path)


def _serialize_raw_features(features_dict: dict, max_depth: int = 3, current_depth: int = 0) -> dict:
    """
    Serialize raw B2AI features for JSON (convert tensors/arrays to summary).
    Limits depth to avoid huge responses.
    """
    import torch
    import pandas as pd

    if current_depth >= max_depth:
        return {"_truncated": "Max depth reached"}

    serialized = {}

    for key, value in features_dict.items():
        # Nested dicts
        if isinstance(value, dict):
            serialized[key] = _serialize_raw_features(value, max_depth, current_depth + 1)

        # Tensors
        elif isinstance(value, torch.Tensor):
            val_np = value.squeeze().detach().cpu().numpy()
            serialized[key] = {
                "type": "tensor",
                "shape": list(val_np.shape),
                "dtype": str(val_np.dtype),
                "mean": float(np.mean(val_np)),
                "std": float(np.std(val_np))
            }

        # Arrays
        elif isinstance(value, np.ndarray):
            serialized[key] = {
                "type": "array",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "mean": float(np.mean(value)),
                "std": float(np.std(value))
            }

        # DataFrames
        elif isinstance(value, pd.DataFrame):
            serialized[key] = {
                "type": "dataframe",
                "shape": list(value.shape),
                "columns": list(value.columns[:10])  # First 10 columns
            }

        # Strings
        elif isinstance(value, str):
            serialized[key] = value

        # Scalars
        elif isinstance(value, (int, float, bool)):
            serialized[key] = value

        else:
            serialized[key] = {"type": str(type(value).__name__)}

    return serialized
