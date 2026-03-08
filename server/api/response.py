"""
Standardized Response Envelope Builder
Every API endpoint uses these functions to ensure consistent response format.
"""
import uuid
from datetime import datetime, timezone
from flask import jsonify
import time


_request_start_time = {}


def start_timer(request_id: str):
    """Call at the start of every request."""
    _request_start_time[request_id] = time.monotonic()


def success_response(data: dict, message: str = "OK", code: int = 200,
                     request_id: str = None) -> tuple:
    """Build a success envelope."""
    request_id = request_id or str(uuid.uuid4())
    elapsed = _get_elapsed(request_id)

    body = {
        "status": "success",
        "code": code,
        "message": message,
        "data": data,
        "errors": None,
        "meta": {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": elapsed
        }
    }
    return jsonify(body), code


def error_response(error_type: str, message: str, code: int = 400,
                   failed_checks: list = None, passed_checks: list = None,
                   warnings: list = None, suggestion: str = None,
                   request_id: str = None) -> tuple:
    """Build an error envelope."""
    request_id = request_id or str(uuid.uuid4())
    elapsed = _get_elapsed(request_id)

    errors = {"type": error_type}
    if failed_checks is not None:
        errors["failed_checks"] = failed_checks
    if passed_checks is not None:
        errors["passed_checks"] = passed_checks
    if warnings is not None:
        errors["warnings"] = warnings
    if suggestion is not None:
        errors["suggestion"] = suggestion

    body = {
        "status": "error",
        "code": code,
        "message": message,
        "data": None,
        "errors": errors,
        "meta": {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": elapsed
        }
    }
    return jsonify(body), code


def _get_elapsed(request_id: str) -> int:
    """Get elapsed time in milliseconds for this request."""
    start = _request_start_time.pop(request_id, None)
    if start is None:
        return 0
    return int((time.monotonic() - start) * 1000)
