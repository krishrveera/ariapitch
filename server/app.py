"""
Flask Application Factory
Main entry point for the Voice Health Analysis API server.
"""
from flask import Flask
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from api.routes import api_bp
from api.response import error_response
from services.prediction import load_model
import uuid
import logging
import numpy as np


class NumpyJSONProvider(DefaultJSONProvider):
    """Custom JSON provider that handles numpy types."""

    def default(self, o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.json_provider_class = NumpyJSONProvider
    app.json = NumpyJSONProvider(app)
    CORS(app)

    # Load ML model into memory once at startup
    with app.app_context():
        load_model()

    # Register routes
    app.register_blueprint(api_bp)

    # Global error handlers
    @app.errorhandler(400)
    def bad_request(e):
        return error_response(
            error_type="validation_error",
            message=str(e),
            code=400,
            request_id=str(uuid.uuid4())
        )

    @app.errorhandler(413)
    def too_large(e):
        return error_response(
            error_type="validation_error",
            message="File too large. Maximum upload size is 30 MB.",
            code=413,
            request_id=str(uuid.uuid4())
        )

    @app.errorhandler(500)
    def server_error(e):
        logging.exception("Unhandled server error")
        return error_response(
            error_type="server_error",
            message="An internal server error occurred. Please try again.",
            code=500,
            request_id=str(uuid.uuid4())
        )

    # Configuration
    app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
