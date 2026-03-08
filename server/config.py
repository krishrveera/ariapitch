"""
Configuration Settings
Environment-based configuration for the Flask server.
"""
import os


class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    MAX_CONTENT_LENGTH = 30 * 1024 * 1024  # 30 MB
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True


# Configuration mapping
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig
}
