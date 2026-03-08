"""
Audio File I/O Utilities
Handles file uploads, temp storage, and cleanup.
"""
import tempfile
import os
from werkzeug.datastructures import FileStorage


ALLOWED_EXTENSIONS = {".m4a", ".caf", ".wav", ".mp3", ".webm", ".ogg", ".flac"}
MAX_FILE_SIZE = 30 * 1024 * 1024  # 30 MB


def save_upload_to_temp(file: FileStorage) -> str:
    """
    Save an uploaded file to a temp path.

    Args:
        file: Flask FileStorage object from request.files

    Returns:
        Path to temporary file

    Raises:
        ValueError: If file format is unsupported or file is too large
    """
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".m4a"
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {ext}")

    temp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    file.save(temp.name)
    temp.close()

    # Check file size
    size = os.path.getsize(temp.name)
    if size > MAX_FILE_SIZE:
        os.unlink(temp.name)
        raise ValueError(f"File too large: {size} bytes (max {MAX_FILE_SIZE})")

    return temp.name


def cleanup_temp(*paths):
    """
    Delete temporary files.

    Args:
        *paths: Variable number of file paths to delete

    Silently ignores missing files and errors.
    """
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except OSError:
            pass
