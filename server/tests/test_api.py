"""
Comprehensive API Test Suite for Voice Health Analysis API
Tests all endpoints with sample audio files.

Usage:
    python tests/test_api.py
"""

import os
import sys
import json
import requests
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
BASE_URL = "http://localhost:5000"
API_BASE = f"{BASE_URL}/api/v1"

# Find sample audio files
PROJECT_ROOT = Path(__file__).parent.parent.parent
SAMPLE_DIRS = [
    PROJECT_ROOT / "samples",
    PROJECT_ROOT / "audio_samples",
    PROJECT_ROOT / "test_audio",
    PROJECT_ROOT / "data" / "samples",
    PROJECT_ROOT,  # project root (for m4a files)
]


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def fail(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def warn(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}\n")


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_sample_audio():
    """Find a sample audio file to use for testing."""
    for sample_dir in SAMPLE_DIRS:
        if sample_dir.exists():
            for ext in [".wav", ".mp3", ".m4a", ".flac"]:
                files = list(sample_dir.glob(f"*{ext}"))
                if files:
                    return files[0]
    return None


def assert_envelope(response, expected_status="success"):
    """Validate the standard API envelope structure."""
    data = response.json()
    assert "status" in data, f"Missing 'status' field: {list(data.keys())}"
    assert "code" in data, f"Missing 'code' field"
    assert "message" in data, f"Missing 'message' field"
    assert "data" in data, f"Missing 'data' field"
    assert "errors" in data, f"Missing 'errors' field"
    assert "meta" in data, f"Missing 'meta' field"
    assert "request_id" in data["meta"], "Missing meta.request_id"
    assert "timestamp" in data["meta"], "Missing meta.timestamp"
    assert data["status"] == expected_status, (
        f"Expected status '{expected_status}', got '{data['status']}'"
    )
    return data


# ── Tests ────────────────────────────────────────────────────────────────────

passed = 0
failed_count = 0
skipped = 0


def run(name, fn):
    global passed, failed_count
    try:
        fn()
        ok(name)
        passed += 1
    except AssertionError as e:
        fail(f"{name}: {e}")
        failed_count += 1
    except Exception as e:
        fail(f"{name}: {type(e).__name__}: {e}")
        failed_count += 1


# noinspection PyPep8Naming
class AssertionError(AssertionError if False else AssertionError):
    ...


# ---------- 1. Health check -----------------------------------------------

def test_health():
    r = requests.get(f"{API_BASE}/health", timeout=10)
    assert r.status_code == 200, f"Status {r.status_code}"
    data = assert_envelope(r, "success")
    d = data["data"]
    assert "version" in d, f"Missing version in health data: {d}"
    assert "model_version" in d, f"Missing model_version in health data: {d}"
    info(f"  version={d.get('version')}  model_version={d.get('model_version')}")


# ---------- 2. Tasks list --------------------------------------------------

def test_tasks_list():
    r = requests.get(f"{API_BASE}/tasks", timeout=10)
    assert r.status_code == 200, f"Status {r.status_code}"
    data = assert_envelope(r, "success")
    tasks = data["data"]["tasks"]
    assert isinstance(tasks, list), f"tasks should be a list, got {type(tasks)}"
    assert len(tasks) > 0, "No tasks returned"
    info(f"  {len(tasks)} task(s) returned")
    for t in tasks:
        info(f"    - {t.get('display_name', t.get('id', '?'))}")


# ---------- 3. Task detail -------------------------------------------------

def test_task_detail():
    r = requests.get(f"{API_BASE}/tasks/sustained_vowel", timeout=10)
    assert r.status_code == 200, f"Status {r.status_code}"
    data = assert_envelope(r, "success")
    task = data["data"]["task"]
    assert task.get("id") == "sustained_vowel" or "display_name" in task


def test_task_not_found():
    r = requests.get(f"{API_BASE}/tasks/nonexistent_xyz", timeout=10)
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    data = assert_envelope(r, "error")
    assert data["errors"]["type"] == "not_found"


# ---------- 4. Analyze (no file) ------------------------------------------

def test_analyze_no_file():
    r = requests.post(f"{API_BASE}/analyze", timeout=10)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    data = assert_envelope(r, "error")
    assert data["errors"]["type"] == "validation_error"


def test_analyze_bad_task_type():
    """Send a file with an invalid task_type."""
    audio = find_sample_audio()
    if audio is None:
        raise Exception("No sample audio file found – skipping")
    with open(audio, "rb") as f:
        files = {"audio": (audio.name, f, "audio/wav")}
        r = requests.post(
            f"{API_BASE}/analyze",
            files=files,
            data={"task_type": "invalid_task"},
            timeout=60,
        )
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"


# ---------- 5. Full analysis with audio ------------------------------------

def test_analyze_with_audio():
    audio = find_sample_audio()
    if audio is None:
        raise Exception("No sample audio file found – skipping")

    info(f"  Using audio file: {audio.name}")
    with open(audio, "rb") as f:
        files = {"audio": (audio.name, f, "audio/wav")}
        data = {"task_type": "sustained_vowel", "device_id": "test_suite"}
        r = requests.post(f"{API_BASE}/analyze", files=files, data=data, timeout=120)

    if r.status_code == 422:
        # Quality gate failure is an acceptable outcome
        env = assert_envelope(r, "error")
        assert env["errors"]["type"] == "quality_gate_failure"
        info("  Quality gate failure (expected for some files)")
        return

    assert r.status_code == 200, f"Status {r.status_code}: {r.text[:200]}"
    env = assert_envelope(r, "success")
    d = env["data"]

    # Validate structure
    assert "quality" in d, "Missing quality"
    assert "features" in d, "Missing features"
    assert "predictions" in d, "Missing predictions"
    assert "explanation" in d, "Missing explanation"

    # Predictions should be a list of condition dicts
    preds = d["predictions"]
    assert isinstance(preds, list), f"predictions should be list, got {type(preds)}"
    if preds:
        p = preds[0]
        assert "condition" in p, f"Prediction missing 'condition': {p.keys()}"
        assert "probability" in p, f"Prediction missing 'probability'"
        assert "severity_tier" in p, f"Prediction missing 'severity_tier'"
        info(f"  Prediction: {p['condition_name']} | "
             f"prob={p['probability_percent']}% | "
             f"severity={p['severity_tier']}")

    # Explanation should have summary
    expl = d["explanation"]
    assert "summary" in expl or "details" in expl, f"Explanation missing summary/details: {expl.keys()}"
    info(f"  Explanation: {(expl.get('summary') or expl.get('details', ''))[:80]}...")

    # Check processing time
    info(f"  Processing time: {env['meta']['processing_time_ms']}ms")


# ---------- 6. Validate endpoint ------------------------------------------

def test_validate_no_file():
    r = requests.post(f"{API_BASE}/validate", timeout=10)
    assert r.status_code == 400
    assert_envelope(r, "error")


def test_validate_with_audio():
    audio = find_sample_audio()
    if audio is None:
        raise Exception("No sample audio file found – skipping")

    with open(audio, "rb") as f:
        files = {"audio": (audio.name, f, "audio/wav")}
        data = {"task_type": "sustained_vowel"}
        r = requests.post(f"{API_BASE}/validate", files=files, data=data, timeout=60)

    # 200 (quality OK) or 422 (quality failure) are both valid
    assert r.status_code in (200, 422), f"Status {r.status_code}: {r.text[:200]}"
    expected = "success" if r.status_code == 200 else "error"
    env = assert_envelope(r, expected)
    if r.status_code == 200:
        assert "quality" in env["data"]
    info(f"  Validation result: {r.status_code}")


# ---------- 7. Demo analyze -----------------------------------------------

def test_demo_analyze():
    audio = find_sample_audio()
    if audio is None:
        raise Exception("No sample audio file found – skipping")

    info(f"  Using audio file: {audio.name}")
    with open(audio, "rb") as f:
        files = {"audio": (audio.name, f, "audio/wav")}
        data = {"task_type": "sustained_vowel"}
        r = requests.post(f"{API_BASE}/demo/analyze", files=files, data=data, timeout=120)

    if r.status_code == 422:
        assert_envelope(r, "error")
        info("  Quality gate failure (expected for some files)")
        return

    assert r.status_code == 200, f"Status {r.status_code}: {r.text[:200]}"
    env = assert_envelope(r, "success")
    d = env["data"]

    assert "quality" in d
    assert "features" in d
    assert "predictions" in d
    assert "explanation" in d
    # Demo also returns visualizations
    info(f"  Has visualizations: {'visualizations' in d}")


# ── Runner ───────────────────────────────────────────────────────────────────

def main():
    header("Voice Health Analysis API — Test Suite")

    # Check connectivity first
    info("Checking server connectivity...")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        fail("Cannot connect to server at " + BASE_URL)
        warn("Start the server with:  cd server && python app.py")
        sys.exit(1)

    if r.status_code != 200:
        fail(f"Health endpoint returned {r.status_code}")
        sys.exit(1)

    ok("Server is reachable\n")

    # Run tests
    tests = [
        ("Health check", test_health),
        ("Tasks list", test_tasks_list),
        ("Task detail (sustained_vowel)", test_task_detail),
        ("Task not found (404)", test_task_not_found),
        ("Analyze — no file (400)", test_analyze_no_file),
        ("Analyze — bad task type (400)", test_analyze_bad_task_type),
        ("Analyze — with audio", test_analyze_with_audio),
        ("Validate — no file (400)", test_validate_no_file),
        ("Validate — with audio", test_validate_with_audio),
        ("Demo analyze — with audio", test_demo_analyze),
    ]

    for name, fn in tests:
        run(name, fn)

    # Summary
    header("Results")
    total = passed + failed_count
    if failed_count == 0:
        ok(f"All {total} tests passed!")
    else:
        fail(f"{failed_count}/{total} tests failed")
        ok(f"{passed}/{total} tests passed")

    sys.exit(1 if failed_count else 0)


if __name__ == "__main__":
    main()
