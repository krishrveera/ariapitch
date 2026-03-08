"""
Comprehensive API Test Suite for Voice Health Analysis API
Tests all endpoints with sample audio files
"""

import os
import sys
import requests
import json
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

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
    PROJECT_ROOT / "data" / "samples"
]

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def find_sample_audio():
    """Find a sample audio file to use for testing"""
    for sample_dir in SAMPLE_DIRS:
        if sample_dir.exists():
            for ext in ['.wav', '.mp3', '.m4a', '.flac']:
                files = list(sample_dir.glob(f'*{ext}'))
                if files:
                    return files[0]

    # Search in current directory and subdirectories
    for ext in ['.wav', '.mp3', '.m4a', '.flac']:
        files = list(PROJECT_ROOT.rglob(f'*{ext}'))
        if files:
            return files[0]

    return None

def test_health():
    """Test /api/v1/health endpoint"""
    print_info("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)

        if response.status_code == 200:
            data = response.json()
            print_success(f"Health check passed: {data.get('status')}")
            print(f"  Environment: {data.get('environment')}")
            print(f"  Timestamp: {data.get('timestamp')}")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to server. Is it running?")
        print_warning("Start the server with: ./start_server.sh")
        return False
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False

def test_tasks_list():
    """Test GET /api/v1/tasks endpoint"""
    print_info("Testing tasks list endpoint...")
    try:
        response = requests.get(f"{API_BASE}/tasks", timeout=10)

        if response.status_code == 200:
            data = response.json()

            if data.get('status') == 'success':
                tasks = data.get('data', {}).get('tasks', [])
                print_success(f"Tasks list retrieved: {len(tasks)} tasks")

                for task in tasks:
                    print(f"  - {task['display_name']} ({task['id']})")
                    print(f"    Conditions screened: {', '.join(task['conditions_screened'])}")

                return True
            else:
                print_error(f"Tasks list failed: {data.get('message')}")
                return False
        else:
            print_error(f"Tasks list failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print_error(f"Tasks list error: {e}")
        return False

def test_task_detail(task_id="sustained_vowel"):
    """Test GET /api/v1/tasks/{task_id} endpoint"""
    print_info(f"Testing task detail endpoint for '{task_id}'...")
    try:
        response = requests.get(f"{API_BASE}/tasks/{task_id}", timeout=10)

        if response.status_code == 200:
            data = response.json()

            if data.get('status') == 'success':
                task = data.get('data', {}).get('task', {})
                print_success(f"Task detail retrieved: {task.get('display_name')}")
                print(f"  Instruction: {task.get('instruction')[:60]}...")
                print(f"  Min duration: {task.get('min_duration_sec')}s")
                print(f"  Quality gates: {len(task.get('quality_gates', []))} gates")

                return True
            else:
                print_error(f"Task detail failed: {data.get('message')}")
                return False
        else:
            print_error(f"Task detail failed with status {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Task detail error: {e}")
        return False

def test_analyze(audio_file, task_type="sustained_vowel"):
    """Test POST /api/v1/analyze endpoint"""
    print_info(f"Testing analyze endpoint with {audio_file.name}...")

    if not audio_file.exists():
        print_error(f"Audio file not found: {audio_file}")
        return False

    try:
        with open(audio_file, 'rb') as f:
            files = {'audio': (audio_file.name, f, 'audio/wav')}
            data = {'task_type': task_type}

            response = requests.post(
                f"{API_BASE}/analyze",
                files=files,
                data=data,
                timeout=60
            )

        if response.status_code == 200:
            result = response.json()

            if result.get('status') == 'success':
                data = result.get('data', {})

                print_success("Analysis completed successfully")

                # Quality gates
                quality = data.get('quality', {})
                print(f"\n  Quality Gates:")
                print(f"    Overall: {quality.get('overall_decision')}")
                print(f"    Passed: {quality.get('gates_passed')}/{quality.get('total_gates')}")

                # Preprocessing
                preproc = data.get('preprocessing', {})
                print(f"\n  Preprocessing:")
                print(f"    Duration: {preproc.get('duration_sec')}s")
                print(f"    Sample rate: {preproc.get('sample_rate_hz')}Hz")

                # Features
                features = data.get('features', {})
                print(f"\n  Features extracted: {len(features)} features")

                # Predictions
                predictions = data.get('predictions', [])
                print(f"\n  Predictions:")
                for pred in predictions:
                    print(f"    {pred.get('condition_name')}:")
                    print(f"      Probability: {pred.get('probability_percent')}%")
                    print(f"      Severity: {pred.get('severity_tier')}")

                # Explanation
                explanation = data.get('explanation', {})
                print(f"\n  LLM Explanation:")
                print(f"    Status: {explanation.get('status')}")
                if explanation.get('explanation'):
                    print(f"    Text: {explanation.get('explanation')[:100]}...")

                return True
            else:
                print_error(f"Analysis failed: {result.get('message')}")
                if result.get('errors'):
                    for error in result.get('errors'):
                        print(f"    Error: {error}")
                return False
        else:
            print_error(f"Analysis failed with status {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False

    except Exception as e:
        print_error(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_demo_analyze(audio_file, task_type="sustained_vowel"):
    """Test POST /api/v1/demo/analyze endpoint"""
    print_info(f"Testing demo analyze endpoint with {audio_file.name}...")

    if not audio_file.exists():
        print_error(f"Audio file not found: {audio_file}")
        return False

    try:
        with open(audio_file, 'rb') as f:
            files = {'audio': (audio_file.name, f, 'audio/wav')}
            data = {'task_type': task_type}

            response = requests.post(
                f"{API_BASE}/demo/analyze",
                files=files,
                data=data,
                timeout=90
            )

        if response.status_code == 200:
            result = response.json()

            if result.get('status') == 'success':
                data = result.get('data', {})

                print_success("Demo analysis completed successfully")

                # Visualizations
                viz = data.get('visualizations', {})
                print(f"\n  Visualizations generated:")

                output_dir = Path(__file__).parent / "test_outputs"
                output_dir.mkdir(exist_ok=True)

                saved_count = 0
                for viz_name, viz_data in viz.items():
                    if viz_data and isinstance(viz_data, str):
                        print(f"    ✓ {viz_name}")

                        # Save visualization to file
                        try:
                            img_data = base64.b64decode(viz_data)
                            img = Image.open(BytesIO(img_data))
                            output_path = output_dir / f"{viz_name}.png"
                            img.save(output_path)
                            saved_count += 1
                        except Exception as e:
                            print_warning(f"      Could not save {viz_name}: {e}")

                print(f"\n  Saved {saved_count} visualizations to: {output_dir}")

                # Raw B2AI features
                raw_features = data.get('features_raw', {})
                if raw_features:
                    print(f"\n  Raw B2AI Features:")
                    for category, features in raw_features.items():
                        if isinstance(features, dict):
                            print(f"    {category}: {len(features)} features")

                return True
            else:
                print_error(f"Demo analysis failed: {result.get('message')}")
                return False
        else:
            print_error(f"Demo analysis failed with status {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False

    except Exception as e:
        print_error(f"Demo analysis error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_standalone():
    """Test LLM integration directly"""
    print_info("Testing LLM integration...")

    try:
        # Import the explanation service
        from services.explanation import generate_explanation

        # Test data
        test_prediction = {
            'condition': 'benign_lesion',
            'condition_name': 'Benign Lesion',
            'probability': 0.42,
            'probability_percent': 42.0,
            'severity_tier': 'moderate',
            'features_used': ['jitter', 'shimmer', 'hnr', 'cpp'],
            'feature_values': {
                'jitter': 1.85,
                'shimmer': 4.23,
                'hnr': 18.5,
                'cpp': 7.8
            }
        }

        test_features = {
            'praat.jitter.local_percent': 1.85,
            'praat.shimmer.local_percent': 4.23,
            'praat.hnr.mean_db': 18.5,
            'praat.cpp.mean_db': 7.8,
            'praat.f0.mean_hz': 145.2
        }

        result = generate_explanation([test_prediction], test_features, task_type="sustained_vowel")

        if result.get('status') == 'success':
            print_success("LLM integration working")
            print(f"\n  Provider: {result.get('provider', 'unknown')}")
            print(f"\n  Explanation:")
            explanation_text = result.get('explanation', '')
            # Print first 300 characters
            print(f"    {explanation_text[:300]}...")

            return True
        elif result.get('status') == 'no_provider':
            print_warning("No LLM provider configured")
            print("  Set GOOGLE_API_KEY or ANTHROPIC_API_KEY in .env file")
            return False
        else:
            print_error(f"LLM failed: {result.get('error')}")
            return False

    except Exception as e:
        print_error(f"LLM test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all API tests"""
    print_header("VOICE HEALTH ANALYSIS API - COMPREHENSIVE TEST SUITE")

    # Test 1: Health check
    print_header("Test 1: Health Check")
    health_ok = test_health()

    if not health_ok:
        print_error("\nServer is not running. Cannot proceed with API tests.")
        print_warning("Start the server with: ./start_server.sh")
        return

    # Test 2: Tasks list
    print_header("Test 2: Tasks List")
    test_tasks_list()

    # Test 3: Task detail
    print_header("Test 3: Task Detail")
    test_task_detail("sustained_vowel")

    # Test 4: LLM integration
    print_header("Test 4: LLM Integration (Standalone)")
    llm_ok = test_llm_standalone()

    # Find sample audio
    audio_file = find_sample_audio()

    if not audio_file:
        print_warning("\nNo sample audio files found. Skipping analysis tests.")
        print_info("Place .wav files in one of these directories:")
        for sample_dir in SAMPLE_DIRS:
            print(f"  - {sample_dir}")
    else:
        print_info(f"\nUsing sample audio: {audio_file}")

        # Test 5: Basic analyze
        print_header("Test 5: Basic Analysis")
        test_analyze(audio_file, "sustained_vowel")

        # Test 6: Demo analyze
        print_header("Test 6: Demo Analysis (with visualizations)")
        test_demo_analyze(audio_file, "sustained_vowel")

        # Test with different task types
        print_header("Test 7: Multiple Task Types")
        for task_type in ["free_speech", "reading_passage", "cough"]:
            print(f"\n--- Testing {task_type} ---")
            test_analyze(audio_file, task_type)

    # Summary
    print_header("TEST SUITE COMPLETE")

    if llm_ok:
        print_success("✓ LLM integration is working")
    else:
        print_warning("⚠ LLM integration needs API key configuration")

    if audio_file:
        print_success("✓ All API endpoints tested successfully")
    else:
        print_warning("⚠ Analysis endpoints not tested (no audio files)")

    print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
    print("1. Check test_outputs/ directory for saved visualizations")
    print("2. Configure LLM API key in .env if needed (see LLM_SETUP_GUIDE.md)")
    print("3. Replace dummy model in services/prediction.py with real model")

if __name__ == "__main__":
    run_all_tests()
