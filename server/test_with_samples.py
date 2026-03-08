#!/usr/bin/env python3
"""
Test Script for Voice Analysis Server
Tests the API with sample audio files and displays results.
"""
import requests
import json
import sys
import os
from pathlib import Path
import base64
from PIL import Image
import io


BASE_URL = "http://localhost:5000/api/v1"


def save_visualization(base64_img: str, output_path: str):
    """Save base64-encoded image to file."""
    img_data = base64.b64decode(base64_img)
    img = Image.open(io.BytesIO(img_data))
    img.save(output_path)
    print(f"   Saved: {output_path}")


def test_health():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        data = response.json()

        print(f"Status: {response.status_code}")
        print(f"Server version: {data['data'].get('version')}")
        print(f"Model loaded: {data['data'].get('model_loaded')}")
        print(f"GPU available: {data['data'].get('gpu_available')}")

        return response.status_code == 200

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server")
        print(f"   Expected server at: {BASE_URL}")
        print("\n💡 To start the server:")
        print("   cd server")
        print("   source venv/bin/activate")
        print("   python app.py")
        return False

    except requests.exceptions.JSONDecodeError:
        print("❌ Server returned invalid JSON")
        print(f"   Response: {response.text[:200]}")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_tasks():
    """Test tasks endpoint."""
    print("\n" + "="*60)
    print("Testing /tasks endpoint")
    print("="*60)

    response = requests.get(f"{BASE_URL}/tasks")
    data = response.json()

    print(f"Status: {response.status_code}")
    print(f"Available tasks: {len(data['data']['tasks'])}")

    for task in data['data']['tasks']:
        print(f"\n  Task: {task['id']}")
        print(f"    Name: {task['display_name']}")
        print(f"    Min duration: {task['min_duration_sec']}s")
        print(f"    Screens for: {', '.join(task['conditions_screened'])}")

    return response.status_code == 200


def test_analyze(audio_file: str, task_type: str = "sustained_vowel"):
    """Test /analyze endpoint."""
    print("\n" + "="*60)
    print(f"Testing /analyze endpoint with: {audio_file}")
    print("="*60)

    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False

    with open(audio_file, 'rb') as f:
        files = {'audio': f}
        data = {
            'device_id': 'test_device',
            'task_type': task_type,
            'silence_duration_sec': '3.0'
        }

        response = requests.post(f"{BASE_URL}/analyze", files=files, data=data)

    result = response.json()

    print(f"Status: {response.status_code}")
    print(f"Message: {result['message']}")
    print(f"Processing time: {result['meta']['processing_time_ms']}ms")

    if result['status'] == 'success':
        print("\n✅ Analysis successful!")

        # Quality
        quality = result['data']['quality']
        print(f"\nQuality Gate:")
        print(f"  Passed: {quality['gate_passed']}")
        print(f"  SNR: {quality['snr_db']} dB ({quality['snr_verdict']})")
        print(f"  Voiced duration: {quality['voiced_duration_sec']}s")

        # Features (show first 5)
        features = result['data']['features']
        print(f"\nExtracted Features (showing first 5):")
        for i, (key, value) in enumerate(list(features.items())[:5]):
            print(f"  {key}: {value}")
        print(f"  ... ({len(features)} total features)")

        # Predictions
        predictions = result['data']['predictions']
        print(f"\nPredictions:")
        for pred in predictions:
            print(f"  {pred['condition_name']}: {pred['risk_percent']}% risk ({pred['severity_tier']})")

        # Explanation
        explanation = result['data']['explanation']
        print(f"\nExplanation:")
        print(f"  {explanation['summary']}")

    else:
        print(f"\n❌ Analysis failed: {result['message']}")
        if result['errors']:
            print(f"\nError type: {result['errors']['type']}")
            if 'failed_checks' in result['errors']:
                print("Failed checks:")
                for check in result['errors']['failed_checks']:
                    print(f"  - {check['check']}: {check['message']}")

    return response.status_code == 200


def test_demo_analyze(audio_file: str, task_type: str = "sustained_vowel", output_dir: str = "demo_output"):
    """Test /demo/analyze endpoint and save visualizations."""
    print("\n" + "="*60)
    print(f"Testing /demo/analyze endpoint with: {audio_file}")
    print("="*60)

    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    with open(audio_file, 'rb') as f:
        files = {'audio': f}
        data = {
            'device_id': 'test_device',
            'task_type': task_type,
            'silence_duration_sec': '3.0'
        }

        response = requests.post(f"{BASE_URL}/demo/analyze", files=files, data=data)

    result = response.json()

    print(f"Status: {response.status_code}")
    print(f"Message: {result['message']}")
    print(f"Processing time: {result['meta']['processing_time_ms']}ms")

    if result['status'] == 'success':
        print("\n✅ Demo analysis successful!")

        data = result['data']

        # Save visualizations
        visualizations = data.get('visualizations', {})
        if visualizations:
            print(f"\n📊 Saving {len(visualizations)} visualizations to {output_dir}/")
            for viz_name, viz_data in visualizations.items():
                output_path = os.path.join(output_dir, f"{viz_name}.png")
                save_visualization(viz_data, output_path)

        # Show feature summary
        features_raw = data.get('features_raw', {})
        print(f"\n📐 Raw B2AI Features:")
        for key, value in list(features_raw.items())[:10]:
            if isinstance(value, dict) and 'shape' in value:
                print(f"  {key}: {value['type']} {value['shape']}")
            elif isinstance(value, dict):
                print(f"  {key}: {list(value.keys())}")
            else:
                print(f"  {key}: {value}")

        print(f"\n🎯 Open {output_dir}/ to view all visualizations!")

    else:
        print(f"\n❌ Demo analysis failed: {result['message']}")

    return response.status_code == 200


def find_sample_audio_files():
    """Find sample audio files in the project."""
    search_paths = [
        "../adult/audio",  # Relative to server directory
        "./test_audio",
        "./samples",
        "../samples",
    ]

    audio_extensions = ['.wav', '.m4a', '.mp3', '.flac']
    found_files = []

    for search_path in search_paths:
        if os.path.exists(search_path):
            for ext in audio_extensions:
                files = list(Path(search_path).glob(f"**/*{ext}"))
                found_files.extend(files)

    return found_files


def main():
    """Run all tests."""
    print("=" * 60)
    print("Voice Analysis Server Test Suite")
    print("=" * 60)

    # Test server health
    if not test_health():
        print("\n❌ Server health check failed. Is the server running?")
        print("   Start server with: python app.py")
        return 1

    # Test tasks endpoint
    test_tasks()

    # Find sample audio files
    print("\n" + "="*60)
    print("Looking for sample audio files...")
    print("="*60)

    sample_files = find_sample_audio_files()

    if not sample_files:
        print("\n⚠️  No sample audio files found.")
        print("   You can:")
        print("   1. Place test audio files in ./test_audio/ or ./samples/")
        print("   2. Generate a test file with: python test_with_samples.py --generate")
        print("   3. Specify a file: python test_with_samples.py --file path/to/audio.wav")
        return 1

    print(f"\n📁 Found {len(sample_files)} sample audio files:")
    for i, f in enumerate(sample_files[:5]):
        print(f"   {i+1}. {f}")
    if len(sample_files) > 5:
        print(f"   ... and {len(sample_files) - 5} more")

    # Test with first sample file
    test_file = str(sample_files[0])

    # Test basic analyze
    print("\n" + "="*60)
    print("Testing Basic /analyze Endpoint")
    print("="*60)
    test_analyze(test_file)

    # Test demo analyze with visualizations
    print("\n" + "="*60)
    print("Testing Demo /demo/analyze Endpoint")
    print("="*60)
    test_demo_analyze(test_file, output_dir=f"demo_output_{Path(test_file).stem}")

    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test voice analysis server")
    parser.add_argument("--file", "-f", help="Specific audio file to test")
    parser.add_argument("--task", "-t", default="sustained_vowel",
                       choices=["sustained_vowel", "free_speech", "reading_passage", "cough"],
                       help="Task type")
    parser.add_argument("--demo-only", action="store_true",
                       help="Only run demo endpoint (with visualizations)")
    parser.add_argument("--url", default="http://localhost:5000/api/v1",
                       help="Base URL of the API")

    args = parser.parse_args()

    BASE_URL = args.url

    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            sys.exit(1)

        if args.demo_only:
            success = test_demo_analyze(args.file, args.task)
        else:
            test_health()
            success = test_analyze(args.file, args.task)
            test_demo_analyze(args.file, args.task)

        sys.exit(0 if success else 1)
    else:
        sys.exit(main())
