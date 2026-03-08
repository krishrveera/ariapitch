"""
Standalone LLM Test - Tests Gemini and Claude integration
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded .env from {env_path}")
else:
    print(f"⚠ No .env file found at {env_path}")

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

def test_llm():
    """Test LLM integration"""
    print_header("LLM INTEGRATION TEST")

    # Check environment variables
    print("Checking API Keys:")
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    claude_key = os.environ.get("ANTHROPIC_API_KEY")
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()

    print(f"  LLM_PROVIDER: {provider}")
    print(f"  GOOGLE_API_KEY: {'✓ Set' if gemini_key else '✗ Not set'}")
    print(f"  ANTHROPIC_API_KEY: {'✓ Set' if claude_key else '✗ Not set'}")

    if not gemini_key and not claude_key:
        print(f"\n{Colors.RED}✗ No API keys configured{Colors.END}")
        print(f"\n{Colors.YELLOW}Setup instructions:{Colors.END}")
        print("1. Copy .env.example to .env")
        print("2. Add your API key:")
        print("   - For Gemini (free): GOOGLE_API_KEY=your_key_here")
        print("   - For Claude: ANTHROPIC_API_KEY=your_key_here")
        print("\nSee LLM_SETUP_GUIDE.md for details")
        return False

    # Import and test
    try:
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
            },
            'reference_thresholds': {
                'jitter_local_percent': {'normal_max': 1.04},
                'shimmer_local_percent': {'normal_max': 3.81},
                'hnr_db': {'normal_min': 20.0},
                'cpp_db': {'normal_min': 8.0}
            }
        }

        test_features = {
            'praat.jitter.local_percent': 1.85,
            'praat.shimmer.local_percent': 4.23,
            'praat.hnr.mean_db': 18.5,
            'praat.cpp.mean_db': 7.8,
            'praat.f0.mean_hz': 145.2,
            'praat.formants.f1.mean_hz': 720.3,
            'praat.formants.f2.mean_hz': 1250.8
        }

        print(f"\n{Colors.BLUE}Generating explanation...{Colors.END}")
        result = generate_explanation(test_features, [test_prediction], task_type="sustained_vowel")

        # Check if result contains expected fields
        if result and 'summary' in result:
            print(f"\n{Colors.GREEN}✓ LLM integration successful!{Colors.END}")
            print(f"\n{Colors.BOLD}Summary:{Colors.END}")
            print(result.get('summary', 'N/A'))
            print(f"\n{Colors.BOLD}Details:{Colors.END}")
            print(result.get('details', 'N/A'))
            print(f"\n{Colors.BOLD}Disclaimer:{Colors.END}")
            print(result.get('disclaimer', 'N/A'))
            print(f"\n{Colors.BOLD}Model Version:{Colors.END} {result.get('model_version', 'N/A')}")

            return True

        else:
            print(f"\n{Colors.RED}✗ LLM failed - unexpected result format{Colors.END}")
            print(f"Result: {result}")
            return False

    except Exception as e:
        print(f"\n{Colors.RED}✗ Test error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llm()
    sys.exit(0 if success else 1)
