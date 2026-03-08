#!/usr/bin/env python3
"""
Test LLM Integration
Quick script to verify your LLM setup is working correctly.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini():
    """Test Google Gemini integration."""
    print("\n🧪 Testing Google Gemini...")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env")
        return False

    print(f"✓ API key found: {api_key[:10]}...")

    try:
        import google.generativeai as genai
        print("✓ google-generativeai package installed")
    except ImportError:
        print("❌ google-generativeai not installed. Run: pip install google-generativeai")
        return False

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content("Say 'Hello from Gemini!' and nothing else.")
        result = response.text.strip()

        print(f"✓ API call successful!")
        print(f"  Response: {result}")
        return True
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False


def test_anthropic():
    """Test Anthropic Claude integration."""
    print("\n🧪 Testing Anthropic Claude...")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ℹ️  ANTHROPIC_API_KEY not found (optional)")
        return None

    print(f"✓ API key found: {api_key[:10]}...")

    try:
        import anthropic
        print("✓ anthropic package installed")
    except ImportError:
        print("❌ anthropic not installed. Run: pip install anthropic")
        return False

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'Hello from Claude!' and nothing else."}]
        )
        result = response.content[0].text.strip()

        print(f"✓ API call successful!")
        print(f"  Response: {result}")
        return True
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False


def test_explanation_service():
    """Test the actual explanation service."""
    print("\n🧪 Testing Explanation Service...")

    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from services.explanation import generate_explanation

        # Mock data
        features = {
            "f0_mean_hz": 120.0,
            "jitter_local_percent": 0.5,
            "shimmer_local_percent": 2.0,
            "hnr_db": 22.0,
            "cpp_db": 14.0
        }

        predictions = [
            {
                "condition": "parkinsons",
                "risk_percent": 0.0,
                "confidence": 0.0,
                "severity_tier": "low"
            }
        ]

        result = generate_explanation(features, predictions, "sustained_vowel")

        print("✓ Explanation service working!")
        print(f"  Summary: {result['summary'][:80]}...")

        if "Analysis complete" in result['summary']:
            print("  ⚠️  Using fallback (no LLM). Check your API keys.")
        else:
            print("  ✅ LLM is being used!")

        return True
    except Exception as e:
        print(f"❌ Explanation service failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("LLM Integration Test")
    print("=" * 60)

    provider = os.environ.get("LLM_PROVIDER", "gemini")
    print(f"\nConfigured provider: {provider}")

    results = {}

    # Test Gemini
    results['gemini'] = test_gemini()

    # Test Claude (optional)
    results['anthropic'] = test_anthropic()

    # Test the actual service
    results['service'] = test_explanation_service()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if results['gemini']:
        print("✅ Gemini is working")
    else:
        print("❌ Gemini is NOT working")

    if results['anthropic'] is True:
        print("✅ Claude is working (optional)")
    elif results['anthropic'] is False:
        print("⚠️  Claude is configured but not working")
    else:
        print("ℹ️  Claude not configured (optional)")

    if results['service']:
        print("✅ Explanation service is working")
    else:
        print("❌ Explanation service is NOT working")

    print("\n" + "=" * 60)

    if results['gemini'] or results['anthropic']:
        print("🎉 At least one LLM provider is working!")
        return 0
    else:
        print("❌ No LLM providers are working. Check your API keys.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
