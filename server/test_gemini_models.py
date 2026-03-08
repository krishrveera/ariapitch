#!/usr/bin/env python3
"""Test script to check available Gemini models."""

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("❌ GOOGLE_API_KEY not set in .env")
    exit(1)

print(f"✓ API Key found: {api_key[:20]}...")

try:
    from google import genai
    print("✓ google-genai package imported")

    client = genai.Client(api_key=api_key)
    print("✓ Client created")

    # Try to list models
    print("\n📋 Attempting to list available models...")
    try:
        models = client.models.list()
        print(f"✓ Found {len(list(models))} models:")
        for model in models:
            print(f"  - {model.name}")
    except Exception as e:
        print(f"❌ Couldn't list models: {e}")

    # Try different ways to call the API
    print("\n🧪 Testing different API call methods...")

    test_prompt = "Say hello in 5 words or less."

    # Method 1: Using models.generate_content
    print("\n1. Trying client.models.generate_content with 'gemini-1.5-flash'...")
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=test_prompt
        )
        print(f"✅ SUCCESS: {response.text}")
    except Exception as e:
        print(f"❌ FAILED: {e}")

    # Method 2: Different model name
    print("\n2. Trying with 'gemini-pro'...")
    try:
        response = client.models.generate_content(
            model='gemini-pro',
            contents=test_prompt
        )
        print(f"✅ SUCCESS: {response.text}")
    except Exception as e:
        print(f"❌ FAILED: {e}")

    # Method 3: Using chat
    print("\n3. Trying client.chats.create...")
    try:
        chat = client.chats.create(model='gemini-1.5-flash')
        response = chat.send_message(test_prompt)
        print(f"✅ SUCCESS: {response.text}")
    except Exception as e:
        print(f"❌ FAILED: {e}")

except ImportError:
    print("❌ google-genai not installed. Run: pip install google-genai")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
