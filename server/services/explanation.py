"""
LLM Explanation Service
Generates human-readable explanations using Google Gemini or Claude AI.
"""
import json
import os

# Try to import both LLM providers
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Determine which LLM provider to use
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()  # "gemini" or "anthropic"

# Initialize clients based on available API keys and provider preference
gemini_client = None
anthropic_client = None

if GEMINI_AVAILABLE and os.environ.get("GOOGLE_API_KEY"):
    gemini_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
    anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _call_gemini(prompt: str, max_tokens: int = 1000) -> str:
    """Call Google Gemini API using new google.genai package."""
    if gemini_client is None:
        raise ValueError("Gemini client not initialized")

    try:
        # Use the latest available Gemini flash model with JSON mode
        response = gemini_client.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=prompt,
            config={
                "max_output_tokens": max_tokens,
                "temperature": 0.7,
                "response_mime_type": "application/json",  # Force JSON output
            }
        )

        # Check if response was truncated
        if response.candidates:
            finish_reason = str(response.candidates[0].finish_reason)
            if 'MAX_TOKENS' in finish_reason or 'LENGTH' in finish_reason:
                print(f"Warning: Gemini response was truncated ({finish_reason})")

        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        raise


def _call_anthropic(prompt: str, max_tokens: int = 1000) -> str:
    """Call Anthropic Claude API."""
    if anthropic_client is None:
        raise ValueError("Anthropic client not initialized")

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def _call_llm(prompt: str, max_tokens: int = 1000) -> str:
    """
    Call the configured LLM provider.

    Tries providers in this order:
    1. User's preferred provider (LLM_PROVIDER env var)
    2. Gemini (if available)
    3. Anthropic (if available)
    4. Returns None if no provider available
    """
    # Try preferred provider first
    if LLM_PROVIDER == "gemini" and gemini_client:
        try:
            return _call_gemini(prompt, max_tokens)
        except Exception:
            # Fall through to try other provider
            pass

    if LLM_PROVIDER == "anthropic" and anthropic_client:
        try:
            return _call_anthropic(prompt, max_tokens)
        except Exception:
            pass

    # Try fallback providers
    if gemini_client and LLM_PROVIDER != "gemini":
        try:
            return _call_gemini(prompt, max_tokens)
        except Exception:
            pass

    if anthropic_client and LLM_PROVIDER != "anthropic":
        try:
            return _call_anthropic(prompt, max_tokens)
        except Exception:
            pass

    return None


def generate_explanation(features: dict, predictions: list, task_type: str) -> dict:
    """
    Generate a human-readable explanation of prediction results.

    The LLM receives extracted features, model predictions, and clinical thresholds.
    It produces plain-language explanations without making diagnoses.

    Supports both Google Gemini and Anthropic Claude.
    """
    # Fallback response when no LLM is available
    fallback = {
        "summary": "Analysis complete. Your voice features have been extracted.",
        "details": "Voice analysis results are available. All measurements have been processed using clinical reference standards.",
        "disclaimer": (
            "This analysis is for informational and screening purposes only. "
            "It is not a medical diagnosis. If you have concerns about your "
            "voice or health, please consult a healthcare professional."
        ),
        "model_version": "b2ai-voice-v0.1.0"
    }

    prompt = f"""You are a voice health analysis assistant integrated into a screening app. Your job is to explain voice analysis results to the user in clear, non-alarming language.

You are NOT making a diagnosis. An ML model has already produced risk percentages. You are explaining what the numbers mean.

VOICE FEATURES EXTRACTED:
{json.dumps(features, indent=2)}

MODEL PREDICTIONS:
{json.dumps(predictions, indent=2)}

TASK TYPE: {task_type}

CLINICAL REFERENCE RANGES (from published literature):
- Jitter (local): normal < 1.04% (Teixeira et al., 2013)
- Shimmer (local): normal < 3.81% (Teixeira et al., 2013)
- HNR: normal > 20 dB (Boersma, 1993)
- CPP: concerning if < 8 dB (Heman-Ackah et al., 2003)
- F0 adult male: 85-180 Hz, adult female: 165-255 Hz

Write your response as JSON with exactly these fields:
{{
  "summary": "1-2 sentences. Plain language. If all risks are low, say so reassuringly. If any risk is elevated, mention it factually without alarming.",
  "details": "One paragraph. Reference specific feature values and whether they fall within normal clinical ranges. Explain which features influenced each prediction and why. Be specific with numbers but accessible in language."
}}

Rules:
- Never say "you have [disease]" or "you are diagnosed with"
- Use phrases like "your voice shows characteristics consistent with..." or "your measurements fall within the normal range for..."
- If a feature is affected by AGC or noise reduction, note that it may be less reliable
- Always be factual and measured in tone
- Reference the clinical threshold values when comparing features"""

    try:
        text = _call_llm(prompt, max_tokens=2000)

        if text is None:
            # No LLM available
            return fallback

        # Parse LLM response - handle markdown code blocks and extract JSON
        cleaned = text.strip()

        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json, ```, or just the language name)
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # Try to extract JSON object if there's still extra text
        if not cleaned.startswith("{"):
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                cleaned = cleaned[start_idx:end_idx]

        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Fallback on JSON parsing error
        print(f"LLM explanation JSON parse error: {e}")
        print(f"Cleaned text (first 500 chars): {cleaned[:500]}")
        print(f"Full cleaned text:\n{cleaned}")
        parsed = {
            "summary": "Analysis complete. Please review the detailed results below.",
            "details": "Your voice features have been extracted and analyzed against clinical reference ranges."
        }
    except Exception as e:
        # Fallback on other errors
        print(f"LLM explanation error: {e}")
        parsed = {
            "summary": "Analysis complete. Please review the detailed results below.",
            "details": "Your voice features have been extracted and analyzed against clinical reference ranges."
        }

    parsed["disclaimer"] = (
        "This analysis is for informational and screening purposes only. "
        "It is not a medical diagnosis. If you have concerns about your "
        "voice or health, please consult a healthcare professional."
    )
    parsed["model_version"] = "b2ai-voice-v0.1.0"

    return parsed


def generate_quality_failure_suggestion(gate_result: dict) -> str:
    """
    Generate helpful suggestion when quality gate fails.

    For single failures, uses templates. For multiple failures, uses LLM.
    """
    failed = gate_result["failed_checks"]

    # Single failure templates (fast, no LLM needed)
    if len(failed) == 1:
        check = failed[0]
        templates = {
            "snr": "Try recording in a quieter room. Close windows and doors, turn off fans or air conditioning, and hold the phone about 15-20 cm from your mouth.",
            "clipping": "The volume was too high. Try holding the phone slightly further from your mouth, or speak at a slightly lower volume.",
            "voiced_duration": "We need a longer recording. Make sure to sustain the sound for the full duration shown on screen.",
            "silence_region": "Make sure to stay completely quiet during the countdown before you start speaking."
        }
        return templates.get(check["check"], "Please try recording again.")

    # Multiple failures - use LLM for combined advice
    # Convert failed checks to JSON-serializable format
    from api.response import _convert_numpy_types
    failed_clean = _convert_numpy_types(failed)

    prompt = f"""A user tried to record their voice for health analysis but the recording failed quality checks. Generate ONE short paragraph (2-3 sentences) of friendly, practical advice addressing all issues.

Failed checks:
{json.dumps(failed_clean, indent=2)}

Be specific and actionable. Don't list the technical details — translate them into simple instructions the user can follow."""

    try:
        text = _call_llm(prompt, max_tokens=200)
        if text:
            return text.strip()
    except Exception as e:
        print(f"LLM suggestion error: {e}")

    # Fallback
    return "Please try recording again in a quieter environment with proper technique."
