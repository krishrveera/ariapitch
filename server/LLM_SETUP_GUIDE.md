# LLM Setup Guide

The Voice Health Analysis API uses LLMs to generate user-friendly explanations of voice analysis results. You can choose between Google Gemini or Anthropic Claude.

## Option 1: Google Gemini (Recommended)

**Advantages:**
- Free tier available (60 requests per minute)
- Fast response times
- No credit card required for API key
- Good at following structured output formats

### Setup Steps

1. **Get API Key**
   - Visit: https://makersuite.google.com/app/apikey
   - Click "Create API Key"
   - Copy your API key

2. **Configure Environment**

   Edit your `.env` file:
   ```bash
   LLM_PROVIDER=gemini
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

3. **Install Package**
   ```bash
   pip install google-generativeai
   ```

4. **Test It**
   ```bash
   python -c "import google.generativeai as genai; print('Gemini SDK installed!')"
   ```

### Pricing (as of 2026)

- **Free Tier**: 60 requests/minute
- **Gemini 1.5 Flash**: Fast, optimized for throughput
- **Gemini 1.5 Pro**: More capable, slower

The server uses Gemini 1.5 Flash by default for speed.

---

## Option 2: Anthropic Claude (Alternative)

**Advantages:**
- Very high quality explanations
- Strong medical/healthcare knowledge
- Excellent at following instructions

**Disadvantages:**
- Requires credit card
- Paid API (though still affordable)

### Setup Steps

1. **Get API Key**
   - Visit: https://console.anthropic.com/
   - Sign up and add payment method
   - Create an API key

2. **Configure Environment**

   Edit your `.env` file:
   ```bash
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=your_actual_api_key_here
   ```

3. **Install Package**
   ```bash
   pip install anthropic
   ```

### Pricing (as of 2026)

- **Claude Sonnet 4**: ~$0.003 per request (typical usage)
- Input: $3 per million tokens
- Output: $15 per million tokens

---

## Running Without an LLM

The server will work without an LLM API key, but explanations will be generic fallback messages.

**Fallback behavior:**
- Feature extraction still works
- Quality gate still works
- Predictions still work
- Explanations are pre-written generic messages

---

## Switching Between Providers

The server supports automatic fallback. If your preferred provider fails, it will try the other.

**Priority order:**
1. Your `LLM_PROVIDER` setting
2. Gemini (if API key exists)
3. Anthropic (if API key exists)
4. Fallback to generic messages

**Example multi-provider setup:**
```bash
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_key
ANTHROPIC_API_KEY=your_anthropic_key
```

With this setup:
- Primary: Gemini
- Fallback: Claude (if Gemini fails)

---

## Testing Your LLM Setup

### 1. Check if LLM is configured

```bash
curl http://localhost:5000/api/v1/health
```

Look for `model_loaded: true` in the response.

### 2. Test with actual audio

```bash
curl -X POST http://localhost:5000/api/v1/analyze \
  -F "audio=@test.m4a" \
  -F "device_id=test" \
  -F "task_type=sustained_vowel" | jq '.data.explanation'
```

You should see a natural-language explanation, not a generic fallback.

---

## Troubleshooting

### "Module not found" error

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall the LLM package
pip install google-generativeai  # or anthropic
```

### "API key not found" error

Check your `.env` file exists and has the correct key:
```bash
cat .env | grep API_KEY
```

### "Rate limit exceeded"

**Gemini free tier:**
- Limit: 60 requests/minute
- Solution: Wait a minute, or upgrade to paid tier

**Claude:**
- Depends on your tier
- Solution: Check console.anthropic.com for limits

### Generic explanations appearing

The LLM isn't being called. Check:
1. `.env` file has correct API key
2. LLM package is installed
3. Server was restarted after adding API key

---

## Recommended Configuration

For a hackathon or development:
```bash
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_key_here
```

For production with high quality requirements:
```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_backup_key_here  # Fallback
```

For offline/no-cost development:
```bash
# Leave both commented out - uses fallback messages
# LLM_PROVIDER=gemini
# GOOGLE_API_KEY=
```

---

## Cost Estimates

### Gemini (Free Tier)
- **Per request**: $0
- **Monthly**: $0
- **Limit**: 60 requests/minute

### Gemini (Paid)
- **Per request**: ~$0.0001
- **1000 requests**: ~$0.10

### Claude
- **Per request**: ~$0.003
- **1000 requests**: ~$3.00

For a typical screening app with 1000 users/month, each doing 2 analyses:
- **Gemini Free**: $0
- **Gemini Paid**: $0.20/month
- **Claude**: $6.00/month

---

## API Key Security

**Never commit API keys to git!**

✅ Good:
```bash
# .env file (gitignored)
GOOGLE_API_KEY=actual_key_here
```

❌ Bad:
```python
# Hardcoded in code (DON'T DO THIS)
api_key = "AIzaSyC..."
```

The `.gitignore` already excludes `.env` files.
