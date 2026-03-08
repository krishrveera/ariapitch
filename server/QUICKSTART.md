# Quick Start Guide

Get the Voice Health Analysis API running in 5 minutes!

## 🚀 Fast Track (Recommended)

### 1. Start the Server

```bash
cd server
./start_server.sh
```

This script will:
- Create virtual environment (if needed)
- Install dependencies (if needed)
- Create `.env` file from template
- Start the Flask server on port 5000

### 2. Configure API Keys (Optional but Recommended)

Edit `.env` and add your LLM API key:

```bash
# For Google Gemini (FREE, recommended)
GOOGLE_API_KEY=your_key_here

# Get your free key at: https://makersuite.google.com/app/apikey
```

**Without an API key:** The server works fine, but explanations will be generic fallback messages.

### 3. Test the Server

In a new terminal:

```bash
cd server
python test_with_samples.py --file path/to/your/audio.wav
```

Or use the included test audio (if you have B2AI data):

```bash
python test_with_samples.py
```

---

## 📋 Manual Setup (If You Prefer)

### 1. Create Virtual Environment

```bash
cd server
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This will install ~2GB of packages including PyTorch. For a faster start, you can install minimal dependencies:

```bash
# Minimal (no B2AI features, just Praat)
pip install Flask flask-cors librosa soundfile pyloudnorm noisereduce scipy numpy praat-parselmouth matplotlib Pillow werkzeug python-dotenv requests google-generativeai
```

### 3. Configure Environment

```bash
cp .env.example .env
nano .env  # Or use your favorite editor
```

Add your API key:

```
GOOGLE_API_KEY=your_actual_key_here
```

### 4. Run Server

```bash
python app.py
```

---

## ✅ Verify Installation

Once the server is running, open another terminal and test:

### Test 1: Health Check

```bash
curl http://localhost:5000/api/v1/health
```

Expected output:
```json
{
  "status": "success",
  "data": {
    "version": "1.0.0",
    "model_loaded": false
  }
}
```

### Test 2: Get Available Tasks

```bash
curl http://localhost:5000/api/v1/tasks
```

You should see 4 tasks: sustained_vowel, free_speech, reading_passage, cough.

### Test 3: Analyze Audio

```bash
curl -X POST http://localhost:5000/api/v1/analyze \
  -F "audio=@path/to/audio.wav" \
  -F "device_id=test" \
  -F "task_type=sustained_vowel"
```

---

## 🎤 Recording Test Audio

Don't have test audio? Generate or record some:

### Option 1: Record on macOS

```bash
# Record 10 seconds (3s silence + 7s speaking "ahhhh")
sox -d test_recording.wav trim 0 10
```

### Option 2: Record in Python

```python
import sounddevice as sd
import soundfile as sf

print("Recording in 3 seconds... Say 'ahhh' for 5 seconds!")
import time
time.sleep(3)

duration = 8  # 3s silence + 5s voice
fs = 44100

recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
sf.write('test_recording.wav', recording, fs)
print("Saved to test_recording.wav")
```

### Option 3: Use Online Tools

- https://online-voice-recorder.com/
- Record 3 seconds of silence
- Then say "ahhh" for 5 seconds
- Download as WAV

---

## 📊 Using the Demo Endpoint

For full visualizations:

```bash
curl -X POST http://localhost:5000/api/v1/demo/analyze \
  -F "audio=@test.wav" \
  -F "device_id=test" \
  -F "task_type=sustained_vowel" \
  -o response.json
```

Then extract visualizations with Python:

```python
import json
import base64

with open('response.json') as f:
    data = json.load(f)

for name, img_data in data['data']['visualizations'].items():
    with open(f'{name}.png', 'wb') as img_file:
        img_file.write(base64.b64decode(img_data))

print("Visualizations saved!")
```

Or use the test script:

```bash
python test_with_samples.py --file test.wav --demo-only
```

This automatically saves all visualizations to `demo_output_test/`!

---

## 🐛 Troubleshooting

### "Cannot connect to server"

**Problem:** Server isn't running.

**Solution:**
```bash
cd server
./start_server.sh
```

### "Module not found: senselab"

**Problem:** Dependencies not installed.

**Solution:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Quality gate failure"

**Problem:** Your audio doesn't meet quality requirements.

**Solutions:**
- Record in a quieter environment
- Hold phone 15-20cm from mouth
- Speak for the full required duration (5s for sustained vowel)
- Make sure there's 3 seconds of silence at the start

**Check visualizations** to see what failed:
```bash
python test_with_samples.py --file bad_audio.wav --demo-only
open demo_output_bad_audio/quality_gate_summary.png
```

### "Generic LLM explanations"

**Problem:** No API key configured.

**Solution:**
1. Get free Gemini key: https://makersuite.google.com/app/apikey
2. Add to `.env`: `GOOGLE_API_KEY=your_key`
3. Restart server

### "Slow feature extraction"

**Problem:** SenseLab with full B2AI features takes 4-8 seconds.

**Solutions:**
- **For development:** Uninstall SenseLab to use faster Praat fallback
  ```bash
  pip uninstall senselab torch torchaudio transformers opensmile
  ```
- **For production:** This is normal. Consider caching results.

---

## 🎯 Next Steps

### For Development

1. **Read the docs:**
   - [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - Full API reference
   - [B2AI_FEATURES.md](B2AI_FEATURES.md) - Feature extraction explained
   - [DEMO_ENDPOINT.md](DEMO_ENDPOINT.md) - Demo endpoint guide

2. **Integrate your ML model:**
   - Update `services/prediction.py`
   - Replace placeholder predictions with your model

3. **Customize for your use case:**
   - Add new conditions to `CONDITION_CONFIGS`
   - Adjust quality gate thresholds
   - Add new task types

### For Production

1. **Use Gunicorn:**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 "app:create_app()"
   ```

2. **Set up nginx reverse proxy with TLS**

3. **Add rate limiting:**
   ```bash
   pip install Flask-Limiter
   ```

4. **Configure logging and monitoring**

5. **Set proper environment variables:**
   ```bash
   FLASK_ENV=production
   SECRET_KEY=your_random_secret
   ```

---

## 📱 iOS App Integration

In your SwiftUI app:

```swift
struct AnalysisResponse: Codable {
    let status: String
    let code: Int
    let message: String
    let data: AnalysisData?
    let errors: APIErrors?
    let meta: APIMeta
}

func analyzeVoice(audioURL: URL) async throws -> AnalysisResponse {
    var request = URLRequest(url: URL(string: "http://localhost:5000/api/v1/analyze")!)
    request.httpMethod = "POST"

    let boundary = UUID().uuidString
    request.setValue("multipart/form-data; boundary=\(boundary)",
                    forHTTPHeaderField: "Content-Type")

    var body = Data()
    let audioData = try Data(contentsOf: audioURL)

    // Add audio file
    body.append("--\(boundary)\r\n")
    body.append("Content-Disposition: form-data; name=\"audio\"; filename=\"recording.m4a\"\r\n")
    body.append("Content-Type: audio/m4a\r\n\r\n")
    body.append(audioData)
    body.append("\r\n")

    // Add metadata
    body.append("--\(boundary)\r\n")
    body.append("Content-Disposition: form-data; name=\"device_id\"\r\n\r\n")
    body.append(UIDevice.current.model)
    body.append("\r\n")

    body.append("--\(boundary)\r\n")
    body.append("Content-Disposition: form-data; name=\"task_type\"\r\n\r\n")
    body.append("sustained_vowel")
    body.append("\r\n")

    body.append("--\(boundary)--\r\n")
    request.httpBody = body

    let (data, _) = try await URLSession.shared.data(for: request)
    return try JSONDecoder().decode(AnalysisResponse.self, from: data)
}
```

---

## 🎉 You're All Set!

The server is now running and ready to analyze voice recordings!

**Key endpoints:**
- `GET /api/v1/health` - Check server status
- `GET /api/v1/tasks` - List available tasks
- `POST /api/v1/analyze` - Analyze voice (production)
- `POST /api/v1/demo/analyze` - Analyze with visualizations (dev/demo)

**Test script:**
```bash
python test_with_samples.py --file your_audio.wav
```

**Documentation:**
- See `README.md` for full documentation
- See `DEMO_ENDPOINT.md` for visualization guide
- See `B2AI_FEATURES.md` for feature details

Happy analyzing! 🎤
