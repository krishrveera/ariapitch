# Voice Health Analysis API Server

Flask-based REST API for voice health screening using the Bridge2AI-Voice aligned preprocessing pipeline.

## 🚀 Quick Start

**Just want to get started?** See **[QUICKSTART.md](QUICKSTART.md)** for a 5-minute setup guide!

```bash
cd server
./start_server.sh
```

## Architecture

```
server/
├── app.py                      # Flask application factory
├── config.py                   # Environment configuration
├── requirements.txt            # Python dependencies
│
├── api/
│   ├── routes.py               # API endpoints
│   └── response.py             # Standardized response envelope
│
├── services/
│   ├── quality_gate.py         # Audio quality validation
│   ├── preprocessing.py        # B2AI-aligned preprocessing
│   ├── feature_extraction.py   # Praat/Parselmouth features
│   ├── prediction.py           # ML model inference
│   ├── explanation.py          # LLM-generated explanations
│   └── task_definitions.py     # Vocal task configurations
│
└── utils/
    └── audio_io.py             # File handling utilities
```

## Setup

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

### 3. Set Environment Variables

Create a `.env` file:

```bash
# Use Google Gemini (recommended - free tier available)
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here

# Or use Claude (alternative)
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

FLASK_ENV=development
```

**Get a free Gemini API key:** https://makersuite.google.com/app/apikey

### 4. Run Development Server

```bash
python app.py
```

Server will start on `http://0.0.0.0:5000`

### 5. Run Production Server

```bash
gunicorn -w 4 -b 0.0.0.0:5000 "app:create_app()"
```

## API Endpoints

### POST /api/v1/analyze

Full voice analysis pipeline.

**Request:**
- `audio` (file): Audio recording (m4a, caf, wav, mp3)
- `device_id` (string): Device model identifier
- `task_type` (string): One of: `sustained_vowel`, `free_speech`, `reading_passage`, `cough`
- `silence_duration_sec` (float, optional): Leading silence duration (default: 3.0)

**Response:**
```json
{
  "status": "success",
  "code": 200,
  "message": "Analysis complete.",
  "data": {
    "quality": { ... },
    "preprocessing": { ... },
    "features": { ... },
    "predictions": [ ... ],
    "explanation": { ... }
  },
  "errors": null,
  "meta": { ... }
}
```

### POST /api/v1/validate

Quick quality check without full analysis.

### GET /api/v1/health

Server health status.

### GET /api/v1/tasks

Available vocal tasks and their requirements.

## Response Format

All endpoints return a standardized envelope:

```json
{
  "status": "success | error",
  "code": 200,
  "message": "Human-readable message",
  "data": { ... },
  "errors": null,
  "meta": {
    "request_id": "uuid",
    "timestamp": "ISO 8601",
    "processing_time_ms": 1234
  }
}
```

## Error Types

- `validation_error` (400): Invalid request parameters
- `quality_gate_failure` (422): Audio quality too low
- `processing_error` (500): Pipeline failure
- `server_error` (500): Unhandled exception

## Quality Gate Checks

1. **Silence Region**: Leading quiet period for noise estimation
2. **SNR**: Signal-to-noise ratio ≥30 dB
3. **Clipping**: Digital distortion ≤0.1%
4. **Voiced Duration**: Sufficient speech detected
5. **AGC Detection**: Automatic gain control warning

## Preprocessing Pipeline

1. Resample to 16 kHz with Butterworth anti-aliasing
2. High-pass filter at 80 Hz (mic compensation)
3. CMVN spectral normalization
4. LUFS loudness normalization (-23 LUFS)
5. Conditional noise reduction (if SNR 30-42 dB)
6. Trim leading silence
7. Export as 16-bit PCM WAV

## Feature Extraction

Uses **SenseLab** for comprehensive B2AI-Voice aligned features:
- **Praat Features**: F0, Jitter, Shimmer, HNR, CPP, Formants
- **MFCCs**: Mel-frequency cepstral coefficients (13-40 per frame)
- **Spectrograms**: Linear and Mel spectrograms
- **PPGs**: Phonetic posteriorgrams (phoneme probabilities)
- **OpenSMILE**: 6000+ industry-standard acoustic features
- **Articulatory**: Vocal tract kinematics (when available)

Falls back to basic Praat extraction if SenseLab is unavailable.

See [B2AI_FEATURES.md](B2AI_FEATURES.md) for detailed documentation.

## ML Prediction

Placeholder for model integration. Currently returns 0% risk for all conditions.

To integrate your model:
1. Place model checkpoint in `models/`
2. Update `services/prediction.py` `load_model()` function
3. Replace placeholder prediction logic with model inference

## LLM Explanations

Generates user-friendly explanations using either:
- **Google Gemini** (recommended, free tier available)
- **Anthropic Claude** (optional alternative)

Set `LLM_PROVIDER=gemini` and `GOOGLE_API_KEY` in your `.env` file.

The service automatically falls back between providers if one fails.

## Testing

```bash
# Health check
curl http://localhost:5000/api/v1/health

# Get tasks
curl http://localhost:5000/api/v1/tasks

# Analyze audio
curl -X POST http://localhost:5000/api/v1/analyze \
  -F "audio=@test.m4a" \
  -F "device_id=iPhone16,2" \
  -F "task_type=sustained_vowel" \
  -F "silence_duration_sec=3.0"
```

## Deployment Notes

- Use Gunicorn with multiple workers for production
- Set up periodic temp file cleanup (cron job)
- Deploy behind nginx with TLS
- Add rate limiting (Flask-Limiter)
- Monitor with structured logging

## License

MIT
