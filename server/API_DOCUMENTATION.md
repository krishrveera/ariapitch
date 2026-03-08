# Voice Health Analysis API Documentation

## Base URL

```
http://localhost:5000/api/v1
```

## Authentication

Currently no authentication required. In production, implement API keys or OAuth.

---

## Endpoints

### 1. POST /analyze

**Full voice analysis pipeline** - Quality gate → Preprocessing → Feature extraction → ML prediction → LLM explanation

#### Request

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | Yes | Audio recording (m4a, caf, wav, mp3, webm, ogg, flac). Max 30MB. |
| `device_id` | string | Yes | Device identifier (e.g., "iPhone16,2", "SM-S928U") |
| `task_type` | string | Yes | One of: `sustained_vowel`, `free_speech`, `reading_passage`, `cough` |
| `silence_duration_sec` | float | No | Leading silence duration in seconds (default: 3.0) |

#### Success Response (200)

```json
{
  "status": "success",
  "code": 200,
  "message": "Analysis complete.",
  "data": {
    "quality": {
      "gate_passed": true,
      "snr_db": 44.2,
      "snr_verdict": "ideal",
      "clipping_fraction": 0.0001,
      "clipping_passed": true,
      "voiced_duration_sec": 4.8,
      "voice_detected": true,
      "agc_detected": false,
      "noise_reduction_applied": false,
      "warnings": []
    },
    "preprocessing": {
      "original_sr": 44100,
      "output_sr": 16000,
      "original_duration_sec": 8.0,
      "output_duration_sec": 5.0,
      "lufs_before": -18.3,
      "lufs_after": -23.0,
      "highpass_applied": true,
      "cmvn_applied": true,
      "noise_reduction_applied": false,
      "device_id": "iPhone16,2"
    },
    "features": {
      "f0_mean_hz": 122.4,
      "f0_std_hz": 2.1,
      "f0_min_hz": 118.2,
      "f0_max_hz": 128.5,
      "jitter_local_percent": 0.45,
      "jitter_rap_percent": 0.32,
      "shimmer_local_percent": 2.1,
      "shimmer_apq3_percent": 1.8,
      "hnr_db": 22.3,
      "cpp_db": 14.8,
      "formant_f1_hz": 720.0,
      "formant_f2_hz": 1240.0,
      "formant_f3_hz": 2800.0
    },
    "predictions": [
      {
        "condition": "parkinsons",
        "condition_name": "Parkinson's Disease",
        "risk_percent": 0.0,
        "confidence": 0.0,
        "severity_tier": "low",
        "features_used": ["jitter_local_percent", "shimmer_local_percent", "hnr_db", "f0_std_hz"],
        "reference_thresholds": {
          "jitter_local_percent": {"normal_max": 1.04, "source": "Teixeira et al., 2013"},
          "shimmer_local_percent": {"normal_max": 3.81, "source": "Teixeira et al., 2013"},
          "hnr_db": {"normal_min": 20.0, "source": "Boersma, 1993"}
        }
      }
    ],
    "explanation": {
      "summary": "Your voice analysis shows healthy vocal characteristics overall.",
      "details": "Your fundamental frequency is stable at 122 Hz...",
      "disclaimer": "This analysis is for informational and screening purposes only...",
      "model_version": "b2ai-voice-v0.1.0"
    }
  },
  "errors": null,
  "meta": {
    "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "timestamp": "2026-03-07T14:30:00Z",
    "processing_time_ms": 3420
  }
}
```

#### Error Response - Quality Gate Failure (422)

```json
{
  "status": "error",
  "code": 422,
  "message": "Recording quality is too low for accurate analysis.",
  "data": null,
  "errors": {
    "type": "quality_gate_failure",
    "failed_checks": [
      {
        "check": "snr",
        "passed": false,
        "value": 22.3,
        "threshold": 30.0,
        "unit": "dB",
        "message": "Background noise is too loud. Your signal-to-noise ratio is 22 dB, but at least 30 dB is needed."
      }
    ],
    "passed_checks": [
      {
        "check": "clipping",
        "passed": true,
        "value": 0.0002,
        "threshold": 0.001,
        "unit": "fraction"
      }
    ],
    "warnings": [],
    "suggestion": "Try recording in a quieter room. Close windows and doors, turn off fans or air conditioning, and hold the phone about 15-20 cm from your mouth."
  },
  "meta": {
    "request_id": "...",
    "timestamp": "...",
    "processing_time_ms": 820
  }
}
```

---

### 2. POST /validate

**Quick quality check** - Runs only the quality gate without full processing.

#### Request

Same as `/analyze`

#### Response

Same envelope format, but `data` contains only `quality` block.

---

### 3. GET /health

**Server health check**

#### Response

```json
{
  "status": "success",
  "code": 200,
  "message": "Server is healthy.",
  "data": {
    "version": "1.0.0",
    "model_loaded": false,
    "model_version": "b2ai-voice-v0.1.0-placeholder",
    "gpu_available": false
  },
  "errors": null,
  "meta": { ... }
}
```

---

### 4. GET /tasks

**Get available vocal tasks**

#### Response

```json
{
  "status": "success",
  "code": 200,
  "message": "Available tasks retrieved.",
  "data": {
    "tasks": [
      {
        "id": "sustained_vowel",
        "display_name": "Sustained Vowel",
        "instruction": "Take a deep breath and say 'ahh' at a comfortable pitch for as long as you can.",
        "min_duration_sec": 5,
        "silence_before_sec": 3,
        "conditions_screened": ["vocal_fold_paralysis", "parkinsons"]
      },
      {
        "id": "free_speech",
        "display_name": "Free Speech",
        "instruction": "Tell us about your day or describe a place you'd like to visit. Speak naturally for about 30 seconds.",
        "min_duration_sec": 20,
        "silence_before_sec": 3,
        "conditions_screened": ["depression", "parkinsons"]
      },
      {
        "id": "reading_passage",
        "display_name": "Reading Passage",
        "instruction": "Read the following passage aloud at your normal speaking pace.",
        "min_duration_sec": 15,
        "silence_before_sec": 3,
        "conditions_screened": ["vocal_fold_paralysis", "parkinsons"]
      },
      {
        "id": "cough",
        "display_name": "Voluntary Cough",
        "instruction": "Cough naturally three times with a short pause between each.",
        "min_duration_sec": 5,
        "silence_before_sec": 3,
        "conditions_screened": ["copd"]
      }
    ]
  },
  "errors": null,
  "meta": { ... }
}
```

---

## Error Types

| Type | HTTP Code | Description |
|------|-----------|-------------|
| `validation_error` | 400 | Invalid request parameters |
| `quality_gate_failure` | 422 | Audio quality checks failed |
| `processing_error` | 500 | Pipeline processing error |
| `server_error` | 500 | Unhandled server exception |

---

## Quality Gate Checks

| Check | Description | Pass Criteria |
|-------|-------------|---------------|
| `silence_region` | Leading quiet period detected | Signal ≥10 dB louder than silence |
| `snr` | Signal-to-noise ratio | ≥30 dB (acceptable), ≥42 dB (ideal) |
| `clipping` | Digital distortion | ≤0.1% of samples clipped |
| `voiced_duration` | Sufficient speech | Duration varies by task type |
| `agc` | Auto gain control detection | Warning only, doesn't fail |

---

## cURL Examples

### Analyze Audio

```bash
curl -X POST http://localhost:5000/api/v1/analyze \
  -F "audio=@recording.m4a" \
  -F "device_id=iPhone16,2" \
  -F "task_type=sustained_vowel" \
  -F "silence_duration_sec=3.0"
```

### Validate Audio

```bash
curl -X POST http://localhost:5000/api/v1/validate \
  -F "audio=@recording.m4a" \
  -F "device_id=iPhone16,2" \
  -F "task_type=sustained_vowel"
```

### Health Check

```bash
curl http://localhost:5000/api/v1/health
```

### Get Tasks

```bash
curl http://localhost:5000/api/v1/tasks
```

---

## Rate Limits (Recommended for Production)

- `/analyze`: 10 requests/minute per IP
- `/validate`: 30 requests/minute per IP
- `/health`: Unlimited
- `/tasks`: Unlimited

---

## Notes

1. **File Size**: Maximum 30 MB per upload
2. **Supported Formats**: m4a, caf, wav, mp3, webm, ogg, flac
3. **Processing Time**: 2-5 seconds for full analysis, <1 second for validation
4. **LLM Explanations**: Requires `ANTHROPIC_API_KEY` environment variable
5. **Model Predictions**: Currently returns 0% risk (placeholder until ML model is integrated)
