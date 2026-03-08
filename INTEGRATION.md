# AriaPitch ML Integration Guide

This document describes how the frontend website integrates with the Flask ML backend for real-time voice analysis.

## Architecture Overview

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   React Web     │  HTTP   │   Flask Server   │         │  ML Pipeline    │
│   Application   ├────────>│   (Port 5000)    ├────────>│  (classifier.py)│
│                 │         │                  │         │                 │
└─────────────────┘         └──────────────────┘         └─────────────────┘
       │                            │                            │
       │                            │                            │
   Audio Blob              Feature Extraction          Voting Ensemble
   (WebM/WAV)              Quality Gate                (SVM+RF+XGBoost)
                          Preprocessing
```

## Setup Instructions

### 1. Backend Server

Start the Flask server:

```bash
cd server
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The server will run on `http://localhost:5000`

### 2. Frontend Application

Configure the API endpoint and start the dev server:

```bash
cd website
npm install

# Create .env file if it doesn't exist
echo "VITE_API_URL=http://localhost:5000/api/v1" > .env

npm run dev
```

The website will run on `http://localhost:5173`

## API Integration

### API Service ([website/src/app/services/api.ts](website/src/app/services/api.ts))

The frontend uses a TypeScript API service that provides:

- **`analyzeAudio(blob, options)`** - Full ML pipeline analysis
- **`validateAudio(blob, options)`** - Quick quality check only
- **`checkHealth()`** - Server health check
- **`extractBiomarkers(features)`** - Convert ML features to UI metrics
- **`generateMessage(predictions, explanation)`** - Create user-friendly messages

### Recording Flow Integration

The [RecordingFlow.tsx](website/src/app/components/pages/RecordingFlow.tsx) component:

1. **Records audio** using MediaRecorder API
2. **Saves audio blob** when user clicks "Stop Recording"
3. **Sends to backend** via `/api/v1/analyze` endpoint
4. **Receives analysis**:
   - Quality gate results
   - Extracted features (pitch, jitter, shimmer, etc.)
   - ML predictions (risk level, confidence)
   - LLM-generated explanation
5. **Updates UI** with real biomarker data
6. **Stores results** in app context

### Data Flow

```typescript
// 1. User stops recording
handleStopRecording() → setFlowState("ANALYZING")

// 2. Effect triggers API call
useEffect(() => {
  if (flowState === "ANALYZING" && audioBlob) {
    const result = await analyzeAudio(audioBlob, {
      deviceId: 'web_app',
      taskType: 'reading_passage',
      silenceDuration: 0.5
    });

    // 3. Extract biomarkers from ML features
    const biomarkers = extractBiomarkers(result.data.features);
    // { pitch: 215, shimmer: 3.2, jitter: 1.1, ... }

    // 4. Generate user message
    const { message, isAnomaly } = generateMessage(
      result.data.predictions,
      result.data.explanation
    );
  }
}, [flowState, audioBlob]);

// 5. Display results
finishRecording() → Navigate to home with updated data
```

## ML Pipeline Details

### Quality Gate
The backend first validates audio quality:
- Silence detection
- SNR (Signal-to-Noise Ratio)
- Clipping detection
- Duration check

**Frontend Handling:**
- If quality gate fails → Show error message with suggestions
- If passed → Continue to feature extraction

### Feature Extraction
Uses Bridge2AI voice features:
- **Pitch** (fundamental frequency)
- **Jitter** (frequency perturbation)
- **Shimmer** (amplitude perturbation)
- **HNR** (Harmonic-to-Noise Ratio)
- **Spectral** features
- **Formants**

### ML Classification
Voting Ensemble classifier ([server/services/classifier.py](server/services/classifier.py)):
- **SVM** (C=10, RBF kernel)
- **Random Forest** (200 trees)
- **XGBoost** (200 estimators, depth=8)
- **SMOTE** oversampling for class balance
- **Soft voting** for final prediction

### Result Processing

Backend returns:
```json
{
  "quality": {
    "score": 0.95,
    "passed": true
  },
  "features": {
    "pitch_mean": 215.3,
    "jitter_local": 0.011,
    "shimmer_local": 0.032,
    "hnr_mean": 18.5
  },
  "predictions": {
    "risk_level": "low",
    "confidence": 0.87,
    "probability": 0.13
  },
  "explanation": {
    "summary": "Your voice shows healthy characteristics...",
    "recommendations": ["Stay hydrated", "Regular voice rest"]
  }
}
```

Frontend transforms this to:
```typescript
{
  pitch: 215,      // Hz
  shimmer: 3.2,    // %
  jitter: 1.1,     // %
  message: "Your voice shows healthy characteristics. Stay hydrated.",
  isAnomaly: false
}
```

## Bridge2AI Study Integration

When users **opt in** to the Bridge2AI study:

1. **Full audio recordings** are saved with complete analysis
2. **Audio Library** becomes accessible (purple button in header)
3. **Detailed analysis views** with:
   - Temporal dynamics graphs
   - Frequency spectrum
   - Interactive heatmaps
   - Comparison to previous recordings

When users **opt out**:
- Only basic metrics saved (no audio data)
- No access to audio library
- Summary insights only

### Data Storage Structure

```typescript
audioRecordings: [
  {
    id: "rec_1234567890_abc123",
    timestamp: 1709856000000,
    date: "Mar 8, 2026",
    pitch: 215,
    shimmer: 3.2,
    jitter: 1.1,
    spectralCentroid: 1523,
    harmonicRatio: 18.5,
    formants: [720, 1215, 2543],
    message: "Voice analysis message...",
    isAnomaly: false,
    duration: 45  // seconds
  }
]
```

## Error Handling

### Network Errors
```typescript
catch (error) {
  return {
    success: false,
    error: {
      type: 'network_error',
      message: 'Please check your connection',
      suggestion: 'Ensure the backend server is running'
    }
  };
}
```

### Quality Gate Failures
```typescript
if (result.error?.type === 'quality_gate_failure') {
  setAnalysisError(result.error.message);
  setIsSuccess(false);
  // Show user-friendly error with suggestions
}
```

### Fallback Behavior
If the backend is unavailable, the app falls back to:
- Random but realistic dummy data for demo purposes
- Clear indication that connection failed
- Option to retry

## Environment Configuration

### Development
`.env`:
```bash
VITE_API_URL=http://localhost:5000/api/v1
```

### Production
Update to your deployed backend URL:
```bash
VITE_API_URL=https://your-api-domain.com/api/v1
```

## Testing the Integration

### 1. Health Check
```bash
curl http://localhost:5000/api/v1/health
```

Should return:
```json
{
  "success": true,
  "data": {
    "model_loaded": true,
    "model_type": "VotingClassifier",
    "server_version": "1.0.0"
  }
}
```

### 2. Audio Analysis
Test with a sample audio file:
```bash
curl -X POST http://localhost:5000/api/v1/analyze \
  -F "audio=@sample.wav" \
  -F "task_type=reading_passage" \
  -F "device_id=test"
```

### 3. Frontend Integration
1. Navigate to recording page
2. Click "I'm Ready"
3. Record audio (read the prompt)
4. Click "Stop Recording"
5. Watch real-time analysis

## Performance Considerations

- **Audio blob size**: WebM typically 10-50KB per second
- **API latency**: 2-5 seconds for full analysis
- **Feature extraction**: ~1-2 seconds
- **ML prediction**: ~500ms
- **Quality gate**: ~500ms

## Security Notes

- Audio is **never stored** on the server (temporary files cleaned up)
- Only opted-in users have audio data persisted (client-side only)
- All communication over HTTP (use HTTPS in production)
- No authentication required for MVP (add for production)

## Troubleshooting

### "Network Error"
- Ensure backend server is running on port 5000
- Check CORS configuration in `app.py`
- Verify firewall allows local connections

### "Quality Gate Failure"
- Record in a quiet environment
- Speak clearly into microphone
- Ensure 3+ seconds of audio
- Check microphone permissions

### "Analysis Failed"
- Check server logs: `tail -f server.log`
- Verify ML model is loaded (health endpoint)
- Ensure all dependencies installed
- Check feature extraction pipeline

## Future Enhancements

1. **Real-time streaming** - Analyze audio during recording
2. **Batch processing** - Multiple recordings at once
3. **Model versioning** - Track ML model updates
4. **Confidence scores** - Per-biomarker confidence intervals
5. **Trend analysis** - Long-term voice health tracking

---

Built with ❤️ for Cornell Health Hack 2026
