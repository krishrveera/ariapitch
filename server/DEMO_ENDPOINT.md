# Demo Endpoint Documentation

## `/api/v1/demo/analyze` - Full Analysis with Visualizations

This endpoint extends `/api/v1/analyze` with comprehensive visualizations and raw feature data, perfect for debugging, demonstrations, and understanding the analysis pipeline.

---

## What's Different from `/analyze`?

| Feature | `/analyze` | `/demo/analyze` |
|---------|-----------|-----------------|
| Quality gate | ✅ | ✅ |
| Preprocessing | ✅ | ✅ |
| Feature extraction | ✅ Flattened only | ✅ Flattened + Raw |
| ML predictions | ✅ | ✅ |
| LLM explanations | ✅ | ✅ |
| **Visualizations** | ❌ | ✅ **Base64 PNGs** |
| **Raw B2AI features** | ❌ | ✅ **Full structure** |
| Response size | ~10-50 KB | ~500 KB - 2 MB |
| Processing time | 4-8s | 6-12s |

---

## Request

Same as `/api/v1/analyze`:

```bash
curl -X POST http://localhost:5000/api/v1/demo/analyze \
  -F "audio=@recording.wav" \
  -F "device_id=iPhone16,2" \
  -F "task_type=sustained_vowel" \
  -F "silence_duration_sec=3.0"
```

---

## Response Structure

```json
{
  "status": "success",
  "code": 200,
  "message": "Demo analysis complete with visualizations.",
  "data": {
    "quality": { ... },         // Same as /analyze
    "preprocessing": { ... },   // Same as /analyze
    "features": { ... },        // Flattened features (same as /analyze)
    "features_raw": {           // NEW: Raw B2AI feature structure
      "praat": {
        "pitch": {"type": "dict", ...},
        "jitter": {"type": "dict", ...}
      },
      "mfcc": {
        "type": "tensor",
        "shape": [13, 312],
        "mean": -15.2,
        "std": 12.8
      },
      "mel_spectrogram": {...},
      "ppg": {...},
      "opensmile": {...}
    },
    "predictions": [ ... ],     // Same as /analyze
    "explanation": { ... },     // Same as /analyze
    "visualizations": {         // NEW: Base64-encoded PNG images
      "original_waveform": "iVBORw0KGgoAAAANSUhE...",
      "original_spectrogram": "iVBORw0KGgoAAAANSUhE...",
      "quality_gate_summary": "iVBORw0KGgoAAAANSUhE...",
      "snr_analysis": "iVBORw0KGgoAAAANSUhE...",
      "mfcc": "iVBORw0KGgoAAAANSUhE...",
      "mel_spectrogram": "iVBORw0KGgoAAAANSUhE...",
      "ppg": "iVBORw0KGgoAAAANSUhE...",
      ...
    }
  },
  "errors": null,
  "meta": {
    "request_id": "...",
    "timestamp": "...",
    "processing_time_ms": 8450
  }
}
```

---

## Visualizations Included

### 1. **Quality Gate Visualizations**

#### `original_waveform`
- Time-domain waveform of original audio
- Shows amplitude over time
- Useful for spotting clipping, silence issues

#### `original_spectrogram`
- Frequency-domain representation
- Mel-scaled, color-coded by intensity
- Shows spectral content over time

#### `quality_gate_summary`
- Bar chart comparing measured values to thresholds
- Color-coded: green = passed, red = failed
- Shows SNR, clipping, voiced duration, etc.

#### `snr_analysis`
- Two-panel plot:
  - Top: Waveform with silence region marked
  - Bottom: RMS energy comparison (noise vs signal)
- Helps understand noise floor vs speech level

### 2. **B2AI Feature Visualizations**

#### `mfcc`
- 2D heatmap: 13 MFCC coefficients × time
- Color: Cool/warm colormap
- Shows spectral envelope evolution

#### `mel_spectrogram`
- 2D heatmap: Mel frequency bins × time
- Color: Magma colormap
- Perceptually-weighted frequency representation

#### `ppg` (Phonetic Posteriorgram)
- 2D heatmap: 41 phoneme classes × time
- Shows phoneme probability distributions
- Useful for understanding articulatory patterns

### 3. **Preprocessing Comparisons** (future enhancement)
- Before/after waveforms
- Before/after spectrograms
- Shows effect of each preprocessing step

---

## Using Visualizations

### In Python

```python
import requests
import base64
from PIL import Image
import io

# Make request
response = requests.post(
    "http://localhost:5000/api/v1/demo/analyze",
    files={"audio": open("recording.wav", "rb")},
    data={
        "device_id": "test",
        "task_type": "sustained_vowel"
    }
)

data = response.json()

# Save visualizations
for viz_name, viz_data in data["data"]["visualizations"].items():
    # Decode base64
    img_bytes = base64.b64decode(viz_data)

    # Open with PIL
    img = Image.open(io.BytesIO(img_bytes))

    # Save to file
    img.save(f"{viz_name}.png")

print(f"Saved {len(data['data']['visualizations'])} visualizations!")
```

### In JavaScript/Web

```javascript
const formData = new FormData();
formData.append('audio', audioFile);
formData.append('device_id', 'test');
formData.append('task_type', 'sustained_vowel');

fetch('http://localhost:5000/api/v1/demo/analyze', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => {
  // Display visualizations as <img> tags
  const visualizations = data.data.visualizations;

  for (const [name, base64Data] of Object.entries(visualizations)) {
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${base64Data}`;
    img.alt = name;
    document.body.appendChild(img);
  }
});
```

### In HTML (Direct Embedding)

```html
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhE..."
     alt="Quality Gate Summary">
```

---

## Raw B2AI Features

The `features_raw` field contains a summary of the raw B2AI feature structure (not flattened):

```json
{
  "praat": {
    "pitch": {
      "f0_mean_hz": 122.4,
      "f0_std_hz": 2.1
    },
    "jitter": {...},
    "shimmer": {...}
  },
  "mfcc": {
    "type": "tensor",
    "shape": [13, 312],
    "dtype": "float32",
    "mean": -15.234,
    "std": 12.851
  },
  "mel_spectrogram": {
    "type": "tensor",
    "shape": [128, 312],
    ...
  },
  "opensmile": {
    "type": "dataframe",
    "shape": [1, 6373],
    "columns": ["F0semitoneFrom27.5Hz_sma3nz_amean", ...]
  }
}
```

**Note:** To save bandwidth, the actual tensor/array data is not included. Only metadata (shape, type, statistics) is returned.

---

## Use Cases

### 1. **Debugging Pipeline Issues**
- Visualize waveforms to spot clipping, noise
- Check spectrograms for frequency content
- Verify quality gate thresholds

### 2. **Research & Analysis**
- Inspect MFCCs, PPGs for feature engineering
- Compare different recording conditions
- Validate preprocessing steps

### 3. **Demos & Presentations**
- Generate publication-ready plots
- Show clients how analysis works
- Explain quality requirements

### 4. **Model Development**
- Visualize features used by ML models
- Understand feature distributions
- Debug poor predictions

---

## Performance Considerations

**Response Size:**
- `/analyze`: ~10-50 KB (JSON only)
- `/demo/analyze`: ~500 KB - 2 MB (with 10-20 PNGs)

**Processing Time:**
- Visualization generation adds ~2-4 seconds
- Consider caching if calling frequently

**Recommendations:**
- Use `/analyze` for production
- Use `/demo/analyze` for development/debugging
- Don't call `/demo/analyze` from mobile apps (too large)
- Consider returning only specific visualizations (future enhancement)

---

## Testing

### Using the Test Script

```bash
cd server
python test_with_samples.py --file path/to/audio.wav
```

This will:
1. Test `/health` endpoint
2. Test `/tasks` endpoint
3. Test `/analyze` endpoint
4. Test `/demo/analyze` endpoint
5. Save all visualizations to `demo_output_[filename]/`

### Manual Testing

```bash
# Health check
curl http://localhost:5000/api/v1/health

# Demo analysis
curl -X POST http://localhost:5000/api/v1/demo/analyze \
  -F "audio=@test.wav" \
  -F "device_id=test" \
  -F "task_type=sustained_vowel" \
  | jq '.data.visualizations | keys'
```

---

## Future Enhancements

- [ ] Selective visualization generation (query param: `?viz=waveform,spectrogram`)
- [ ] Interactive HTML report instead of PNGs
- [ ] Side-by-side preprocessing comparisons
- [ ] 3D spectrograms for longer recordings
- [ ] Real-time WebSocket streaming for live analysis
- [ ] Export to PDF report

---

## Error Handling

Even if quality gate fails, visualizations are still generated:

```json
{
  "status": "error",
  "code": 422,
  "message": "Recording quality is too low...",
  "errors": {
    "type": "quality_gate_failure",
    "failed_checks": [...]
  },
  "data": null,
  "meta": {...}
}
```

**Note:** In future, we could add visualizations to error responses as well to help users understand what went wrong.

---

## Security Notes

- Visualizations are generated server-side (matplotlib)
- No user code execution
- Images are rendered in memory (not saved to disk)
- Base64 encoding prevents file path injection
- Consider rate limiting (visualization generation is CPU-intensive)
