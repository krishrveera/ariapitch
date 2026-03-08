# Dummy Prediction Model

The server currently uses **realistic dummy predictions** based on clinical feature thresholds. This allows you to demo the full pipeline while you develop your actual ML model.

## How It Works

The dummy model generates probabilities by:

1. **Extracting feature values** from the audio analysis
2. **Comparing to clinical thresholds** from published literature
3. **Calculating risk scores** based on deviations from normal
4. **Adding realistic randomness** for believability
5. **Generating confidence scores** based on signal clarity

## Condition Logic

### Parkinson's Disease

**Features Used:**
- Jitter (pitch perturbation)
- Shimmer (amplitude perturbation)
- HNR (harmonics-to-noise ratio)
- F0 std deviation (pitch variability)

**Risk Calculation:**
```python
# Jitter: normal < 1.04%
if jitter > 1.04:
    risk += (jitter - 1.04) / 2.0 * 30

# Shimmer: normal < 3.81%
if shimmer > 3.81:
    risk += (shimmer - 3.81) / 5.0 * 25

# HNR: normal > 20 dB
if hnr < 20:
    risk += (20 - hnr) / 10.0 * 20

# F0 std: tremor indicator if > 5 Hz
if f0_std > 5:
    risk += (f0_std - 5) / 10.0 * 15
```

**Example Output:**
- Healthy voice (jitter=0.5%, shimmer=2%, HNR=25): **~5% risk, 85% confidence**
- Abnormal voice (jitter=2%, shimmer=6%, HNR=15): **~45% risk, 75% confidence**

---

### Vocal Fold Paralysis

**Features Used:**
- CPP (cepstral peak prominence)
- HNR (harmonics-to-noise ratio)
- Shimmer (amplitude perturbation)

**Risk Calculation:**
```python
# CPP: concerning if < 8 dB
if cpp < 8:
    risk += (8 - cpp) / 5.0 * 40

# HNR: breathiness indicator if < 18 dB
if hnr < 18:
    risk += (18 - hnr) / 8.0 * 30

# Shimmer: amplitude instability if > 4%
if shimmer > 4:
    risk += (shimmer - 4.0) / 6.0 * 25
```

**Example Output:**
- Healthy voice (CPP=12, HNR=22, shimmer=2): **~3% risk, 90% confidence**
- Abnormal voice (CPP=6, HNR=14, shimmer=7): **~55% risk, 80% confidence**

---

### Depression

**Features Used:**
- F0 mean (average pitch)
- F0 std deviation (pitch variation)
- Jitter

**Risk Calculation:**
```python
# Reduced pitch variability (monotone)
if f0_std < 3:
    risk += (3 - f0_std) / 3.0 * 30

# Lower average pitch
if f0_mean < 100:
    risk += (100 - f0_mean) / 20.0 * 20

# Add randomness (voice-based depression detection is difficult)
risk += random.uniform(-10, 10)
```

**Note:** Depression is harder to detect from voice alone, so predictions have higher variance and lower confidence.

**Example Output:**
- Typical voice (F0=120Hz, std=5Hz): **~8% risk, 70% confidence**
- Monotone voice (F0=95Hz, std=1.5Hz): **~35% risk, 60% confidence**

---

### COPD

**Features Used:**
- CPP (cepstral peak prominence)
- HNR (harmonics-to-noise ratio)
- Shimmer

**Risk Calculation:**
```python
# CPP reduction from airway issues
if cpp < 9:
    risk += (9 - cpp) / 4.0 * 35

# HNR reduction from turbulent airflow
if hnr < 18:
    risk += (18 - hnr) / 8.0 * 30

# Add randomness (cough analysis is complex)
risk += random.uniform(-5, 15)
```

**Example Output:**
- Healthy cough (CPP=11, HNR=20): **~6% risk, 75% confidence**
- Abnormal cough (CPP=6, HNR=12): **~50% risk, 70% confidence**

---

## Confidence Scores

Confidence is based on how clear the signal is:

| Risk Level | Confidence | Reasoning |
|-----------|-----------|-----------|
| **0-10%** (Normal) | 75-95% | Clear normal signal |
| **10-30%** (Low risk) | 60-85% | Slight abnormalities |
| **30-60%** (Moderate) | 50-75% | Ambiguous signal |
| **60-95%** (High risk) | 70-90% | Clear abnormal signal |

---

## Consistency

The dummy model uses **deterministic seeding** based on feature values:

```python
feature_sum = sum(abs(v) for v in feature_values.values())
random.seed(int(feature_sum * 1000) % 10000)
```

This means:
- **Same audio → same prediction** (repeatable)
- **Different audio → different prediction** (realistic variation)
- No stored state needed (stateless)

---

## Replacing with Real Model

When your ML model is ready:

### 1. Update `load_model()` function

```python
def load_model():
    global _model
    import torch
    _model = torch.load("models/your_model.pt")
    _model.eval()
```

### 2. Update `predict()` function

Replace this line:
```python
risk_percent, confidence = _generate_dummy_prediction(condition_id, feature_values, config)
```

With:
```python
# Prepare feature vector
feature_vector = list(feature_values.values())

# Run model inference
with torch.no_grad():
    logits = _model(torch.tensor([feature_vector], dtype=torch.float32))
    probs = torch.softmax(logits, dim=1)
    risk_percent = float(probs[0][1]) * 100  # Probability of positive class

# Calculate confidence (e.g., from prediction entropy or model uncertainty)
confidence = float(torch.max(probs))
```

### 3. Update model version

```python
_model_version = "your-model-v1.0.0"
```

### 4. Test

```bash
python test_with_samples.py --file test.wav
```

The API response format stays exactly the same!

---

## Benefits of Dummy Predictions

✅ **Realistic demos** - Shows what the system will do with a real model
✅ **Frontend development** - iOS team can build UI without waiting for ML model
✅ **API testing** - Test full pipeline end-to-end
✅ **Clinical validation** - Uses real thresholds from literature
✅ **Easy to replace** - Drop in your model when ready

---

## Example API Response

```json
{
  "predictions": [
    {
      "condition": "parkinsons",
      "condition_name": "Parkinson's Disease",
      "risk_percent": 12.3,
      "confidence": 0.82,
      "severity_tier": "low",
      "features_used": ["jitter", "shimmer", "hnr", "f0_std"],
      "feature_values": {
        "jitter": 0.68,
        "shimmer": 2.45,
        "hnr": 21.2,
        "f0_std": 3.8
      },
      "reference_thresholds": {
        "jitter_local_percent": {
          "normal_max": 1.04,
          "source": "Teixeira et al., 2013"
        },
        "shimmer_local_percent": {
          "normal_max": 3.81,
          "source": "Teixeira et al., 2013"
        },
        "hnr_db": {
          "normal_min": 20.0,
          "source": "Boersma, 1993"
        }
      }
    }
  ]
}
```

---

## Clinical References

The thresholds are based on peer-reviewed research:

- **Teixeira et al., 2013** - "Vocal acoustic analysis – jitter, shimmer and HNR parameters"
- **Boersma, 1993** - "Accurate short-term analysis of the fundamental frequency"
- **Heman-Ackah et al., 2003** - "Cepstral peak prominence: a more reliable measure of dysphonia"

These are the same thresholds used in clinical voice labs worldwide.

---

## Testing Different Scenarios

### Normal Voice
```bash
# Record with good technique
# Expected: Low risk (0-15%), high confidence (75-95%)
```

### Elevated Jitter/Shimmer
```bash
# Record with vocal strain or poor mic
# Expected: Moderate risk (15-40%), medium confidence (60-75%)
```

### Poor Quality (should fail quality gate)
```bash
# Record with background noise
# Expected: Quality gate failure before prediction
```

The dummy model provides a **complete, believable demo** of the final system!
