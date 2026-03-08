# B2AI-Voice Feature Extraction

The server now uses **SenseLab** for comprehensive Bridge2AI-Voice aligned feature extraction.

## What is B2AI-Voice?

Bridge2AI-Voice is a standardized framework for voice biomarker research developed by the NIH Bridge to Artificial Intelligence (Bridge2AI) program. It ensures consistent, reproducible voice analysis across different research sites and applications.

## Feature Extraction Methods

### Primary: SenseLab (Recommended)

When SenseLab is installed, the server extracts comprehensive features including:

#### 1. **Praat Features**
- **F0 (Pitch)**: Fundamental frequency statistics (mean, std, min, max)
- **Jitter**: Pitch perturbation measurements
- **Shimmer**: Amplitude perturbation measurements
- **HNR**: Harmonics-to-noise ratio
- **CPP**: Cepstral peak prominence
- **Formants**: F1, F2, F3 resonance frequencies

#### 2. **MFCCs** (Mel-Frequency Cepstral Coefficients)
- 13-40 coefficients per time frame
- Captures spectral envelope
- Standard for speech/speaker recognition
- Returned as 2D array: `[coefficients × time]`

#### 3. **Spectrograms**
- **Linear Spectrogram**: Full frequency resolution
- **Mel Spectrogram**: Perceptually-weighted frequencies
- Useful for visual inspection and deep learning models

#### 4. **PPGs** (Phonetic Posteriorgrams)
- Phoneme probability distributions over time
- Extracted using pre-trained speech recognition models
- Shape: `[phoneme_classes × time]`

#### 5. **OpenSMILE Features**
- Industry-standard acoustic feature set
- 6000+ features including:
  - Energy, loudness, spectral features
  - Voice quality measures
  - Temporal dynamics
- Returned as Pandas DataFrame

#### 6. **Articulatory Features** (if available)
- Vocal tract kinematics
- Tongue/lip movement patterns

### Fallback: Praat/Parselmouth

If SenseLab is not available, the server falls back to basic Praat feature extraction using Parselmouth. This provides essential voice features but lacks the comprehensive B2AI suite.

## Feature Structure

### B2AI Features (Nested)

SenseLab returns features in a nested dictionary structure:

```python
{
    "praat": {
        "pitch": {
            "f0_mean_hz": 120.5,
            "f0_std_hz": 2.3,
            ...
        },
        "jitter": {...},
        "shimmer": {...},
        "harmonicity": {...}
    },
    "mfcc": torch.Tensor([13, 500]),  # 13 coefficients, 500 time frames
    "mel_spectrogram": torch.Tensor([128, 500]),
    "ppg": torch.Tensor([41, 500]),  # 41 phoneme classes
    "opensmile": pd.DataFrame(...),  # 6000+ features
    ...
}
```

### Flattened for API Response

The server automatically flattens this structure for JSON serialization:

```json
{
  "praat.pitch.f0_mean_hz": 120.5,
  "praat.pitch.f0_std_hz": 2.3,
  "praat.jitter.local_percent": 0.45,
  "mfcc.shape": [13, 500],
  "mfcc.mean": -12.3,
  "mfcc.coef_0": 15.2,
  "mfcc.coef_1": -3.4,
  ...
  "opensmile.F0semitoneFrom27.5Hz_sma3nz_amean.mean": 18.5,
  ...
}
```

**Benefits of flattening:**
- JSON-serializable (no tensors/DataFrames in API response)
- Direct access to feature values
- Compatible with ML models expecting flat feature vectors
- Reduces response size (statistics instead of full arrays)

## Installation

### SenseLab (Full B2AI Features)

```bash
pip install senselab>=0.20.0
pip install torch torchaudio transformers opensmile
```

**Note:** SenseLab requires PyTorch, which has a large download (~2GB). For faster development, you can use the Praat fallback.

### Praat Only (Fallback)

```bash
pip install praat-parselmouth
```

## Feature Selection for ML Models

The prediction service automatically maps condition-specific features from the B2AI feature dictionary:

```python
CONDITION_CONFIGS = {
    "parkinsons": {
        "features": {
            "jitter": ["praat.jitter.local_percent", "praat.jitter.local.mean"],
            "shimmer": ["praat.shimmer.local_percent", "praat.shimmer.local.mean"],
            "hnr": ["praat.hnr.mean_db", "praat.harmonicity.mean"],
            "f0_std": ["praat.pitch.f0_std_hz", "praat.pitch.std"],
        }
    }
}
```

**Multiple paths** allow the system to work with:
1. B2AI/SenseLab feature names (preferred)
2. Praat fallback names
3. Different SenseLab versions

## Visualization (for debugging)

The B2AI features can be visualized using the included helper function from the pipeline:

```python
from services.feature_extraction import extract_features

features = extract_features("audio.wav", "sustained_vowel")

# In a Jupyter notebook:
visualize_bridge2ai_features(features)
```

This will show:
- 2D heatmaps for spectrograms, MFCCs, PPGs
- 1D time-series plots for pitch, loudness
- Tables for OpenSMILE features
- Text for transcriptions

## API Response Example

With B2AI features:

```json
{
  "data": {
    "features": {
      "praat.pitch.f0_mean_hz": 122.4,
      "praat.pitch.f0_std_hz": 2.1,
      "praat.jitter.local_percent": 0.45,
      "praat.shimmer.local_percent": 2.1,
      "praat.hnr.mean_db": 22.3,
      "praat.cpp.mean_db": 14.8,
      "praat.formants.f1_hz": 720.0,
      "praat.formants.f2_hz": 1240.0,
      "mfcc.shape": [13, 312],
      "mfcc.mean": -15.2,
      "mfcc.std": 12.8,
      "mfcc.coef_0": 18.3,
      "mfcc.coef_1": -3.2,
      "mel_spectrogram.shape": [128, 312],
      "mel_spectrogram.mean": -42.1,
      "ppg.shape": [41, 312],
      "opensmile.F0semitoneFrom27.5Hz_sma3nz_amean.mean": 18.5,
      "_extraction_method": "senselab_b2ai"
    }
  }
}
```

## Performance Considerations

**SenseLab Extraction Time:**
- Praat features: ~0.5s
- MFCCs: ~0.2s
- Spectrograms: ~0.3s
- PPGs (with transformers): ~2-5s
- OpenSMILE: ~1-2s
- **Total: ~4-8 seconds** for full B2AI extraction

**Praat Fallback Time:**
- ~0.5-1s total

**Recommendation:**
- Development: Use Praat fallback for speed
- Production: Use full B2AI for comprehensive analysis
- Option: Make PPG/OpenSMILE extraction optional via config

## Clinical Validity

The B2AI-Voice pipeline follows evidence-based clinical standards:

| Feature | Clinical Use | Normal Range | Source |
|---------|-------------|--------------|---------|
| Jitter | Voice disorders, Parkinson's | < 1.04% | Teixeira et al., 2013 |
| Shimmer | Voice disorders | < 3.81% | Teixeira et al., 2013 |
| HNR | Voice quality | > 20 dB | Boersma, 1993 |
| CPP | Dysphonia severity | > 8 dB | Heman-Ackah et al., 2003 |

## Extending Features

To add new B2AI features to your model:

1. Extract raw B2AI features (already done)
2. Add feature paths to `CONDITION_CONFIGS` in `prediction.py`
3. Train your ML model on the new features
4. The API will automatically include them in responses

Example:

```python
"your_condition": {
    "features": {
        "mfcc_mean": ["mfcc.mean"],
        "spectral_entropy": ["opensmile.spectralEntropy_sma3.mean"],
        "voicing_prob": ["ppg.voicing_probability.mean"]
    }
}
```

## References

- Bridge2AI-Voice: https://github.com/sensein/b2aiprep
- SenseLab: https://github.com/sensein/senselab
- Praat: https://www.fon.hum.uva.nl/praat/
- OpenSMILE: https://www.audeering.com/research/opensmile/
