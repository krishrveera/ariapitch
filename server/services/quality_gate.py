"""
Quality Gate Service
Wraps the pipeline's quality gate functions and returns structured check objects.
"""
import sys
import os
import numpy as np
import librosa

# Add parent directory to path to import pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Human-readable messages for each check failure
FAILURE_MESSAGES = {
    "snr": (
        "Background noise is too loud. Your recording has a signal-to-noise "
        "ratio of {value:.0f} dB, but at least {threshold:.0f} dB is needed "
        "for accurate voice analysis."
    ),
    "clipping": (
        "The audio is distorted because the volume was too high. "
        "{value_pct:.1f}% of the recording hit the maximum level. "
        "Try holding the phone slightly further from your mouth."
    ),
    "voiced_duration": (
        "Not enough speech was detected. We found {value:.1f} seconds of "
        "voice, but at least {threshold:.0f} seconds is needed. "
        "Try speaking for longer."
    ),
    "silence_region": (
        "We couldn't detect a quiet period at the start of the recording. "
        "Make sure to stay silent for the first few seconds before speaking."
    ),
}


class PipelineConfig:
    """Configuration for the quality gate pipeline."""
    def __init__(self, device_id="unknown", silence_duration_sec=3.0):
        self.target_sr = 16000
        self.bit_depth = 16
        self.snr_ideal = 42.0
        self.snr_acceptable = 30.0
        self.snr_reject = 30.0
        self.silence_duration_sec = silence_duration_sec
        self.min_voiced_duration_sec = 3.0
        self.clipping_margin = 0.001
        self.max_clipping_fraction = 0.001
        self.target_lufs = -23.0
        self.noise_reduce_strength = 0.7
        self.butter_order = 8
        self.vad_energy_multiplier = 3.0
        self.vad_frame_length_ms = 25.0
        self.vad_hop_length_ms = 10.0
        self.device_id = device_id
        self.apply_cmvn = True
        self.highpass_cutoff_hz = 80.0
        self.highpass_order = 4
        self.agc_coefficient_of_variation_threshold = 0.15


def gate1_check_silence_region(audio: np.ndarray, sr: int, config: PipelineConfig) -> tuple:
    """Verify that the recording begins with a silence period for noise estimation."""
    silence_samples = int(config.silence_duration_sec * sr)

    if len(audio) < silence_samples:
        return False, 0

    silence_region = audio[:silence_samples]
    signal_region = audio[silence_samples:]

    if len(signal_region) == 0:
        return False, 0

    silence_rms = np.sqrt(np.mean(silence_region ** 2)) + 1e-10
    signal_rms = np.sqrt(np.mean(signal_region ** 2)) + 1e-10

    ratio_db = 20 * np.log10(signal_rms / silence_rms)
    has_silence = ratio_db > 10

    return has_silence, silence_samples


def gate2_check_snr(audio: np.ndarray, sr: int, silence_end_sample: int, config: PipelineConfig) -> tuple:
    """Compute Signal-to-Noise Ratio using the silence region as noise estimate."""
    noise_region = audio[:silence_end_sample]
    signal_region = audio[silence_end_sample:]

    noise_power = np.mean(noise_region ** 2) + 1e-20
    signal_power = np.mean(signal_region ** 2) + 1e-20

    snr_db = 10 * np.log10(signal_power / noise_power)

    if snr_db >= config.snr_ideal:
        verdict = "ideal"
    elif snr_db >= config.snr_acceptable:
        verdict = "acceptable"
    else:
        verdict = "reject"

    return verdict, snr_db


def gate3_check_clipping(audio: np.ndarray, config: PipelineConfig) -> tuple:
    """Detect digital clipping."""
    clipped = np.abs(audio) >= (1.0 - config.clipping_margin)
    clipping_fraction = np.mean(clipped)
    passed = clipping_fraction <= config.max_clipping_fraction
    return passed, clipping_fraction


def gate4_check_voice_activity(audio: np.ndarray, sr: int, silence_end_sample: int, config: PipelineConfig) -> tuple:
    """Energy-based Voice Activity Detection."""
    frame_length = int(config.vad_frame_length_ms * sr / 1000)
    hop_length = int(config.vad_hop_length_ms * sr / 1000)

    noise_region = audio[:silence_end_sample]
    noise_rms = np.sqrt(np.mean(noise_region ** 2)) + 1e-10
    energy_threshold = noise_rms * config.vad_energy_multiplier

    signal_region = audio[silence_end_sample:]
    n_frames = max(1, (len(signal_region) - frame_length) // hop_length + 1)

    voiced_mask = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        if end > len(signal_region):
            break
        frame_rms = np.sqrt(np.mean(signal_region[start:end] ** 2))
        voiced_mask[i] = frame_rms > energy_threshold

    voiced_duration = np.sum(voiced_mask) * (hop_length / sr)
    has_voice = voiced_duration >= config.min_voiced_duration_sec

    return has_voice, voiced_duration, np.any(voiced_mask)


def gate5_check_agc(audio: np.ndarray, sr: int, silence_end_sample: int, config: PipelineConfig) -> bool:
    """Detect whether Automatic Gain Control (AGC) is active."""
    signal_region = audio[silence_end_sample:]
    if len(signal_region) < sr:
        return True  # Not enough data, skip check

    frame_length = int(config.vad_frame_length_ms * sr / 1000)
    hop_length = int(config.vad_hop_length_ms * sr / 1000)
    n_frames = (len(signal_region) - frame_length) // hop_length + 1

    frame_rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame_rms[i] = np.sqrt(np.mean(signal_region[start:end] ** 2))

    noise_rms = np.sqrt(np.mean(audio[:silence_end_sample] ** 2)) + 1e-10
    voiced_rms = frame_rms[frame_rms > noise_rms * config.vad_energy_multiplier]

    if len(voiced_rms) < 10:
        return True  # Not enough voiced frames

    cv = np.std(voiced_rms) / (np.mean(voiced_rms) + 1e-10)
    agc_likely = cv < config.agc_coefficient_of_variation_threshold

    return not agc_likely


def run_quality_gate(audio_path: str, device_id: str, task_type: str, silence_sec: float) -> dict:
    """
    Run all quality gates and return structured results.

    Returns a dict with:
        passed: bool
        failed_checks: list of check dicts
        passed_checks: list of check dicts
        warnings: list of warning dicts
        summary: dict (for success response)
    """
    config = PipelineConfig(device_id=device_id, silence_duration_sec=silence_sec)

    # Adjust thresholds per task type
    if task_type == "sustained_vowel":
        config.min_voiced_duration_sec = 3.0
    elif task_type == "free_speech":
        config.min_voiced_duration_sec = 10.0
    elif task_type == "reading_passage":
        config.min_voiced_duration_sec = 8.0
    elif task_type == "cough":
        config.min_voiced_duration_sec = 1.0

    # Load audio
    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    all_checks = []

    # Gate 1: Silence region
    has_silence, silence_end = gate1_check_silence_region(audio, sr, config)
    if not has_silence:
        silence_end = int(config.silence_duration_sec * sr)
        if silence_end >= len(audio):
            silence_end = len(audio) // 4

    silence_rms = np.sqrt(np.mean(audio[:silence_end] ** 2)) + 1e-10
    signal_rms = np.sqrt(np.mean(audio[silence_end:] ** 2)) + 1e-10
    ratio_db = 20 * np.log10(signal_rms / silence_rms)

    all_checks.append({
        "check": "silence_region",
        "passed": has_silence,
        "value": round(ratio_db, 1),
        "threshold": 10.0,
        "unit": "dB",
        "message": None if has_silence else FAILURE_MESSAGES["silence_region"]
    })

    # Gate 2: SNR
    snr_verdict, snr_db = gate2_check_snr(audio, sr, silence_end, config)
    snr_passed = snr_verdict != "reject"
    all_checks.append({
        "check": "snr",
        "passed": snr_passed,
        "value": round(snr_db, 1),
        "threshold": config.snr_reject,
        "unit": "dB",
        "message": None if snr_passed else FAILURE_MESSAGES["snr"].format(
            value=snr_db, threshold=config.snr_reject
        )
    })

    # Gate 3: Clipping
    clipping_passed, clipping_fraction = gate3_check_clipping(audio, config)
    all_checks.append({
        "check": "clipping",
        "passed": clipping_passed,
        "value": round(clipping_fraction, 6),
        "threshold": config.max_clipping_fraction,
        "unit": "fraction",
        "message": None if clipping_passed else FAILURE_MESSAGES["clipping"].format(
            value_pct=clipping_fraction * 100,
        )
    })

    # Gate 4: Voice activity
    has_voice, voiced_dur, voice_detected = gate4_check_voice_activity(audio, sr, silence_end, config)
    all_checks.append({
        "check": "voiced_duration",
        "passed": has_voice,
        "value": round(voiced_dur, 2),
        "threshold": config.min_voiced_duration_sec,
        "unit": "seconds",
        "message": None if has_voice else FAILURE_MESSAGES["voiced_duration"].format(
            value=voiced_dur, threshold=config.min_voiced_duration_sec
        )
    })

    # Gate 5: AGC (warning only)
    agc_ok = gate5_check_agc(audio, sr, silence_end, config)
    warnings = []
    agc_detected = not agc_ok
    if agc_detected:
        warnings.append({
            "check": "agc",
            "message": (
                "Automatic gain control may be active on your device. "
                "This can reduce the accuracy of some measurements. "
                "For best results, use a headset microphone."
            )
        })

    # Separate passed vs failed
    failed = [c for c in all_checks if not c["passed"]]
    passed = [c for c in all_checks if c["passed"]]
    gate_passed = len(failed) == 0

    # Summary for success response
    summary = {
        "gate_passed": gate_passed,
        "snr_db": round(snr_db, 1),
        "snr_verdict": snr_verdict,
        "clipping_fraction": round(clipping_fraction, 6),
        "clipping_passed": clipping_passed,
        "voiced_duration_sec": round(voiced_dur, 2),
        "voice_detected": voice_detected,
        "agc_detected": agc_detected,
        "noise_reduction_applied": False,
        "warnings": [w["message"] for w in warnings]
    }

    return {
        "passed": gate_passed,
        "failed_checks": failed,
        "passed_checks": passed,
        "warnings": warnings,
        "summary": summary,
        "_silence_end_sample": silence_end,
        "_sr": sr,
        "_snr_verdict": snr_verdict,
    }
