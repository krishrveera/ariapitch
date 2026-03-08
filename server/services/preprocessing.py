"""
Preprocessing Service
Runs the B2AI-aligned audio preprocessing pipeline.
"""
import numpy as np
import librosa
import tempfile
from scipy.signal import butter, sosfilt
from scipy.io import wavfile
import pyloudnorm as pyln
import noisereduce as nr


class PipelineConfig:
    """Configuration for preprocessing pipeline."""
    def __init__(self, device_id="unknown", silence_duration_sec=3.0):
        self.target_sr = 16000
        self.device_id = device_id
        self.silence_duration_sec = silence_duration_sec
        self.butter_order = 8
        self.highpass_cutoff_hz = 80.0
        self.highpass_order = 4
        self.apply_cmvn = True
        self.target_lufs = -23.0
        self.noise_reduce_strength = 0.7
        self.vad_hop_length_ms = 10.0
        self.snr_ideal = 42.0
        self.snr_acceptable = 30.0


def step1_resample(audio: np.ndarray, sr: int, config: PipelineConfig) -> tuple:
    """Resample to 16 kHz with Butterworth anti-aliasing filter."""
    if sr == config.target_sr:
        return audio, sr

    if sr > config.target_sr:
        # Apply Butterworth low-pass anti-aliasing filter
        nyquist_new = config.target_sr / 2
        nyquist_old = sr / 2
        normalized_cutoff = min(nyquist_new / nyquist_old, 0.9999)

        sos = butter(config.butter_order, normalized_cutoff, btype='low', output='sos')
        audio_filtered = sosfilt(sos, audio).astype(np.float32)
    else:
        audio_filtered = audio

    # Resample
    audio_resampled = librosa.resample(audio_filtered, orig_sr=sr, target_sr=config.target_sr)
    return audio_resampled, config.target_sr


def step2_highpass_filter(audio: np.ndarray, sr: int, config: PipelineConfig) -> np.ndarray:
    """Apply high-pass filter for microphone roll-off compensation."""
    nyquist = sr / 2
    normalized_cutoff = config.highpass_cutoff_hz / nyquist

    if normalized_cutoff >= 1.0:
        return audio

    sos = butter(config.highpass_order, normalized_cutoff, btype='high', output='sos')
    audio_filtered = sosfilt(sos, audio).astype(np.float32)
    return audio_filtered


def step3_cmvn_compensation(audio: np.ndarray, sr: int, config: PipelineConfig) -> np.ndarray:
    """Apply Cepstral Mean and Variance Normalization."""
    if not config.apply_cmvn:
        return audio

    n_fft = 512
    hop_length = int(config.vad_hop_length_ms * sr / 1000)

    # Compute STFT
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(S)
    phase = np.angle(S)

    # Compute mean log-magnitude spectrum
    log_mag = np.log(magnitude + 1e-10)
    mean_log_spectrum = np.mean(log_mag, axis=1, keepdims=True)

    # Subtract mean and normalize variance
    log_mag_normalized = log_mag - mean_log_spectrum
    std_log_spectrum = np.std(log_mag, axis=1, keepdims=True) + 1e-10
    log_mag_normalized = log_mag_normalized / std_log_spectrum

    # Reconstruct
    magnitude_normalized = np.exp(log_mag_normalized)
    S_normalized = magnitude_normalized * np.exp(1j * phase)
    audio_cmvn = librosa.istft(S_normalized, hop_length=hop_length, length=len(audio))
    audio_cmvn = audio_cmvn.astype(np.float32)

    # Preserve original peak level
    peak_before = np.max(np.abs(audio))
    peak_after = np.max(np.abs(audio_cmvn))
    if peak_after > 0:
        audio_cmvn = audio_cmvn * (peak_before / peak_after)

    return audio_cmvn


def step4_normalize_lufs(audio: np.ndarray, sr: int, config: PipelineConfig) -> tuple:
    """Normalize loudness to consistent LUFS target."""
    meter = pyln.Meter(sr)
    loudness_before = meter.integrated_loudness(audio)

    # Apply constant gain normalization
    audio_normalized = pyln.normalize.loudness(audio, loudness_before, config.target_lufs)

    # Prevent clipping
    peak = np.max(np.abs(audio_normalized))
    if peak > 0.99:
        audio_normalized = audio_normalized * (0.99 / peak)

    loudness_after = meter.integrated_loudness(audio_normalized)
    return audio_normalized, loudness_before, loudness_after


def step5_conditional_noise_reduction(audio: np.ndarray, sr: int, snr_verdict: str,
                                       silence_end_sample: int, config: PipelineConfig) -> tuple:
    """Apply noise reduction if SNR is marginal."""
    if snr_verdict == "ideal":
        return audio, False

    # Apply light noise reduction for acceptable SNR
    noise_clip = audio[:silence_end_sample]

    if len(noise_clip) < sr // 4:
        return audio, False

    audio_cleaned = nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_clip,
        prop_decrease=config.noise_reduce_strength,
        stationary=True,
        n_fft=512,
        hop_length=int(config.vad_hop_length_ms * sr / 1000),
    )

    return audio_cleaned, True


def step6_trim_silence(audio: np.ndarray, silence_end_sample: int) -> np.ndarray:
    """Remove leading silence region."""
    return audio[silence_end_sample:]


def step7_export_wav(audio: np.ndarray, sr: int, output_path: str) -> str:
    """Export as 16-bit PCM WAV."""
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, sr, audio_int16)
    return output_path


def run_preprocessing(audio_path: str, device_id: str, silence_sec: float,
                     gate_result: dict) -> tuple:
    """
    Run the full preprocessing pipeline.

    Returns:
        (processed_wav_path, preprocessing_info_dict)
    """
    config = PipelineConfig(device_id=device_id, silence_duration_sec=silence_sec)

    # Load audio
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    original_sr = sr
    original_duration = len(audio) / sr

    # Get silence end from gate result
    silence_end = gate_result.get("_silence_end_sample", int(silence_sec * sr))
    snr_verdict = gate_result.get("_snr_verdict", "acceptable")

    # Step 1: Resample
    audio, sr = step1_resample(audio, sr, config)
    silence_end_resampled = int(silence_end * config.target_sr / original_sr)

    # Step 2: High-pass filter
    audio = step2_highpass_filter(audio, sr, config)

    # Step 3: CMVN
    audio = step3_cmvn_compensation(audio, sr, config)

    # Step 4: LUFS normalize
    audio, lufs_before, lufs_after = step4_normalize_lufs(audio, sr, config)

    # Step 5: Conditional noise reduction
    audio, noise_reduction_applied = step5_conditional_noise_reduction(
        audio, sr, snr_verdict, silence_end_resampled, config
    )

    # Step 6: Trim silence
    audio = step6_trim_silence(audio, silence_end_resampled)

    # Step 7: Export
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    step7_export_wav(audio, sr, temp_wav.name)

    preproc_info = {
        "original_sr": original_sr,
        "output_sr": sr,
        "original_duration_sec": round(original_duration, 2),
        "output_duration_sec": round(len(audio) / sr, 2),
        "lufs_before": round(lufs_before, 1),
        "lufs_after": round(lufs_after, 1),
        "highpass_applied": True,
        "cmvn_applied": config.apply_cmvn,
        "noise_reduction_applied": noise_reduction_applied,
        "device_id": device_id
    }

    return temp_wav.name, preproc_info
