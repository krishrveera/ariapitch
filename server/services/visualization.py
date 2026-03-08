"""
Visualization Service
Generates base64-encoded PNG images for quality gate, preprocessing, and features.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List
import librosa
import librosa.display


def plot_to_base64(fig) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_waveform_plot(audio: np.ndarray, sr: int, title: str = "Waveform") -> str:
    """Generate waveform visualization."""
    fig, ax = plt.subplots(figsize=(12, 3))
    time = np.arange(len(audio)) / sr
    ax.plot(time, audio, linewidth=0.5, color='steelblue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return plot_to_base64(fig)


def generate_spectrogram_plot(audio: np.ndarray, sr: int, title: str = "Spectrogram") -> str:
    """Generate spectrogram visualization."""
    fig, ax = plt.subplots(figsize=(12, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    return plot_to_base64(fig)


def generate_snr_analysis_plot(audio: np.ndarray, sr: int, silence_end: int,
                               snr_db: float, snr_verdict: str) -> str:
    """Generate SNR analysis visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Waveform with silence region marked
    time = np.arange(len(audio)) / sr
    axes[0].plot(time, audio, linewidth=0.5, color='steelblue')
    axes[0].axvline(silence_end / sr, color='red', linestyle='--',
                    label='Silence End', linewidth=2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Signal with Silence Region | SNR: {snr_db:.1f} dB ({snr_verdict})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RMS energy comparison
    silence_rms = np.sqrt(np.mean(audio[:silence_end] ** 2))
    signal_rms = np.sqrt(np.mean(audio[silence_end:] ** 2))

    axes[1].bar(['Noise (Silence Region)', 'Signal (Voice Region)'],
                [silence_rms, signal_rms],
                color=['coral', 'limegreen'])
    axes[1].set_ylabel('RMS Energy')
    axes[1].set_title('Noise vs Signal Energy')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return plot_to_base64(fig)


def generate_vad_plot(audio: np.ndarray, sr: int, voiced_mask: np.ndarray,
                      hop_length: int, voiced_duration: float) -> str:
    """Generate Voice Activity Detection visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Waveform
    time = np.arange(len(audio)) / sr
    axes[0].plot(time, audio, linewidth=0.5, color='steelblue')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Waveform | Voiced Duration: {voiced_duration:.2f}s')
    axes[0].grid(True, alpha=0.3)

    # VAD mask
    time_frames = np.arange(len(voiced_mask)) * hop_length / sr
    axes[1].fill_between(time_frames, 0, voiced_mask.astype(int),
                         color='limegreen', alpha=0.7, label='Voiced')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Voice Activity')
    axes[1].set_title('Voice Activity Detection (VAD)')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return plot_to_base64(fig)


def generate_preprocessing_comparison(before: np.ndarray, after: np.ndarray,
                                      sr: int, title: str) -> str:
    """Generate before/after preprocessing comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Before waveform
    time_before = np.arange(len(before)) / sr
    axes[0, 0].plot(time_before, before, linewidth=0.5, color='coral')
    axes[0, 0].set_title(f'{title} - Before')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)

    # After waveform
    time_after = np.arange(len(after)) / sr
    axes[0, 1].plot(time_after, after, linewidth=0.5, color='limegreen')
    axes[0, 1].set_title(f'{title} - After')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)

    # Before spectrogram
    D_before = librosa.amplitude_to_db(np.abs(librosa.stft(before)), ref=np.max)
    librosa.display.specshow(D_before, sr=sr, x_axis='time', y_axis='hz',
                            ax=axes[1, 0], cmap='magma')
    axes[1, 0].set_title('Spectrogram - Before')

    # After spectrogram
    D_after = librosa.amplitude_to_db(np.abs(librosa.stft(after)), ref=np.max)
    librosa.display.specshow(D_after, sr=sr, x_axis='time', y_axis='hz',
                            ax=axes[1, 1], cmap='magma')
    axes[1, 1].set_title('Spectrogram - After')

    plt.tight_layout()
    return plot_to_base64(fig)


def generate_quality_gate_summary(gate_result: dict) -> str:
    """Generate quality gate checks summary visualization."""
    all_checks = gate_result["passed_checks"] + gate_result["failed_checks"]

    fig, ax = plt.subplots(figsize=(10, 6))

    check_names = []
    check_values = []
    check_thresholds = []
    check_colors = []

    for check in all_checks:
        check_names.append(check["check"].replace("_", " ").title())
        check_values.append(check["value"])
        check_thresholds.append(check["threshold"])
        check_colors.append('limegreen' if check["passed"] else 'coral')

    x = np.arange(len(check_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, check_values, width, label='Measured', color=check_colors, alpha=0.8)
    bars2 = ax.bar(x + width/2, check_thresholds, width, label='Threshold',
                   color='gray', alpha=0.5)

    ax.set_xlabel('Quality Check')
    ax.set_ylabel('Value')
    ax.set_title('Quality Gate Checks')
    ax.set_xticks(x)
    ax.set_xticklabels(check_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return plot_to_base64(fig)


def generate_feature_heatmap(features_2d: np.ndarray, title: str,
                             y_label: str = "Coefficients") -> str:
    """Generate 2D feature heatmap (for MFCCs, spectrograms, PPGs)."""
    fig, ax = plt.subplots(figsize=(12, 4))

    cmap = 'coolwarm' if 'mfcc' in title.lower() else 'viridis'

    im = ax.imshow(features_2d, aspect='auto', origin='lower', cmap=cmap)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Time Frames")
    fig.colorbar(im, ax=ax, format='%+2.2f')

    plt.tight_layout()
    return plot_to_base64(fig)


def generate_feature_time_series(features_1d: np.ndarray, title: str) -> str:
    """Generate 1D feature time series plot (for pitch, loudness)."""
    fig, ax = plt.subplots(figsize=(12, 3))

    color = 'coral' if 'pitch' in title.lower() or 'f0' in title.lower() else 'dodgerblue'

    ax.plot(features_1d, color=color, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Magnitude / Value")
    ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    return plot_to_base64(fig)


def generate_b2ai_feature_visualizations(raw_features: dict) -> Dict[str, str]:
    """
    Generate visualizations for all B2AI features.
    Returns dict of {feature_name: base64_image}
    """
    import torch
    import pandas as pd

    visualizations = {}

    def process_features(features_dict, prefix=""):
        for key, value in features_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key

            # Nested dicts - recurse
            if isinstance(value, dict):
                process_features(value, full_key)
                continue

            # Convert tensors to numpy
            if isinstance(value, torch.Tensor):
                value = value.squeeze().detach().cpu().numpy()
            elif isinstance(value, pd.DataFrame):
                continue  # Skip DataFrames for now (too many columns)
            elif not isinstance(value, np.ndarray):
                continue

            value = np.squeeze(value)

            # 2D features (spectrograms, MFCCs, PPGs)
            if value.ndim == 2:
                y_label = "MFCC Coefficients" if 'mfcc' in key.lower() else \
                         "Phoneme Classes" if 'ppg' in key.lower() else \
                         "Frequency Bins"

                visualizations[full_key] = generate_feature_heatmap(
                    value, f"2D Feature: {full_key}", y_label
                )

            # 1D features (pitch contour, loudness)
            elif value.ndim == 1 and len(value) > 1:
                visualizations[full_key] = generate_feature_time_series(
                    value, f"1D Time-Series: {full_key}"
                )

    process_features(raw_features)
    return visualizations


def generate_all_visualizations(audio_path: str, gate_result: dict,
                                preprocessing_info: dict) -> Dict[str, str]:
    """
    Generate all visualizations for demo endpoint.

    Returns dict with base64-encoded PNG images for:
    - Quality gate checks
    - SNR analysis
    - VAD analysis
    - Preprocessing comparisons
    - B2AI features
    """
    import librosa

    visualizations = {}

    # Load original audio
    audio_orig, sr_orig = librosa.load(audio_path, sr=None, mono=True)

    # Original waveform
    visualizations["original_waveform"] = generate_waveform_plot(
        audio_orig, sr_orig, "Original Audio Waveform"
    )

    # Original spectrogram
    visualizations["original_spectrogram"] = generate_spectrogram_plot(
        audio_orig, sr_orig, "Original Audio Spectrogram"
    )

    # Quality gate summary
    visualizations["quality_gate_summary"] = generate_quality_gate_summary(gate_result)

    # SNR analysis
    if "_silence_end_sample" in gate_result and "_sr" in gate_result:
        silence_end = gate_result["_silence_end_sample"]
        sr = gate_result["_sr"]
        snr_db = gate_result["summary"]["snr_db"]
        snr_verdict = gate_result["summary"]["snr_verdict"]

        visualizations["snr_analysis"] = generate_snr_analysis_plot(
            audio_orig, sr, silence_end, snr_db, snr_verdict
        )

    # VAD analysis (if available)
    # Note: We'd need to re-run VAD to get the mask, so skip for now

    return visualizations
