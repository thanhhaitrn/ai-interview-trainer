"""Acoustic voice-quality analysis (pitch, jitter, shimmer) via Praat.

These measures gauge how steady or shaky the voice is. The
classic jitter/shimmer norms come from sustained vowels, so on running speech
they are heuristic indicators rather than clinical thresholds.
"""

from __future__ import annotations

import math

import numpy as np

from app.speech_analysis.schemas import VoiceMetrics

# Pitch search range (Hz) covering typical adult male+female speech.
F0_MIN_HZ = 75.0
F0_MAX_HZ = 500.0

# Heuristic thresholds for flagging an unstable/shaky voice.
JITTER_MILD_PCT = 1.04
JITTER_SHAKY_PCT = 2.0
SHIMMER_MILD_PCT = 3.81
SHIMMER_SHAKY_PCT = 5.0
F0_CV_HIGH = 0.35

_NOTE = (
    "Jitter/shimmer norms are for sustained vowels; on running speech these "
    "are heuristic indicators of vocal steadiness, not clinical diagnoses."
)


def _clean(value: float) -> float | None:
    """Convert NaN/inf Praat results to None for clean JSON output."""
    if value is None or math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), 4)


def analyze_voice(
    samples: np.ndarray,
    sample_rate: int,
    f0min: float = F0_MIN_HZ,
    f0max: float = F0_MAX_HZ,
) -> VoiceMetrics:
    """Compute pitch/jitter/shimmer voice-stability metrics from a waveform."""
    import parselmouth
    from parselmouth.praat import call

    if samples.size == 0:
        raise ValueError("No audio samples to analyze.")

    sound = parselmouth.Sound(
        samples.astype(np.float64),
        sampling_frequency=float(sample_rate),
    )

    pitch = sound.to_pitch(pitch_floor=f0min, pitch_ceiling=f0max)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    voiced_fraction = float(voiced.size) / float(f0.size) if f0.size else 0.0

    if voiced.size:
        f0_mean = float(np.mean(voiced))
        f0_std = float(np.std(voiced))
        f0_min = float(np.min(voiced))
        f0_max = float(np.max(voiced))
        f0_cv = f0_std / f0_mean if f0_mean else None
    else:
        f0_mean = f0_std = f0_min = f0_max = f0_cv = None

    # Jitter/shimmer from the periodic point process.
    try:
        point_process = call(
            sound, "To PointProcess (periodic, cc)", f0min, f0max
        )
        jitter_local = call(
            point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        shimmer_local = call(
            [sound, point_process],
            "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        )
    except Exception:
        jitter_local = float("nan")
        shimmer_local = float("nan")

    # Praat reports jitter/shimmer as fractions; express as percentages.
    jitter_pct = _clean(jitter_local * 100) if not math.isnan(jitter_local) else None
    shimmer_pct = (
        _clean(shimmer_local * 100) if not math.isnan(shimmer_local) else None
    )

    intensity = sound.to_intensity()
    intensity_values = intensity.values[0]
    intensity_values = intensity_values[np.isfinite(intensity_values)]
    if intensity_values.size:
        intensity_mean = float(np.mean(intensity_values))
        intensity_std = float(np.std(intensity_values))
    else:
        intensity_mean = intensity_std = None

    indicators: list[str] = []
    if jitter_pct is not None and jitter_pct > JITTER_SHAKY_PCT:
        indicators.append(f"Elevated jitter ({jitter_pct:.2f}%).")
    if shimmer_pct is not None and shimmer_pct > SHIMMER_SHAKY_PCT:
        indicators.append(f"Elevated shimmer ({shimmer_pct:.2f}%).")
    if f0_cv is not None and f0_cv > F0_CV_HIGH:
        indicators.append(f"High pitch variability (F0 CV {f0_cv:.2f}).")

    shaky = (
        (jitter_pct is not None and jitter_pct > JITTER_SHAKY_PCT)
        or (shimmer_pct is not None and shimmer_pct > SHIMMER_SHAKY_PCT)
    )
    mild = (
        (jitter_pct is not None and jitter_pct > JITTER_MILD_PCT)
        or (shimmer_pct is not None and shimmer_pct > SHIMMER_MILD_PCT)
    )
    if shaky:
        tremor_label = "shaky"
    elif mild:
        tremor_label = "mildly_unstable"
    else:
        tremor_label = "steady"

    return VoiceMetrics(
        sample_rate=int(sample_rate),
        voiced_fraction=round(voiced_fraction, 3),
        f0_mean_hz=_clean(f0_mean) if f0_mean is not None else None,
        f0_std_hz=_clean(f0_std) if f0_std is not None else None,
        f0_min_hz=_clean(f0_min) if f0_min is not None else None,
        f0_max_hz=_clean(f0_max) if f0_max is not None else None,
        f0_cv=_clean(f0_cv) if f0_cv is not None else None,
        jitter_local_pct=jitter_pct,
        shimmer_local_pct=shimmer_pct,
        intensity_mean_db=_clean(intensity_mean) if intensity_mean is not None else None,
        intensity_std_db=_clean(intensity_std) if intensity_std is not None else None,
        tremor_label=tremor_label,
        tremor_indicators=indicators,
        note=_NOTE,
    )
