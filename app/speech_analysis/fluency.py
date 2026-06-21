"""Fluency analysis from a transcript and its word-level timestamps."""

from __future__ import annotations

import re

from app.speech_analysis.schemas import FluencyMetrics
from app.transcription.schemas import TranscriptWord

# Gap (seconds) between words that counts as a pause / run boundary.
PAUSE_THRESHOLD_SEC = 0.3
# Pauses at/above this are treated as hesitation-level long pauses.
LONG_PAUSE_THRESHOLD_SEC = 1.0

# Single-token disfluencies and common verbal fillers. Note: Whisper tends to
# clean up "um/uh", so filler counts are a lower bound on real disfluency.
FILLER_WORDS = {
    "um", "umm", "uhm", "uh", "uhh", "er", "err", "erm", "ah", "ahh",
    "hmm", "mm", "mhm", "eh", "huh", "like", "basically", "actually",
    "literally",
}
# Multi-word fillers checked as token n-grams.
FILLER_PHRASES = (
    ("you", "know"),
    ("i", "mean"),
    ("sort", "of"),
    ("kind", "of"),
)

_WORD_RE = re.compile(r"[a-z0-9']+")


def _normalize(token: str) -> str:
    """Lowercase a word token and strip surrounding punctuation."""
    match = _WORD_RE.search(token.lower())
    return match.group(0) if match else ""


def _round(value: float, ndigits: int = 2) -> float:
    return round(float(value), ndigits)


def analyze_fluency(
    words: list[TranscriptWord],
    pause_threshold: float = PAUSE_THRESHOLD_SEC,
    long_pause_threshold: float = LONG_PAUSE_THRESHOLD_SEC,
) -> FluencyMetrics:
    """Compute delivery metrics from ordered word timestamps."""
    tokens = [_normalize(word.word) for word in words]
    total_words = sum(1 for token in tokens if token)

    if total_words == 0:
        return FluencyMetrics(
            total_words=0,
            span_sec=0.0,
            speech_rate_wpm=0.0,
            articulation_rate_wpm=0.0,
            pause_count=0,
            pause_total_sec=0.0,
            mean_pause_sec=0.0,
            long_pause_count=0,
            filler_count=0,
            filler_per_min=0.0,
            repetition_count=0,
            mean_length_of_run=0.0,
            run_count=0,
            max_run_length=0,
            warnings=["No words with timestamps to analyze."],
        )

    # Pauses and runs from inter-word gaps.
    pause_durations: list[float] = []
    long_pause_count = 0
    run_lengths: list[int] = []
    current_run = 1
    for prev, current in zip(words, words[1:]):
        gap = current.start - prev.end
        if gap >= pause_threshold:
            pause_durations.append(gap)
            if gap >= long_pause_threshold:
                long_pause_count += 1
            run_lengths.append(current_run)
            current_run = 1
        else:
            current_run += 1
    run_lengths.append(current_run)

    pause_count = len(pause_durations)
    pause_total = sum(pause_durations)
    mean_pause = pause_total / pause_count if pause_count else 0.0

    span_sec = max(words[-1].end - words[0].start, 0.0)
    phonation_sec = max(span_sec - pause_total, 1e-6)
    speech_rate = total_words / span_sec * 60 if span_sec > 0 else 0.0
    articulation_rate = total_words / phonation_sec * 60

    # Filler words (single tokens + phrases).
    filler_breakdown: dict[str, int] = {}
    non_empty = [token for token in tokens if token]
    for token in non_empty:
        if token in FILLER_WORDS:
            filler_breakdown[token] = filler_breakdown.get(token, 0) + 1
    for first, second in FILLER_PHRASES:
        phrase = f"{first} {second}"
        for a, b in zip(non_empty, non_empty[1:]):
            if a == first and b == second:
                filler_breakdown[phrase] = filler_breakdown.get(phrase, 0) + 1
    filler_count = sum(filler_breakdown.values())
    filler_per_min = filler_count / span_sec * 60 if span_sec > 0 else 0.0

    # Immediate word repetitions / stumbles (e.g. "I I think", "the the").
    repetition_count = sum(
        1
        for a, b in zip(non_empty, non_empty[1:])
        if a == b and len(a) > 1
    )

    mean_run = sum(run_lengths) / len(run_lengths)

    warnings: list[str] = []
    if speech_rate < 100:
        warnings.append("Slow speaking pace (below ~100 wpm).")
    elif speech_rate > 180:
        warnings.append("Fast speaking pace (above ~180 wpm).")
    if filler_per_min > 8:
        warnings.append("High filler rate (>8 per minute).")
    if long_pause_count >= 3:
        warnings.append("Frequent long pauses suggest hesitation.")
    if repetition_count >= 3:
        warnings.append("Multiple word repetitions/stumbles detected.")

    return FluencyMetrics(
        total_words=total_words,
        span_sec=_round(span_sec),
        speech_rate_wpm=_round(speech_rate, 1),
        articulation_rate_wpm=_round(articulation_rate, 1),
        pause_count=pause_count,
        pause_total_sec=_round(pause_total),
        mean_pause_sec=_round(mean_pause),
        long_pause_count=long_pause_count,
        filler_count=filler_count,
        filler_per_min=_round(filler_per_min),
        repetition_count=repetition_count,
        mean_length_of_run=_round(mean_run),
        run_count=len(run_lengths),
        max_run_length=max(run_lengths),
        filler_breakdown=filler_breakdown,
        warnings=warnings,
    )
