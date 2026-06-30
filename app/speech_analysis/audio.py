"""Decode the audio track of a media file into a mono waveform."""

from __future__ import annotations

import numpy as np


def has_audio_stream(path: str) -> bool:
    """Return True if ``path`` contains at least one decodable audio stream."""
    import av

    container = av.open(str(path))
    try:
        return len(container.streams.audio) > 0
    finally:
        container.close()


def decode_audio_mono(
    path: str,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """Decode ``path`` to a mono float32 waveform resampled to ``target_sr``.

    Uses PyAV (already pulled in by faster-whisper), so MP4 audio is read
    directly without a separate ffmpeg binary.
    """
    import av

    container = av.open(str(path))
    chunks: list[np.ndarray] = []
    try:
        audio_streams = container.streams.audio
        if not audio_streams:
            raise ValueError(f"No audio stream found in {path}")

        resampler = av.audio.resampler.AudioResampler(
            format="flt",
            layout="mono",
            rate=target_sr,
        )

        def _collect(frames: object) -> None:
            frame_list = frames if isinstance(frames, list) else [frames]
            for frame in frame_list:
                if frame is None:
                    continue
                array = np.asarray(frame.to_ndarray()).reshape(-1)
                if array.size:
                    chunks.append(array)

        for frame in container.decode(audio_streams[0]):
            _collect(resampler.resample(frame))

        # Flush any buffered samples held by the resampler.
        try:
            _collect(resampler.resample(None))
        except (ValueError, TypeError, av.error.EOFError):
            pass
    finally:
        container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32), target_sr

    return np.concatenate(chunks).astype(np.float32), target_sr
