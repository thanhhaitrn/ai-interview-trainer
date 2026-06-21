"""Convert video/audio files into text using faster-whisper (local)."""

from __future__ import annotations

from pathlib import Path

from app.transcription.schemas import (
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)


class VideoTranscriber:
    """Transcribe speech from a video/audio file with a local Whisper model.

    ``faster-whisper`` decodes the media file directly through PyAV, so MP4
    inputs work without a separate audio-extraction step or a system ffmpeg
    binary.
    """

    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _get_model(self):
        """Load the Whisper model lazily so importing stays lightweight."""
        if self._model is None:
            # Imported here so the heavy dependency only loads when used.
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(
        self,
        video_path: str,
        language: str = "en",
        beam_size: int = 5,
        word_timestamps: bool = True,
    ) -> TranscriptResult:
        """Transcribe ``video_path`` and return a JSON-friendly result.

        ``word_timestamps`` is on by default so downstream fluency analysis
        (pauses, speaking rate, mean length of run) has per-word timing.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        model = self._get_model()
        segments_iter, info = model.transcribe(
            str(path),
            language=language,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
        )

        segments = [
            TranscriptSegment(
                start=round(float(segment.start), 3),
                end=round(float(segment.end), 3),
                text=segment.text.strip(),
                words=[
                    TranscriptWord(
                        start=round(float(word.start), 3),
                        end=round(float(word.end), 3),
                        word=word.word.strip(),
                    )
                    for word in (segment.words or [])
                ],
            )
            for segment in segments_iter
        ]

        full_text = " ".join(
            segment.text for segment in segments if segment.text
        ).strip()

        return TranscriptResult(
            source_path=str(path),
            language=info.language or language,
            duration_sec=round(float(info.duration), 3),
            model=self.model_size,
            text=full_text,
            segments=segments,
        )
