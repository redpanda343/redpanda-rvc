from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

FIRERED_SAMPLE_RATE = 16000
FIRERED_MODEL_DIR = (
    Path(__file__).resolve().parents[3]
    / "rvc"
    / "models"
    / "pretraineds"
    / "FireRedVAD"
    / "AED"
)
_REQUIRED_MODEL_FILES = ("cmvn.ark", "model.pth.tar")

VOICE_PADDING_MS = 120
MERGE_VOICE_GAP_MS = 400

_AED_MODEL = None
_AED_MODEL_PID = None


def _get_aed_model():
    global _AED_MODEL, _AED_MODEL_PID

    pid = os.getpid()
    if _AED_MODEL is not None and _AED_MODEL_PID == pid:
        return _AED_MODEL

    try:
        from fireredvad import FireRedAed, FireRedAedConfig
    except ImportError as exc:
        raise RuntimeError(
            "Automatic slicing requires FireRedVAD. Install the Applio "
            "requirements (or run `pip install fireredvad==0.0.2`)."
        ) from exc

    missing = [
        name for name in _REQUIRED_MODEL_FILES if not (FIRERED_MODEL_DIR / name).is_file()
    ]
    if missing:
        expected = ", ".join(str(FIRERED_MODEL_DIR / name) for name in missing)
        raise FileNotFoundError(
            "FireRedVAD AED model files are missing. Automatic slicing requires: "
            f"{expected}"
        )

    config = FireRedAedConfig(
        use_gpu=False,
        smooth_window_size=5,
        speech_threshold=0.4,
        singing_threshold=0.5,
        music_threshold=0.5,
        min_event_frame=20,
        max_event_frame=3000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0,
        chunk_max_frame=30000,
    )
    _AED_MODEL = FireRedAed.from_pretrained(str(FIRERED_MODEL_DIR), config)
    _AED_MODEL_PID = pid
    return _AED_MODEL


def _merge_intervals(intervals, duration_s: float):
    if not intervals or duration_s <= 0:
        return []

    clean = []
    for start, end in intervals:
        start = max(0.0, min(float(start), duration_s))
        end = max(0.0, min(float(end), duration_s))
        if end > start:
            clean.append((start, end))
    if not clean:
        return []

    clean.sort(key=lambda item: (item[0], item[1]))
    merge_gap_s = MERGE_VOICE_GAP_MS / 1000.0

    merged = []
    current_start, current_end = clean[0]
    for start, end in clean[1:]:
        if start <= current_end + merge_gap_s:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))

    padding_s = VOICE_PADDING_MS / 1000.0
    padded = [
        (max(0.0, start - padding_s), min(duration_s, end + padding_s))
        for start, end in merged
    ]

    result = []
    for start, end in padded:
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


class Slicer:

    def __init__(self, sr: int, **_legacy_options):
        if sr <= 0:
            raise ValueError("Sampling rate must be greater than zero")
        self.sr = int(sr)

    def _to_firered_pcm16(self, waveform: np.ndarray) -> np.ndarray:
        samples = waveform.mean(axis=0) if waveform.ndim > 1 else waveform
        samples = np.asarray(samples, dtype=np.float32)

        if self.sr != FIRERED_SAMPLE_RATE:
            divisor = math.gcd(self.sr, FIRERED_SAMPLE_RATE)
            samples = resample_poly(
                samples,
                FIRERED_SAMPLE_RATE // divisor,
                self.sr // divisor,
            ).astype(np.float32, copy=False)

        return np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)

    @staticmethod
    def merge_voice_intervals(intervals, duration_s: float):
        return _merge_intervals(intervals, duration_s)

    def detect_voice_intervals_16k(self, waveform: np.ndarray):
        samples = np.asarray(waveform, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=0)
        if samples.size < int(FIRERED_SAMPLE_RATE * 0.025):
            return []

        detector_audio = np.rint(
            np.clip(samples, -1.0, 1.0) * 32767.0
        ).astype(np.int16)
        aed = _get_aed_model()
        result, _ = aed.detect(detector_audio)
        events = result.get("event2timestamps", {})
        intervals = list(events.get("speech", ()))
        intervals.extend(events.get("singing", ()))
        return intervals

    def slice(self, waveform: np.ndarray):
        waveform = np.asarray(waveform)
        sample_count = waveform.shape[-1] if waveform.ndim > 1 else waveform.shape[0]
        if sample_count == 0:
            return []

        detector_audio = self._to_firered_pcm16(waveform)
        if detector_audio.size < int(FIRERED_SAMPLE_RATE * 0.025):
            return []

        detector_float = detector_audio.astype(np.float32) / 32767.0
        voice_intervals = self.detect_voice_intervals_16k(detector_float)
        duration_s = sample_count / self.sr
        intervals = _merge_intervals(voice_intervals, duration_s)

        chunks = []
        for start_s, end_s in intervals:
            start = max(0, int(round(start_s * self.sr)))
            end = min(sample_count, int(round(end_s * self.sr)))
            if end > start:
                if waveform.ndim > 1:
                    chunks.append(waveform[:, start:end])
                else:
                    chunks.append(waveform[start:end])
        return chunks
