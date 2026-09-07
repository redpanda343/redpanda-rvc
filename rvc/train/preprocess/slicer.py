from __future__ import annotations

import math
import os
import queue
import threading
import time
from concurrent.futures import Future
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
_AED_MODEL_USE_GPU = None
_AED_MODEL_LOCK = threading.RLock()
_AED_GPU_BATCHER = None
_AED_GPU_BATCHER_PID = None
_AED_THREAD_LOCAL = threading.local()


def _get_aed_model(use_gpu: bool = False):
    global _AED_MODEL, _AED_MODEL_PID, _AED_MODEL_USE_GPU

    pid = os.getpid()
    if (
        _AED_MODEL is not None
        and _AED_MODEL_PID == pid
        and _AED_MODEL_USE_GPU == use_gpu
    ):
        return _AED_MODEL

    with _AED_MODEL_LOCK:
        if (
            _AED_MODEL is not None
            and _AED_MODEL_PID == pid
            and _AED_MODEL_USE_GPU == use_gpu
        ):
            return _AED_MODEL

        try:
            from fireredvad import FireRedAed, FireRedAedConfig
        except ImportError as exc:
            raise RuntimeError(
                "Automatic slicing requires FireRedVAD. Install the Applio "
                "requirements (or run `pip install fireredvad==0.0.2`)."
            ) from exc

        missing = [
            name
            for name in _REQUIRED_MODEL_FILES
            if not (FIRERED_MODEL_DIR / name).is_file()
        ]
        if missing:
            expected = ", ".join(str(FIRERED_MODEL_DIR / name) for name in missing)
            raise FileNotFoundError(
                "FireRedVAD AED model files are missing. Automatic slicing requires: "
                f"{expected}"
            )

        config = FireRedAedConfig(
            use_gpu=use_gpu,
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
        _AED_MODEL_USE_GPU = use_gpu
        return _AED_MODEL


def fireredvad_cuda_available():
    setting = os.getenv("APPLIO_FIREREDVAD_DEVICE", "auto").strip().lower()
    if setting == "cpu":
        return False
    if setting not in {"auto", "cuda"}:
        raise ValueError(
            "APPLIO_FIREREDVAD_DEVICE must be 'auto', 'cpu', or 'cuda'."
        )
    try:
        import torch
    except ImportError:
        if setting == "cuda":
            raise RuntimeError("CUDA FireRedVAD requires PyTorch.")
        return False
    available = torch.cuda.is_available()
    if setting == "cuda" and not available:
        raise RuntimeError("CUDA was requested for FireRedVAD but is unavailable.")
    if not available:
        return False
    try:
        free_memory, _ = torch.cuda.mem_get_info()
    except RuntimeError:
        if setting == "cuda":
            raise
        return False
    if free_memory < 1024**3:
        if setting == "cuda":
            raise RuntimeError(
                "CUDA FireRedVAD requires at least 1 GB of currently free VRAM."
            )
        return False
    return True


def _get_thread_audio_feat():
    audio_feat = getattr(_AED_THREAD_LOCAL, "audio_feat", None)
    if audio_feat is None:
        from fireredvad.core.audio_feat import AudioFeat

        audio_feat = AudioFeat(str(FIRERED_MODEL_DIR / "cmvn.ark"))
        _AED_THREAD_LOCAL.audio_feat = audio_feat
    return audio_feat


def _postprocess_aed_probs(aed, probs, duration):
    events = {}
    for idx, event in aed.IDX2EVENT.items():
        event_probs = probs[:, idx].tolist()
        postprocessor = aed.event2postprocessor[event]
        decision = postprocessor.process(event_probs)
        events[event] = postprocessor.decision_to_segment(decision, duration)
    return events


class _GpuAedBatcher:
    def __init__(self, aed):
        import torch

        free_memory, _ = torch.cuda.mem_get_info()
        self.batch_size = 2 if free_memory >= 2 * 1024**3 else 1
        self.aed = aed
        self.requests = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, features):
        future = Future()
        self.requests.put((features, future))
        return future

    def _run_batch(self, requests):
        import torch

        features = torch.stack([item[0] for item in requests]).cuda()
        probs, _ = self.aed.model.forward(features)
        probs = probs.cpu()
        for index, (_, future) in enumerate(requests):
            future.set_result(probs[index].clone())

    def _run(self):
        import torch

        stop = False
        with torch.inference_mode():
            while not stop:
                first = self.requests.get()
                if first is None:
                    break
                pending = [first]
                deadline = time.perf_counter() + 0.002
                while len(pending) < self.batch_size * 4:
                    timeout = deadline - time.perf_counter()
                    if timeout <= 0:
                        break
                    try:
                        request = self.requests.get(timeout=timeout)
                    except queue.Empty:
                        break
                    if request is None:
                        stop = True
                        break
                    pending.append(request)

                groups = {}
                for request in pending:
                    groups.setdefault(request[0].size(0), []).append(request)
                for group in groups.values():
                    for start in range(0, len(group), self.batch_size):
                        batch = group[start : start + self.batch_size]
                        try:
                            self._run_batch(batch)
                        except Exception as error:
                            for _, future in batch:
                                future.set_exception(error)

    def close(self):
        self.requests.put(None)
        self.thread.join()


def _get_gpu_batcher():
    global _AED_GPU_BATCHER, _AED_GPU_BATCHER_PID

    pid = os.getpid()
    if _AED_GPU_BATCHER is not None and _AED_GPU_BATCHER_PID == pid:
        return _AED_GPU_BATCHER
    with _AED_MODEL_LOCK:
        if _AED_GPU_BATCHER is not None and _AED_GPU_BATCHER_PID == pid:
            return _AED_GPU_BATCHER
        aed = _get_aed_model(True)
        _AED_GPU_BATCHER = _GpuAedBatcher(aed)
        _AED_GPU_BATCHER_PID = pid
        return _AED_GPU_BATCHER


def shutdown_fireredvad_gpu():
    global _AED_GPU_BATCHER, _AED_GPU_BATCHER_PID

    if _AED_GPU_BATCHER is not None and _AED_GPU_BATCHER_PID == os.getpid():
        _AED_GPU_BATCHER.close()
    _AED_GPU_BATCHER = None
    _AED_GPU_BATCHER_PID = None


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

    def __init__(self, sr: int, use_gpu: bool = False, **_legacy_options):
        if sr <= 0:
            raise ValueError("Sampling rate must be greater than zero")
        self.sr = int(sr)
        self.use_gpu = bool(use_gpu)

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
        if self.use_gpu:
            aed = _get_aed_model(True)
            audio_feat = _get_thread_audio_feat()
            features, duration = audio_feat.extract(detector_audio)
            batcher = _get_gpu_batcher()
            futures = [
                batcher.submit(chunk)
                for chunk in features.split(aed.config.chunk_max_frame, dim=0)
            ]
            probs = [future.result() for future in futures]
            if not probs:
                return []
            import torch

            events = _postprocess_aed_probs(aed, torch.cat(probs, dim=0), duration)
        else:
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
