import concurrent.futures
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time

def strtobool(val):
    """Convert a string representation of truth to a bool."""
    return val.lower() in ("yes", "true", "t", "y", "1")


import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

now_directory = os.getcwd()
sys.path.append(now_directory)

import logging

from rvc.train.preprocess.slicer import Slicer

logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)

OVERLAP = 0.3
PERCENTAGE = 3.0
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
POST_NORMALIZATION_MAX_GAIN = 4.0
HIGH_PASS_CUTOFF = 48
SAMPLE_RATE_16K = 16000
AUTOMATIC_VAD_BLOCK_SECONDS = 180.0
AUTOMATIC_VAD_CONTEXT_SECONDS = 2.0
AUTOMATIC_DECODE_BLOCK_SECONDS = 60.0
AUTOMATIC_PROCESS_CONTEXT_SECONDS = 1.0
SUPPORTED_DATASET_FORMATS = {"wav", "flac"}
SIMPLE_SILENCE_THRESHOLD_DB = -45.0
SIMPLE_MIN_SILENCE_SECONDS = 0.5
SIMPLE_TRUNCATE_TO_SECONDS = 0.5
SIMPLE_BLEND_FRAMES = 100


def normalize_dataset_format(dataset_format: str) -> str:
    normalized_format = str(dataset_format).strip().lower()
    if normalized_format not in SUPPORTED_DATASET_FORMATS:
        raise ValueError(
            f"Unsupported dataset format '{dataset_format}'. Expected WAV or FLAC."
        )
    return normalized_format


def write_training_audio(
    directory: str,
    stem: str,
    sample_rate: int,
    audio: np.ndarray,
    dataset_format: str,
):
    """Write a processed training slice without changing the existing WAV path."""
    if dataset_format == "wav":
        wavfile.write(
            os.path.join(directory, f"{stem}.wav"),
            sample_rate,
            audio.astype(np.float32),
        )
        return

    audio = np.asarray(audio, dtype=np.float32)
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"Cannot write non-finite audio samples to {stem}.flac")
    sf.write(
        os.path.join(directory, f"{stem}.flac"),
        np.clip(audio, -1.0, 1.0),
        sample_rate,
        format="FLAC",
        subtype="PCM_24",
    )


def clear_flac_preprocess_artifacts(exp_dir: str):
    """Remove only FLAC-derived caches so a FLAC reprocess cannot reuse stale data."""
    patterns_by_directory = {
        "sliced_audios": (".flac", ".spec.pt"),
        "sliced_audios_16k": (".flac",),
        "f0": (".flac.npy",),
        "f0_voiced": (".flac.npy",),
        "extracted": (".flac.npy",),
    }
    for directory_name, suffixes in patterns_by_directory.items():
        directory = os.path.join(exp_dir, directory_name)
        if not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            if filename.lower().endswith(suffixes):
                os.remove(os.path.join(directory, filename))

    filelist_path = os.path.join(exp_dir, "filelist.txt")
    if os.path.isfile(filelist_path):
        os.remove(filelist_path)


def clear_simple_preprocess_artifacts(exp_dir: str):
    patterns_by_directory = {
        "sliced_audios": (".wav", ".flac", ".spec.pt"),
        "sliced_audios_16k": (".wav", ".flac"),
        "f0": (".wav.npy", ".flac.npy"),
        "f0_voiced": (".wav.npy", ".flac.npy"),
        "extracted": (".wav.npy", ".flac.npy"),
    }
    for directory_name, suffixes in patterns_by_directory.items():
        directory = os.path.join(exp_dir, directory_name)
        if not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            if filename.lower().endswith(suffixes):
                os.remove(os.path.join(directory, filename))

    filelist_path = os.path.join(exp_dir, "filelist.txt")
    if os.path.isfile(filelist_path):
        os.remove(filelist_path)


def _ffmpeg_path():
    bundled_ffmpeg = os.path.join(now_directory, "ffmpeg.exe")
    if os.name == "nt" and os.path.isfile(bundled_ffmpeg):
        return bundled_ffmpeg
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _clean_audio_path(file: str) -> str:
    if os.name == "nt":
        file = file.replace("/", "\\")
    return file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")


def load_audio_ffmpeg(file: str, sample_rate: int) -> np.ndarray:
    file = _clean_audio_path(file)
    command = [
        _ffmpeg_path(),
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).flatten()


def truncate_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = SIMPLE_SILENCE_THRESHOLD_DB,
    minimum_silence: float = SIMPLE_MIN_SILENCE_SECONDS,
    truncate_to: float = SIMPLE_TRUNCATE_TO_SECONDS,
    blend_frames: int = SIMPLE_BLEND_FRAMES,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio

    threshold = 10.0 ** (threshold_db / 20.0)
    silent = np.abs(audio) < threshold
    boundaries = np.flatnonzero(
        np.diff(np.pad(silent.astype(np.int8), (1, 1)))
    )
    if boundaries.size == 0:
        return audio

    minimum_frames = max(1, int(round(minimum_silence * sample_rate)))
    truncate_frames = max(0, int(round(truncate_to * sample_rate)))
    cuts = []
    for start, end in boundaries.reshape(-1, 2):
        silence_frames = int(end - start)
        if silence_frames < minimum_frames:
            continue

        output_frames = min(truncate_frames, silence_frames)
        cut_frames = silence_frames - output_frames
        if cut_frames <= 0:
            continue

        cut_start = int(start + output_frames // 2)
        cut_end = cut_start + cut_frames
        cuts.append((cut_start, cut_end))

    if not cuts:
        return audio

    parts = []
    cursor = 0
    for cut_start, cut_end in cuts:
        splice_frames = min(
            blend_frames,
            cut_start * 2,
            (len(audio) - cut_end) * 2,
        )
        if splice_frames > 0:
            half_blend = splice_frames // 2
            blend_start = cut_start - half_blend
            right_start = cut_end - half_blend
            left = audio[blend_start : blend_start + splice_frames]
            right = audio[right_start : right_start + splice_frames]
            weights = np.arange(splice_frames, dtype=np.float32) / splice_frames
            blended = left * (1.0 - weights) + right * weights
            parts.append(audio[cursor:blend_start])
            parts.append(blended)
            cursor = right_start + splice_frames
        else:
            parts.append(audio[cursor:cut_start])
            cursor = cut_end

    parts.append(audio[cursor:])
    return np.concatenate(parts)


def load_audio_ffmpeg_segment(
    file: str, sample_rate: int, start_s: float, duration_s: float
) -> np.ndarray:
    file = _clean_audio_path(file)
    command = [
        _ffmpeg_path(),
        "-nostdin",
        "-threads",
        "0",
        "-ss",
        f"{max(0.0, start_s):.9f}",
        "-i",
        file,
        "-t",
        f"{max(0.0, duration_s):.9f}",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def iter_audio_ffmpeg(file: str, sample_rate: int, block_seconds: float):
    file = _clean_audio_path(file)
    block_samples = max(1, int(round(sample_rate * block_seconds)))
    block_bytes = block_samples * np.dtype(np.float32).itemsize
    command = [
        _ffmpeg_path(),
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=block_bytes,
    )
    try:
        pending = bytearray()
        while True:
            while len(pending) < block_bytes:
                data = process.stdout.read(block_bytes - len(pending))
                if not data:
                    break
                pending.extend(data)
            if not pending:
                break
            usable = len(pending) - (len(pending) % np.dtype(np.float32).itemsize)
            if usable:
                yield np.frombuffer(pending[:usable], dtype=np.float32).copy()
            pending.clear()
            if usable < block_bytes:
                break
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()


class PreProcess:
    def __init__(self, sr: int, exp_dir: str, dataset_format: str = "wav"):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.b_high, self.a_high = signal.butter(
            N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=self.sr
        )
        self.exp_dir = exp_dir
        self.device = "cpu"
        self.dataset_format = normalize_dataset_format(dataset_format)
        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def _normalize_audio(self, audio: np.ndarray):
        tmp_max = np.abs(audio).max()
        if tmp_max > 2.5:
            return None
        return (audio / tmp_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio

    @staticmethod
    def _post_normalization_gain(voice_peak: float):
        if not np.isfinite(voice_peak) or voice_peak <= 0:
            return 1.0
        return min(MAX_AMPLITUDE / voice_peak, POST_NORMALIZATION_MAX_GAIN)

    def _detect_post_normalization_gain(self, audio: np.ndarray):
        voiced_chunks = self.slicer.slice(audio)
        if not voiced_chunks:
            return 1.0
        voice_peak = max(float(np.max(np.abs(chunk))) for chunk in voiced_chunks)
        return self._post_normalization_gain(voice_peak)

    def _peak_normalize_audio(self, audio: np.ndarray, source_gain: float):
        if audio.size == 0:
            return audio
        peak = np.abs(audio).max()
        if not np.isfinite(peak) or peak > 2.5:
            return None
        if peak == 0:
            return audio
        gain = min(source_gain, MAX_AMPLITUDE / peak)
        return audio * gain

    def process_audio_segment(
        self,
        normalized_audio: np.ndarray,
        sid: int,
        idx0: int,
        idx1: int,
        normalization_mode: str,
        normalization_gain: float = 1.0,
    ):
        if normalized_audio is None:
            print(f"{sid}-{idx0}-{idx1}-filtered")
            return
        if normalization_mode == "post":
            normalized_audio = self._peak_normalize_audio(
                normalized_audio, normalization_gain
            )
        write_training_audio(
            self.gt_wavs_dir,
            f"{sid}_{idx0}_{idx1}",
            self.sr,
            normalized_audio,
            self.dataset_format,
        )
        audio_16k = librosa.resample(
            normalized_audio,
            orig_sr=self.sr,
            target_sr=SAMPLE_RATE_16K,
        )
        write_training_audio(
            self.wavs16k_dir,
            f"{sid}_{idx0}_{idx1}",
            SAMPLE_RATE_16K,
            audio_16k,
            self.dataset_format,
        )

    def simple_cut(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
        normalization_gain: float = 1.0,
    ):
        chunk_length = int(self.sr * chunk_len)
        overlap_length = int(self.sr * overlap_len)
        i = 0
        while i < len(audio):
            chunk = audio[i : i + chunk_length]
            if normalization_mode == "post":
                chunk = self._peak_normalize_audio(chunk, normalization_gain)
            if len(chunk) == chunk_length:
                # full SR for training
                slice_stem = f"{sid}_{idx0}_{i // (chunk_length - overlap_length)}"
                write_training_audio(
                    self.gt_wavs_dir,
                    slice_stem,
                    self.sr,
                    chunk,
                    self.dataset_format,
                )
                # 16KHz for feature extraction
                chunk_16k = librosa.resample(
                    chunk, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K
                )
                write_training_audio(
                    self.wavs16k_dir,
                    slice_stem,
                    SAMPLE_RATE_16K,
                    chunk_16k,
                    self.dataset_format,
                )
            i += chunk_length - overlap_length

    def process_simple_audio(
        self,
        paths: list[str],
        idx0: int,
        sid: int,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
    ):
        audio_parts = [load_audio_ffmpeg(path, self.sr) for path in paths]
        audio_length = sum(len(part) for part in audio_parts) / self.sr
        audio = (
            audio_parts[0]
            if len(audio_parts) == 1
            else np.concatenate(audio_parts)
        )
        audio = truncate_silence(audio, self.sr)
        audio = self._prepare_audio(
            audio,
            process_effects,
            noise_reduction,
            reduction_strength,
            normalization_mode,
        )
        normalization_gain = 1.0
        if normalization_mode == "post" and audio is not None:
            normalization_gain = self._detect_post_normalization_gain(audio)
        self.simple_cut(
            audio,
            sid,
            idx0,
            chunk_len,
            overlap_len,
            normalization_mode,
            normalization_gain,
        )
        return audio_length

    def _prepare_audio(
        self,
        audio: np.ndarray,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        normalization_mode: str,
    ):
        if process_effects:
            audio = signal.lfilter(self.b_high, self.a_high, audio)
        if normalization_mode == "pre":
            audio = self._normalize_audio(audio)
        if noise_reduction and audio is not None and audio.size:
            audio = nr.reduce_noise(
                y=audio, sr=self.sr, prop_decrease=reduction_strength
            )
        return audio

    def _detect_automatic_intervals(self, path: str):
        context_samples = int(round(SAMPLE_RATE_16K * AUTOMATIC_VAD_CONTEXT_SECONDS))
        intervals = []
        previous = None
        previous_start = 0
        past_context = np.empty(0, dtype=np.float32)
        total_samples = 0
        voice_peak = 0.0

        for current in iter_audio_ffmpeg(
            path, SAMPLE_RATE_16K, AUTOMATIC_VAD_BLOCK_SECONDS
        ):
            current = np.asarray(current, dtype=np.float32)
            if previous is None:
                previous = current
                total_samples += len(current)
                continue

            future_context = current[:context_samples]
            analysis = np.concatenate((past_context, previous, future_context))
            analysis_start = previous_start - len(past_context)
            core_start = len(past_context)
            core_end = core_start + len(previous)

            for start_s, end_s in self.slicer.detect_voice_intervals_16k(analysis):
                start_sample = int(round(start_s * SAMPLE_RATE_16K))
                end_sample = int(round(end_s * SAMPLE_RATE_16K))
                if end_sample <= core_start or start_sample >= core_end:
                    continue
                voice_start = max(start_sample, core_start)
                voice_end = min(end_sample, core_end)
                if voice_end > voice_start:
                    voice_peak = max(
                        voice_peak,
                        float(np.max(np.abs(analysis[voice_start:voice_end]))),
                    )
                intervals.append(
                    (
                        (analysis_start + start_sample) / SAMPLE_RATE_16K,
                        (analysis_start + end_sample) / SAMPLE_RATE_16K,
                    )
                )

            past_context = previous[-context_samples:].copy()
            previous_start += len(previous)
            previous = current
            total_samples += len(current)

        if previous is not None:
            analysis = (
                np.concatenate((past_context, previous))
                if past_context.size
                else previous
            )
            analysis_start = previous_start - len(past_context)
            core_start = len(past_context)
            core_end = core_start + len(previous)
            for start_s, end_s in self.slicer.detect_voice_intervals_16k(analysis):
                start_sample = int(round(start_s * SAMPLE_RATE_16K))
                end_sample = int(round(end_s * SAMPLE_RATE_16K))
                if end_sample <= core_start or start_sample >= core_end:
                    continue
                voice_start = max(start_sample, core_start)
                voice_end = min(end_sample, core_end)
                if voice_end > voice_start:
                    voice_peak = max(
                        voice_peak,
                        float(np.max(np.abs(analysis[voice_start:voice_end]))),
                    )
                intervals.append(
                    (
                        (analysis_start + start_sample) / SAMPLE_RATE_16K,
                        (analysis_start + end_sample) / SAMPLE_RATE_16K,
                    )
                )

        duration_s = total_samples / SAMPLE_RATE_16K
        return (
            self.slicer.merge_voice_intervals(intervals, duration_s),
            duration_s,
            voice_peak,
        )

    @staticmethod
    def _automatic_clip_ranges(intervals):
        step = PERCENTAGE - OVERLAP
        ranges = []
        for interval_start, interval_end in intervals:
            start = interval_start
            while start < interval_end:
                remaining = interval_end - start
                if remaining > PERCENTAGE + OVERLAP:
                    end = start + PERCENTAGE
                    ranges.append((start, end))
                    start += step
                else:
                    ranges.append((start, interval_end))
                    break
        return ranges

    @staticmethod
    def _group_clip_ranges(ranges):
        if not ranges:
            return []
        groups = []
        current = [ranges[0]]
        group_start = ranges[0][0]
        for clip_range in ranges[1:]:
            if clip_range[1] - group_start <= AUTOMATIC_DECODE_BLOCK_SECONDS:
                current.append(clip_range)
            else:
                groups.append(current)
                current = [clip_range]
                group_start = clip_range[0]
        groups.append(current)
        return groups

    def _process_automatic(
        self,
        path: str,
        idx0: int,
        sid: int,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        normalization_mode: str,
    ):
        intervals, duration_s, voice_peak = self._detect_automatic_intervals(path)
        if not intervals:
            print(f"No speech or singing detected in: {path}")
            return duration_s

        ranges = self._automatic_clip_ranges(intervals)
        normalization_gain = self._post_normalization_gain(voice_peak)
        idx1 = 0
        for group in self._group_clip_ranges(ranges):
            batch_start = group[0][0]
            batch_end = group[-1][1]
            decode_start = max(0.0, batch_start - AUTOMATIC_PROCESS_CONTEXT_SECONDS)
            decode_end = min(
                duration_s, batch_end + AUTOMATIC_PROCESS_CONTEXT_SECONDS
            )
            audio = load_audio_ffmpeg_segment(
                path, self.sr, decode_start, decode_end - decode_start
            )
            audio = self._prepare_audio(
                audio,
                process_effects,
                noise_reduction,
                reduction_strength,
                normalization_mode,
            )
            if audio is None:
                for _ in group:
                    print(f"{sid}-{idx0}-{idx1}-filtered")
                    idx1 += 1
                continue

            for clip_start, clip_end in group:
                local_start = max(
                    0, int(round((clip_start - decode_start) * self.sr))
                )
                local_end = min(
                    len(audio), int(round((clip_end - decode_start) * self.sr))
                )
                if local_end > local_start:
                    self.process_audio_segment(
                        audio[local_start:local_end],
                        sid,
                        idx0,
                        idx1,
                        normalization_mode,
                        normalization_gain,
                    )
                idx1 += 1
        return duration_s

    def process_audio(
        self,
        path: str,
        idx0: int,
        sid: int,
        cut_preprocess: str,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
    ):
        audio_length = 0
        try:
            if cut_preprocess == "Automatic":
                return self._process_automatic(
                    path,
                    idx0,
                    sid,
                    process_effects,
                    noise_reduction,
                    reduction_strength,
                    normalization_mode,
                )

            audio = load_audio_ffmpeg(path, self.sr)
            audio_length = librosa.get_duration(y=audio, sr=self.sr)
            audio = self._prepare_audio(
                audio,
                process_effects,
                noise_reduction,
                reduction_strength,
                normalization_mode,
            )
            normalization_gain = 1.0
            if normalization_mode == "post" and audio is not None:
                normalization_gain = self._detect_post_normalization_gain(audio)
            if cut_preprocess == "Skip":
                # no cutting
                self.process_audio_segment(
                    audio,
                    sid,
                    idx0,
                    0,
                    normalization_mode,
                    normalization_gain,
                )
            elif cut_preprocess == "Simple":
                # simple
                self.simple_cut(
                    audio,
                    sid,
                    idx0,
                    chunk_len,
                    overlap_len,
                    normalization_mode,
                    normalization_gain,
                )
        except Exception as error:
            print(f"Error processing audio: {error}")
            if cut_preprocess == "Automatic" or self.dataset_format == "flac":
                raise
        return audio_length


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def save_dataset_duration(file_path, dataset_duration, dataset_format="wav"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    formatted_duration = format_duration(dataset_duration)
    new_data = {
        "total_dataset_duration": formatted_duration,
        "total_seconds": dataset_duration,
        "dataset_format": normalize_dataset_format(dataset_format),
    }
    data.update(new_data)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def process_audio_wrapper(args):
    (
        pp,
        file,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
    ) = args
    file_path, idx0, sid = file
    return pp.process_audio(
        file_path,
        idx0,
        sid,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
    )


def process_simple_audio_wrapper(args):
    (
        pp,
        paths,
        idx0,
        sid,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
    ) = args
    return pp.process_simple_audio(
        paths,
        idx0,
        sid,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
    )


def preprocess_training_set(
    input_root: str,
    sr: int,
    num_processes: int,
    exp_dir: str,
    cut_preprocess: str,
    process_effects: bool,
    noise_reduction: bool,
    reduction_strength: float,
    chunk_len: float,
    overlap_len: float,
    normalization_mode: str,
    dataset_format: str = "wav",
):
    if not os.path.exists(input_root):
        print(f"The dataset path does not exist: '{input_root}'.")
        sys.exit(1)

    if not os.path.isdir(input_root):
        print(f"The dataset path is not a directory: '{input_root}'.")
        sys.exit(1)
    start_time = time.time()
    dataset_format = normalize_dataset_format(dataset_format)
    print(f"Starting preprocess with {num_processes} processes...")

    files = []
    idx = 0

    for root, directories, filenames in os.walk(input_root):
        directories.sort()
        try:
            sid = 0 if root == input_root else int(os.path.basename(root))
            for f in sorted(filenames):
                if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                    files.append((os.path.join(root, f), idx, sid))
                    idx += 1
        except ValueError:
            print(
                f'Speaker ID folder is expected to be integer, got "{os.path.basename(root)}" instead.'
            )

    # print(f"Number of files: {len(files)}")
    if len(files) == 0:
        print(
            f"No audio files found in the dataset path: '{input_root}'. Please check that the path is correct and contains valid audio files."
        )
        sys.exit(1)

    if cut_preprocess == "Simple":
        clear_simple_preprocess_artifacts(exp_dir)
    elif dataset_format == "flac":
        clear_flac_preprocess_artifacts(exp_dir)
    pp = PreProcess(sr, exp_dir, dataset_format)

    audio_length = []
    if cut_preprocess == "Simple":
        files_by_speaker = {}
        for file_path, idx0, sid in files:
            files_by_speaker.setdefault(sid, []).append((file_path, idx0))
        work_items = [
            (
                pp,
                [file_path for file_path, _ in speaker_files],
                speaker_files[0][1],
                sid,
                process_effects,
                noise_reduction,
                reduction_strength,
                chunk_len,
                overlap_len,
                normalization_mode,
            )
            for sid, speaker_files in sorted(files_by_speaker.items())
        ]
        worker = process_simple_audio_wrapper
    else:
        work_items = [
            (
                pp,
                file,
                cut_preprocess,
                process_effects,
                noise_reduction,
                reduction_strength,
                chunk_len,
                overlap_len,
                normalization_mode,
            )
            for file in files
        ]
        worker = process_audio_wrapper

    with tqdm(total=len(work_items)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_processes
        ) as executor:
            futures = [
                executor.submit(worker, work_item) for work_item in work_items
            ]
            for future in concurrent.futures.as_completed(futures):
                audio_length.append(future.result())
                pbar.update(1)

    audio_length = sum(audio_length)
    save_dataset_duration(
        os.path.join(exp_dir, "model_info.json"),
        dataset_duration=audio_length,
        dataset_format=dataset_format,
    )
    elapsed_time = time.time() - start_time
    print(
        f"Preprocess completed in {elapsed_time:.2f} seconds on {format_duration(audio_length)} seconds of audio."
    )


if __name__ == "__main__":
    experiment_directory = str(sys.argv[1])
    input_root = str(sys.argv[2])
    sample_rate = int(sys.argv[3])
    num_processes = sys.argv[4]
    if num_processes.lower() == "none":
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = int(num_processes)
    cut_preprocess = str(sys.argv[5])
    process_effects = strtobool(sys.argv[6])
    noise_reduction = strtobool(sys.argv[7])
    reduction_strength = float(sys.argv[8])
    chunk_len = float(sys.argv[9])
    overlap_len = float(sys.argv[10])
    normalization_mode = str(sys.argv[11])
    dataset_format = str(sys.argv[12]) if len(sys.argv) > 12 else "WAV"
    preprocess_training_set(
        input_root,
        sample_rate,
        num_processes,
        experiment_directory,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
        dataset_format,
    )
