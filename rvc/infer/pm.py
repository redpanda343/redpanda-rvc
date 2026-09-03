import atexit
import os
import pickle
import subprocess
import sys
import threading

import numpy as np
import parselmouth


LOGICAL_CPU_COUNT = os.cpu_count() or 1
PM_INFERENCE_WORKERS = max(1, LOGICAL_CPU_COUNT // 2)
PM_CONTEXT_SECONDS = 0.5


def _extract_pitch(
    audio,
    sample_rate,
    start_time,
    time_step,
    voicing_threshold,
    pitch_floor,
    pitch_ceiling,
):
    pitch = parselmouth.Sound(
        audio,
        sample_rate,
        start_time=start_time,
    ).to_pitch_ac(
        time_step=time_step,
        voicing_threshold=voicing_threshold,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
    )
    return pitch.xs(), pitch.selected_array["frequency"]


def _single_process_pm(
    audio,
    sample_rate,
    p_len,
    time_step,
    voicing_threshold,
    pitch_floor,
    pitch_ceiling,
):
    _, f0 = _extract_pitch(
        audio,
        sample_rate,
        0.0,
        time_step,
        voicing_threshold,
        pitch_floor,
        pitch_ceiling,
    )
    pad_size = (p_len - len(f0) + 1) // 2
    if pad_size > 0 or p_len - len(f0) - pad_size > 0:
        f0 = np.pad(
            f0,
            [[pad_size, p_len - len(f0) - pad_size]],
            mode="constant",
        )
    return f0


class _PmWorker:
    def __init__(self):
        self.process = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def is_running(self):
        return self.process.poll() is None

    def submit(self, payload):
        pickle.dump(payload, self.process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
        self.process.stdin.flush()

    def receive(self):
        response = pickle.load(self.process.stdout)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(response["error"])
        return response

    def close(self):
        if self.is_running():
            try:
                pickle.dump(None, self.process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
                self.process.stdin.flush()
                self.process.wait(timeout=2)
            except Exception:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass


class _PmWorkerPool:
    def __init__(self, worker_count):
        self.worker_count = worker_count
        self.workers = []
        self.lock = threading.Lock()

    def _ensure_workers(self):
        if len(self.workers) == self.worker_count and all(
            worker.is_running() for worker in self.workers
        ):
            return
        self._close_workers()
        self.workers = [_PmWorker() for _ in range(self.worker_count)]

    def _close_workers(self):
        for worker in self.workers:
            worker.close()
        self.workers = []

    def close(self):
        with self.lock:
            self._close_workers()

    def extract(self, tasks):
        with self.lock:
            self._ensure_workers()
            active_workers = self.workers[: len(tasks)]
            for worker, task in zip(active_workers, tasks):
                worker.submit(task)
            return [worker.receive() for worker in active_workers]


_worker_pool = _PmWorkerPool(PM_INFERENCE_WORKERS)
atexit.register(_worker_pool.close)


def extract_pm(
    audio,
    sample_rate,
    p_len,
    time_step,
    voicing_threshold,
    pitch_floor,
    pitch_ceiling,
):
    audio = np.ascontiguousarray(audio, dtype=np.float64)
    if p_len < PM_INFERENCE_WORKERS:
        return _single_process_pm(
            audio,
            sample_rate,
            p_len,
            time_step,
            voicing_threshold,
            pitch_floor,
            pitch_ceiling,
        )

    hop_samples = max(1, round(time_step * sample_rate))
    context_samples = max(
        round(PM_CONTEXT_SECONDS * sample_rate),
        round(3 * sample_rate / pitch_floor),
    )
    frame_boundaries = np.linspace(
        0,
        p_len,
        PM_INFERENCE_WORKERS + 1,
        dtype=int,
    )

    tasks = []
    frame_ranges = []
    for worker_index in range(PM_INFERENCE_WORKERS):
        frame_start = int(frame_boundaries[worker_index])
        frame_end = int(frame_boundaries[worker_index + 1])
        sample_start = max(0, frame_start * hop_samples - context_samples)
        sample_end = min(
            len(audio),
            frame_end * hop_samples + context_samples,
        )
        tasks.append(
            (
                audio[sample_start:sample_end],
                sample_rate,
                sample_start / sample_rate,
                time_step,
                voicing_threshold,
                pitch_floor,
                pitch_ceiling,
            )
        )
        frame_ranges.append((frame_start, frame_end))

    try:
        worker_results = _worker_pool.extract(tasks)
        f0 = np.zeros(p_len, dtype=np.float64)
        filled = np.zeros(p_len, dtype=bool)
        for (times, frequencies), (frame_start, frame_end) in zip(
            worker_results, frame_ranges
        ):
            frame_indices = np.floor(times / time_step + 0.5).astype(int)
            keep = (
                (frame_indices >= frame_start)
                & (frame_indices < frame_end)
                & (frame_indices >= 0)
                & (frame_indices < p_len)
            )
            frame_indices = frame_indices[keep]
            f0[frame_indices] = frequencies[keep]
            filled[frame_indices] = True

        populated = np.flatnonzero(filled)
        if populated.size and filled[populated[0] : populated[-1] + 1].all():
            return f0
    except Exception as error:
        print(f"Parallel PM extraction failed; using one core instead: {error}")

    return _single_process_pm(
        audio,
        sample_rate,
        p_len,
        time_step,
        voicing_threshold,
        pitch_floor,
        pitch_ceiling,
    )


def _worker_main():
    input_pipe = sys.stdin.buffer
    output_pipe = sys.stdout.buffer
    while True:
        try:
            payload = pickle.load(input_pipe)
        except EOFError:
            break
        if payload is None:
            break
        try:
            response = _extract_pitch(*payload)
        except Exception as error:
            response = {"error": str(error)}
        pickle.dump(response, output_pipe, protocol=pickle.HIGHEST_PROTOCOL)
        output_pipe.flush()


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker_main()
