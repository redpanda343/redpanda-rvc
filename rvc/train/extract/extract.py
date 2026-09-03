import concurrent.futures
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import parselmouth
import torch
import tqdm

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

# Zluda hijack
import rvc.lib.zluda
from rvc.configs.config import Config
from rvc.lib.predictors.f0 import RMVPE
from rvc.lib.utils import load_audio_16k, load_embedding
from rvc.train.extract.preparing_files import generate_config, generate_filelist

# Load config
config = Config()
mp.set_start_method("spawn", force=True)


class FeatureInput:
    def __init__(self, f0_method="rmvpe", device="cpu"):
        self.hop_size = 160  # default
        self.sample_rate = 16000  # default
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device
        if f0_method == "rmvpe":
            self.model = RMVPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        elif f0_method != "pm":
            raise ValueError(f"Unsupported F0 method: {f0_method}")
        self.f0_method = f0_method

    def compute_f0(self, x, p_len=None):
        if self.f0_method == "pm":
            if p_len is None:
                p_len = x.shape[0] // self.hop_size
            f0 = self.get_pm(x, p_len)
        elif self.f0_method == "rmvpe":
            f0 = self.model.get_f0(x, filter_radius=0.03)
        return np.asarray(f0).copy()

    def get_pm(self, x, p_len):
        f0 = (
            parselmouth.Sound(x, self.sample_rate)
            .to_pitch_ac(
                time_step=self.hop_size / self.sample_rate,
                voicing_threshold=0.6,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max,
            )
            .selected_array["frequency"]
        )
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(
                f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
            )
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def process_file(self, file_info):
        inp_path, opt_path_coarse, opt_path_full, _ = file_info
        if os.path.exists(opt_path_coarse) and os.path.exists(opt_path_full):
            return

        try:
            np_arr = load_audio_16k(inp_path)
            feature_pit = self.compute_f0(np_arr)
            if feature_pit is None:
                return
            np.save(opt_path_full, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path_coarse, coarse_pit, allow_pickle=False)
        except Exception as error:
            print(
                f"An error occurred extracting file {inp_path} on {self.device}: {error}"
            )


def process_files(files, f0_method, device):
    fe = FeatureInput(f0_method=f0_method, device=device)
    with tqdm.tqdm(total=len(files), leave=True) as pbar:
        for file_info in files:
            fe.process_file(file_info)
            pbar.update(1)


def run_pitch_extraction(files, devices, f0_method, threads):
    threads = max(1, int(threads))
    if f0_method == "pm":
        worker_count = min(threads, len(files))
        worker_devices = ["cpu"] * worker_count
        print(
            f"Starting pitch extraction with {worker_count} CPU worker(s) using pm..."
        )
    else:
        worker_count = min(len(devices), len(files))
        worker_devices = devices[:worker_count]
        devices_str = ", ".join(worker_devices)
        print(f"Starting pitch extraction on {devices_str} using {f0_method}...")
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        tasks = [
            executor.submit(
                process_files,
                files[i::worker_count],
                f0_method,
                worker_devices[i],
            )
            for i in range(worker_count)
        ]
        concurrent.futures.wait(tasks)

    print(f"Pitch extraction completed in {time.time() - start_time:.2f} seconds.")


def process_file_embedding(
    files, embedder_model, embedder_model_custom, device_num, device, n_threads
):
    model = load_embedding(embedder_model, embedder_model_custom).to(device).float()
    model.eval()
    n_threads = max(1, n_threads)

    def worker(file_info):
        wav_file_path, _, _, out_file_path = file_info
        if os.path.exists(out_file_path):
            return
        feats = torch.from_numpy(load_audio_16k(wav_file_path)).float()
        if getattr(model, "audio_requires_normalization", False):
            feats = torch.nn.functional.layer_norm(feats, feats.shape)
        feats = feats.to(device)
        feats = feats.view(1, -1)
        with torch.no_grad():
            result = model(feats)["last_hidden_state"]
        feats_out = result.squeeze(0).float().cpu().numpy()
        if not np.isnan(feats_out).any():
            np.save(out_file_path, feats_out, allow_pickle=False)
        else:
            print(f"{wav_file_path} produced NaN values; skipping.")

    with tqdm.tqdm(total=len(files), leave=True, position=device_num) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker, f) for f in files]
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


def run_embedding_extraction(
    files, devices, embedder_model, embedder_model_custom, threads
):
    devices_str = ", ".join(devices)
    print(
        f"Starting embedding extraction with {threads} cores on {devices_str}..."
    )
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        tasks = [
            executor.submit(
                process_file_embedding,
                files[i :: len(devices)],
                embedder_model,
                embedder_model_custom,
                i,
                devices[i],
                threads // len(devices),
            )
            for i in range(len(devices))
        ]
        concurrent.futures.wait(tasks)

    print(f"Embedding extraction completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    exp_dir = sys.argv[1]
    f0_method = sys.argv[2]
    num_processes = int(sys.argv[3])
    gpus = sys.argv[4]
    sample_rate = sys.argv[5]
    embedder_model = sys.argv[6]
    embedder_model_custom = sys.argv[7] if len(sys.argv) > 7 else None
    include_mutes = int(sys.argv[8]) if len(sys.argv) > 8 else 2

    wav_path = os.path.join(exp_dir, "sliced_audios_16k")

    if not os.path.exists(wav_path):
        print(
            f"Folder for feature extraction not found at {wav_path}. Did you run the preprocessing step?"
        )
        sys.exit(1)

    os.makedirs(os.path.join(exp_dir, "f0"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "f0_voiced"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "extracted"), exist_ok=True)

    chosen_embedder_model = (
        embedder_model_custom if embedder_model == "custom" else embedder_model
    )
    file_path = os.path.join(exp_dir, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    dataset_format = str(data.get("dataset_format", "wav")).strip().lower()
    if dataset_format not in {"wav", "flac"}:
        dataset_format = "wav"
    data["embedder_model"] = chosen_embedder_model
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    files = []
    audio_extension = f".{dataset_format}"
    for file in sorted(glob.glob(os.path.join(wav_path, f"*{audio_extension}"))):
        file_name = os.path.basename(file)
        feature_name = (
            file_name.replace("wav", "npy")
            if dataset_format == "wav"
            else f"{file_name}.npy"
        )
        file_info = [
            file,
            os.path.join(exp_dir, "f0", file_name + ".npy"),
            os.path.join(exp_dir, "f0_voiced", file_name + ".npy"),
            os.path.join(exp_dir, "extracted", feature_name),
        ]
        files.append(file_info)

    if not files:
        print(
            f"Sliced audios not found at {wav_path}. Did you run the preprocessing step?"
        )
        sys.exit(1)

    devices = ["cpu"] if gpus == "-" else [f"cuda:{idx}" for idx in gpus.split("-")]

    run_pitch_extraction(files, devices, f0_method, num_processes)

    run_embedding_extraction(
        files, devices, embedder_model, embedder_model_custom, num_processes
    )

    generate_config(sample_rate, exp_dir)
    generate_filelist(exp_dir, sample_rate, include_mutes)
