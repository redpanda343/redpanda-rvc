import gc
import importlib.util
import os
import random
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def deterministic_validation_scope(seed, cuda_devices=None):
    seed = int(seed) % (2**63 - 1)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    previous_settings = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    devices = list(cuda_devices or [])
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            try:
                yield
            finally:
                (
                    deterministic_algorithms,
                    deterministic_warn_only,
                    cudnn_benchmark,
                    cudnn_deterministic,
                    matmul_allow_tf32,
                    cudnn_allow_tf32,
                    matmul_precision,
                ) = previous_settings
                torch.use_deterministic_algorithms(
                    deterministic_algorithms,
                    warn_only=deterministic_warn_only,
                )
                torch.backends.cudnn.benchmark = cudnn_benchmark
                torch.backends.cudnn.deterministic = cudnn_deterministic
                torch.backends.cuda.matmul.allow_tf32 = matmul_allow_tf32
                torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
                torch.set_float32_matmul_precision(matmul_precision)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class UTMOSv2Validator:
    def __init__(self, model_paths, seed, device):
        missing_paths = [path for path in model_paths if not os.path.isfile(path)]
        if missing_paths:
            raise FileNotFoundError(
                f"UTMOSv2 checkpoint not found: {missing_paths[0]}"
            )
        if importlib.util.find_spec("utmosv2") is None:
            raise ModuleNotFoundError("UTMOSv2 is not installed")
        self.model_paths = list(model_paths)
        self.seed = int(seed) % (2**63 - 1)
        self.device = torch.device(device)
        self.repetitions = 5

    def _load_model(self, model_path, fold):
        import utmosv2

        with deterministic_validation_scope(self.seed):
            model = utmosv2.create_model(
                pretrained=True,
                fold=fold,
                checkpoint_path=model_path,
                device="cpu",
            )
            model.eval().float().to("cpu")
        floating_dtypes = {
            parameter.dtype
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
        if floating_dtypes != {torch.float32}:
            raise RuntimeError(
                f"UTMOSv2 must use FP32 parameters, got {sorted(map(str, floating_dtypes))}"
            )
        return model

    @staticmethod
    def _prepare_clips(generated, lengths, speaker_ids, sample_rate):
        generated = generated.detach().to(device="cpu", dtype=torch.float32)
        length_values = lengths.detach().cpu().tolist()
        speaker_values = speaker_ids.detach().cpu().tolist()
        minimum_length = int(2 * sample_rate)
        maximum_length = int(3 * sample_rate)
        clips = []
        speakers = []
        for index, requested_length in enumerate(length_values):
            clip = generated[index]
            while clip.dim() > 1:
                clip = clip.mean(dim=0)
            valid_length = min(max(0, int(requested_length)), clip.numel())
            clip = clip[:valid_length].contiguous()
            if clip.numel() < minimum_length or not torch.isfinite(clip).all():
                continue
            if clip.abs().mean().item() <= 1e-6:
                continue
            if clip.numel() > maximum_length:
                start = (clip.numel() - maximum_length) // 2
                clip = clip[start : start + maximum_length].contiguous()
            clips.append(clip.numpy())
            speakers.append(int(speaker_values[index]))
        if not clips:
            raise ValueError("No valid generated audio is available for UTMOSv2")
        return clips, speakers

    @staticmethod
    def _resample_clip(model, clip, sample_rate):
        import torchaudio

        audio = torch.from_numpy(clip)
        target_sample_rate = int(model._cfg.sr)
        if int(sample_rate) != target_sample_rate:
            audio = torchaudio.functional.resample(
                audio,
                int(sample_rate),
                target_sample_rate,
            )
        return audio.numpy().astype(np.float32, copy=False)

    @staticmethod
    def _score_one(model, clip, device):
        from utmosv2.dataset._schema import InMemoryData
        from utmosv2.utils import get_dataset

        data = InMemoryData(
            data=clip,
            dataset_name="sarulab",
        )
        initial_state = getattr(model._cfg.dataset, "remove_silent_section", None)
        model._cfg.dataset.remove_silent_section = False
        try:
            sample = get_dataset(model._cfg, data, model._cfg.phase)[0]
        finally:
            model._cfg.dataset.remove_silent_section = initial_state
        inputs = [value.unsqueeze(0).to(device) for value in sample[:-1]]
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=False):
                prediction = model(*inputs).reshape(-1)[0]
        result = float(prediction.float().cpu().item())
        del prediction, inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    def _score_clips(self, model, clips, sample_rate, device):
        cuda_devices = [device.index or 0] if device.type == "cuda" else []
        model.eval().float().to(device)
        predictions = []
        for clip in clips:
            prepared_clip = self._resample_clip(model, clip, sample_rate)
            with deterministic_validation_scope(
                self.seed, cuda_devices=cuda_devices
            ):
                repeated = [
                    self._score_one(model, prepared_clip, device)
                    for _ in range(self.repetitions)
                ]
            predictions.append(sum(repeated) / len(repeated))
        return predictions

    def _score_fold(self, model, clips, sample_rate):
        target_device = self.device
        try:
            return self._score_clips(model, clips, sample_rate, target_device)
        except RuntimeError as error:
            if target_device.type != "cuda" or "out of memory" not in str(error).lower():
                raise
            model.to("cpu")
            torch.cuda.empty_cache()
            print("UTMOSv2 GPU validation ran out of VRAM; retrying safely on CPU.")
            return self._score_clips(
                model, clips, sample_rate, torch.device("cpu")
            )

    def score_batch(self, generated, lengths, speaker_ids, sample_rate):
        clips, speakers = self._prepare_clips(
            generated, lengths, speaker_ids, sample_rate
        )
        fold_predictions = []
        for fold, model_path in enumerate(self.model_paths):
            print(f"UTMOSv2 validating fold {fold + 1}/{len(self.model_paths)}.")
            model = self._load_model(model_path, fold)
            try:
                fold_predictions.append(
                    self._score_fold(model, clips, sample_rate)
                )
            finally:
                model.to("cpu")
                del model
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        predictions = np.asarray(fold_predictions, dtype=np.float32).mean(axis=0)
        predictions = predictions.reshape(-1)
        if predictions.size != len(speakers) or not np.isfinite(predictions).all():
            raise RuntimeError("UTMOSv2 returned invalid predictions")
        by_speaker = defaultdict(list)
        for speaker_id, prediction in zip(speakers, predictions):
            by_speaker[speaker_id].append(float(prediction))
        speaker_means = [sum(values) / len(values) for values in by_speaker.values()]
        return sum(speaker_means) / len(speaker_means)
