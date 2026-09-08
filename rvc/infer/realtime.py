import os
import threading
import time

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from rvc.infer.infer import VoiceConverter


SUPPORTED_VOCODERS = {"HiFi-GAN", "RefineGAN"}


class RealTimeRVC:
    def __init__(
        self,
        model_path,
        index_path="",
        index_rate=0.0,
        pitch=0,
        speaker_id=0,
    ):
        self.converter = VoiceConverter()
        self.converter.get_vc(model_path, speaker_id)
        if self.converter.cpt is None:
            raise FileNotFoundError(f"Voice model not found: {model_path}")
        self.vocoder = self.converter.vocoder
        if self.vocoder not in SUPPORTED_VOCODERS:
            supported = ", ".join(sorted(SUPPORTED_VOCODERS))
            raise ValueError(
                f"Real-time mode supports {supported}; this model uses {self.vocoder}."
            )
        if not self.converter.use_f0:
            raise ValueError("Real-time mode currently requires a pitch-guided model.")
        if speaker_id < 0 or speaker_id >= self.converter.n_spk:
            raise ValueError(
                f"Speaker ID must be between 0 and {self.converter.n_spk - 1}."
            )
        self.device = self.converter.config.device
        self.model = self.converter.net_g.to(
            device=self.device, dtype=torch.float32
        )
        self.pipeline = self.converter.vc
        self.sample_rate = self.converter.tgt_sr
        self.version = self.converter.version
        self.speaker_id = speaker_id
        self.pitch = pitch
        self.index_rate = float(index_rate)
        self.index = None
        self.big_npy = None
        self.lock = threading.RLock()
        self.infer_count = 0
        self.cache_pitch = torch.zeros(4096, device=self.device, dtype=torch.long)
        self.cache_pitchf = torch.zeros(
            4096, device=self.device, dtype=torch.float32
        )
        expected_embedder = self.converter.cpt.get("embedder_model") or "contentvec"
        self.embedder_name = expected_embedder
        self.converter.load_hubert(expected_embedder)
        self.embedder = self.converter.hubert_model.to(
            device=self.device, dtype=torch.float32
        )
        self.expected_feature_dim = int(self.model.enc_p.emb_phone.in_features)
        if index_path:
            self._load_index(index_path)
        if self.index_rate > 0 and self.index is None:
            raise ValueError("Select a valid feature index or set Index Rate to 0.")

    @property
    def speaker_count(self):
        return self.converter.n_spk

    def _load_index(self, index_path):
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Feature index not found: {index_path}")
        index = faiss.read_index(index_path)
        if int(index.d) != self.expected_feature_dim:
            raise ValueError(
                f"The index has {index.d} channels, but this model expects "
                f"{self.expected_feature_dim}."
            )
        self.index = index
        self.big_npy = index.reconstruct_n(0, index.ntotal)

    def change_pitch(self, pitch):
        with self.lock:
            self.pitch = int(pitch)

    def change_index_rate(self, index_rate):
        rate = float(index_rate)
        if rate > 0 and self.index is None:
            raise ValueError("No feature index is loaded.")
        with self.lock:
            self.index_rate = rate

    def reset_caches(self):
        self.cache_pitch.zero_()
        self.cache_pitchf.zero_()

    def _extract_features(self, input_wav):
        source = input_wav.float().view(1, -1)
        if getattr(self.embedder, "audio_requires_normalization", False):
            source = F.layer_norm(source, source.shape)
        output = self.embedder(source)
        features = output["last_hidden_state"]
        if self.version == "v1":
            features = self.embedder.final_proj(features[0]).unsqueeze(0)
        if features.shape[-1] != self.expected_feature_dim:
            raise RuntimeError(
                f"{self.embedder_name} outputs {features.shape[-1]} channels, but "
                f"this model expects {self.expected_feature_dim}."
            )
        return torch.cat((features, features[:, -1:, :]), dim=1)

    def _apply_index(self, features, skip_head):
        if self.index is None or self.index_rate <= 0:
            return features
        start = skip_head // 2
        query = features[0, start:].detach().cpu().numpy().astype("float32")
        score, indices = self.index.search(query, k=8)
        if not (indices >= 0).all():
            raise RuntimeError("The selected index contains invalid neighbors.")
        score = np.maximum(score, 1e-6)
        weight = np.square(1.0 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        retrieved = np.sum(
            self.big_npy[indices] * np.expand_dims(weight, axis=2), axis=1
        )
        replacement = torch.from_numpy(retrieved).unsqueeze(0).to(self.device)
        features[0, start:] = (
            replacement * self.index_rate
            + features[0, start:] * (1.0 - self.index_rate)
        )
        return features

    def _update_pitch(self, input_wav, block_frame_16k, method):
        extractor_frame = block_frame_16k + 800
        if method == "rmvpe":
            extractor_frame = 5120 * ((extractor_frame - 1) // 5120 + 1) - 160
        source = input_wav[-extractor_frame:].detach().cpu().numpy()
        pitch, pitchf = self.pipeline.get_f0(
            source,
            source.shape[0] // 160,
            f0_method=method,
            pitch=self.pitch,
        )
        predictor = getattr(self.pipeline, f"model_{method}", None)
        current = predictor
        for _ in range(4):
            if isinstance(current, torch.nn.Module):
                current.float()
            mel_extractor = getattr(current, "mel_extractor", None)
            if isinstance(mel_extractor, torch.nn.Module):
                mel_extractor.float()
            current = getattr(current, "model", None)
            if current is None:
                break
        pitch = torch.as_tensor(pitch, device=self.device, dtype=torch.long).flatten()
        pitchf = torch.as_tensor(
            pitchf, device=self.device, dtype=torch.float32
        ).flatten()
        shift = max(1, block_frame_16k // 160)
        self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
        self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
        usable_pitch = pitch[3:-1] if pitch.numel() > 4 else pitch
        usable_pitchf = pitchf[3:-1] if pitchf.numel() > 4 else pitchf
        count = min(usable_pitch.numel(), self.cache_pitch.numel())
        if count:
            self.cache_pitch[-count:] = usable_pitch[-count:]
            self.cache_pitchf[-count:] = usable_pitchf[-count:]

    @torch.inference_mode()
    def infer(
        self,
        input_wav,
        block_frame_16k,
        skip_head,
        return_length,
        f0_method,
    ):
        started = time.perf_counter()
        with self.lock:
            features = self._extract_features(input_wav)
            features = self._apply_index(features, skip_head)
            p_len = min(input_wav.shape[0] // 160, features.shape[1] * 2)
            self._update_pitch(input_wav, block_frame_16k, f0_method)
            features = F.interpolate(
                features.permute(0, 2, 1), scale_factor=2
            ).permute(0, 2, 1)
            features = features[:, :p_len]
            lengths = torch.tensor([p_len], device=self.device, dtype=torch.long)
            speaker = torch.tensor(
                [self.speaker_id], device=self.device, dtype=torch.long
            )
            coarse = self.cache_pitch[None, -p_len:]
            continuous = self.cache_pitchf[None, -p_len:]
            audio = self.model.infer(
                features.float(),
                lengths,
                coarse,
                continuous,
                speaker,
                int(skip_head),
                int(return_length),
            )[0]
        self.infer_count += 1
        return audio.squeeze().float(), time.perf_counter() - started
