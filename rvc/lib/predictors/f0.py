import os
import torch

from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from torchfcpe import spawn_bundled_infer_model
import numpy as np


class RMVPE:
    def __init__(self, device, model_name="rmvpe.pt", sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = RMVPE0Predictor(
            os.path.join("rvc", "models", "predictors", model_name),
            device=self.device,
        )

    def get_f0(self, x, filter_radius=0.03, cuda_graph_manager=None):
        f0 = self.model.infer_from_audio(
            x,
            thred=filter_radius,
            cuda_graph_manager=cuda_graph_manager,
        )
        return f0


class FCPE:
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = spawn_bundled_infer_model(self.device)

    def get_f0(
        self,
        x,
        p_len=None,
        filter_radius=0.006,
        cuda_graph_manager=None,
    ):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        audio = x.float().to(self.device).unsqueeze(0)

        if cuda_graph_manager is None:
            f0 = self.model.infer(
                audio,
                sr=self.sample_rate,
                decoder_mode="local_argmax",
                threshold=filter_radius,
            )
        else:
            mel = self.model.wav2mel(audio, self.sample_rate)
            latent = cuda_graph_manager.run(
                self.model.model,
                "fcpe-network",
                self.model.model,
                mel,
            )
            cents = self.model.model.latent2cents_local_decoder(
                latent,
                threshold=filter_radius,
            )
            f0 = self.model.model.cent_to_f0(cents)
        f0 = f0.squeeze().cpu().numpy()

        return f0
