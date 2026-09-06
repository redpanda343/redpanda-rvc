import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import WavLMModel


class SpinWavLMModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        config_path = os.path.join(model_path, "spin_config.json")
        projection_path = os.path.join(model_path, "spin_projection.safetensors")
        with open(config_path, "r", encoding="utf-8") as config_file:
            self.spin_config = json.load(config_file)

        self.encoder = WavLMModel.from_pretrained(model_path, local_files_only=True)
        input_dim = int(self.spin_config["encoder_dim"])
        output_dim = int(self.spin_config["feature_dim"])
        self.projection = torch.nn.Linear(input_dim, output_dim)
        self.projection.load_state_dict(load_file(projection_path), strict=True)
        self.audio_requires_normalization = bool(
            self.spin_config.get("audio_requires_normalization", False)
        )
        self.feature_dim = output_dim
        self.feature_output = self.spin_config["feature_output"]
        self.feature_fingerprint = self.spin_config["source_checkpoint_sha256"]

    def forward(self, input_values):
        hidden_states = self.encoder(input_values).last_hidden_state
        features = self.projection(hidden_states)
        if self.spin_config.get("l2_normalize", True):
            features = F.normalize(features, dim=-1)
        return {"last_hidden_state": features}
