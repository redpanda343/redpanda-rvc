import argparse
import hashlib
import json
import os
import re

import torch
from safetensors.torch import save_file
from transformers import WavLMConfig, WavLMModel


EXPECTED_SHA256 = "1915d2a05e69a33fa644de4db206232a3b2d1d7ca57de7e089c8d7a431dd0340"
SOURCE_URL = "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_wavlm_512.ckpt"


class WandbLogger:
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path):
    actual_sha256 = file_sha256(path)
    if actual_sha256 != EXPECTED_SHA256:
        raise ValueError(
            f"Unexpected checkpoint SHA-256: {actual_sha256}. Expected {EXPECTED_SHA256}."
        )
    safe_types = [(WandbLogger, "pytorch_lightning.loggers.wandb.WandbLogger")]
    with torch.serialization.safe_globals(safe_types):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint, actual_sha256


def validate_recipe(checkpoint):
    recipe = checkpoint["hyper_parameters"]
    encoder = recipe["model"]["encoder"]
    prediction = recipe["model"]["pred_head"]
    loss = recipe["model"]["loss"]
    expected = {
        "encoder_type": (encoder["type"], "WavLM"),
        "use_layer": (encoder["use_layer"], 12),
        "projection": (prediction["hid_dims"], [256]),
        "clusters": (loss["num_vars"], 512),
        "l2_norm": (loss["l2_norm"], True),
        "sample_rate": (recipe["data"]["sample_rate"], 16000),
    }
    invalid = [name for name, values in expected.items() if values[0] != values[1]]
    if invalid:
        raise ValueError("Unexpected SPIN recipe values: " + ", ".join(invalid))


def wavlm_config():
    return WavLMConfig(
        activation_dropout=0.0,
        apply_spec_augment=True,
        attention_dropout=0.1,
        conv_bias=False,
        conv_dim=[512, 512, 512, 512, 512, 512, 512],
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        do_stable_layer_norm=False,
        feat_extract_norm="group",
        feat_proj_dropout=0.1,
        hidden_act="gelu",
        hidden_dropout=0.1,
        hidden_size=768,
        intermediate_size=3072,
        layer_norm_eps=1e-5,
        layerdrop=0.05,
        mask_feature_prob=0.0,
        mask_time_prob=0.05,
        max_bucket_distance=800,
        num_attention_heads=12,
        num_buckets=320,
        num_conv_pos_embedding_groups=16,
        num_conv_pos_embeddings=128,
        num_hidden_layers=12,
    )


def map_encoder_key(source_key):
    key = source_key.removeprefix("encoder.model.")
    if key == "mask_emb":
        return "masked_spec_embed"
    match = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)", key)
    if match:
        return f"feature_extractor.conv_layers.{match.group(1)}.conv.{match.group(2)}"
    match = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.2\.(weight|bias)", key)
    if match:
        return f"feature_extractor.conv_layers.{match.group(1)}.layer_norm.{match.group(2)}"
    if key.startswith("post_extract_proj."):
        return key.replace("post_extract_proj.", "feature_projection.projection.")
    if key.startswith("layer_norm."):
        return key.replace("layer_norm.", "feature_projection.layer_norm.")
    if key.startswith("encoder.pos_conv.0."):
        suffix = key.removeprefix("encoder.pos_conv.0.")
        suffix = {
            "weight_g": "parametrizations.weight.original0",
            "weight_v": "parametrizations.weight.original1",
        }.get(suffix, suffix)
        return f"encoder.pos_conv_embed.conv.{suffix}"
    match = re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", key)
    if match:
        layer, suffix = match.groups()
        suffix = suffix.replace("self_attn.grep_a", "attention.gru_rel_pos_const")
        suffix = suffix.replace("self_attn.grep_linear", "attention.gru_rel_pos_linear")
        suffix = suffix.replace(
            "self_attn.relative_attention_bias", "attention.rel_attn_embed"
        )
        suffix = suffix.replace("self_attn.", "attention.")
        suffix = suffix.replace("self_attn_layer_norm.", "layer_norm.")
        suffix = suffix.replace("fc1.", "feed_forward.intermediate_dense.")
        suffix = suffix.replace("fc2.", "feed_forward.output_dense.")
        return f"encoder.layers.{layer}.{suffix}"
    if key.startswith("encoder.layer_norm."):
        return key
    raise KeyError(f"Unsupported encoder tensor: {source_key}")


def convert(checkpoint_path, output_path):
    checkpoint, checkpoint_sha256 = load_checkpoint(checkpoint_path)
    validate_recipe(checkpoint)
    source = checkpoint["state_dict"]
    encoder_source = {
        key: value for key, value in source.items() if key.startswith("encoder.model.")
    }
    converted = {map_encoder_key(key): value for key, value in encoder_source.items()}

    model = WavLMModel(wavlm_config())
    model.load_state_dict(converted, strict=True)
    model.eval()
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    projection = {
        "weight": source["pred_head.layers.0.weight"].contiguous(),
        "bias": source["pred_head.layers.0.bias"].contiguous(),
    }
    save_file(projection, os.path.join(output_path, "spin_projection.safetensors"))

    spin_config = {
        "model_type": "spin-wavlm",
        "source": SOURCE_URL,
        "source_checkpoint_sha256": checkpoint_sha256,
        "sample_rate": 16000,
        "feature_hop_samples": 320,
        "encoder_layer": 12,
        "encoder_dim": 768,
        "feature_dim": 256,
        "feature_output": "spin_projection",
        "clusters": 512,
        "l2_normalize": True,
        "audio_requires_normalization": False,
    }
    with open(
        os.path.join(output_path, "spin_config.json"), "w", encoding="utf-8"
    ) as config_file:
        json.dump(spin_config, config_file, indent=2)
        config_file.write("\n")

    preprocessor_config = {
        "do_normalize": False,
        "feature_extractor_type": "Wav2Vec2FeatureExtractor",
        "feature_size": 1,
        "padding_side": "right",
        "padding_value": 0.0,
        "return_attention_mask": False,
        "sampling_rate": 16000,
    }
    with open(
        os.path.join(output_path, "preprocessor_config.json"),
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(preprocessor_config, config_file, indent=2)
        config_file.write("\n")
    print(
        f"Converted SPIN WavLM 512 to {output_path} with SHA-256 "
        f"{checkpoint_sha256}."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to spin_wavlm_512.ckpt")
    parser.add_argument("output", help="Destination embedder directory")
    args = parser.parse_args()
    convert(os.path.abspath(args.checkpoint), os.path.abspath(args.output))


if __name__ == "__main__":
    main()
