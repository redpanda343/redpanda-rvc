import json
import os
import sys
import soxr
import librosa
import soundfile as sf
import numpy as np
import re
import unicodedata
import wget
from torch import nn

import logging
from transformers import AutoFeatureExtractor, HubertModel
import warnings

from rvc.lib.embedders.spin_wavlm import SpinWavLMModel

# Remove this to see warnings about transformers models
warnings.filterwarnings("ignore")

logging.getLogger("fairseq").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

now_dir = os.getcwd()
sys.path.append(now_dir)


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def get_embedding_metadata(embedder_model, custom_embedder=None):
    embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")
    chosen_model = custom_embedder if embedder_model == "custom" else embedder_model
    model_path = (
        custom_embedder
        if embedder_model == "custom"
        else os.path.join(embedder_root, embedder_model)
    )
    if embedder_model == "custom" and not os.path.exists(str(model_path)):
        chosen_model = "contentvec"
        model_path = os.path.join(embedder_root, "contentvec")
        embedder_model = "contentvec"
    if embedder_model == "spin-wavlm-512":
        from rvc.lib.tools.convert_spin_wavlm import ensure_converted

        ensure_converted(model_path)
        config_path = os.path.join(model_path, "spin_config.json")
        with open(config_path, "r", encoding="utf-8") as config_file:
            spin_config = json.load(config_file)
        return {
            "embedder_model": chosen_model,
            "feature_dim": int(spin_config["feature_dim"]),
            "feature_output": spin_config["feature_output"],
            "feature_fingerprint": spin_config["source_checkpoint_sha256"],
        }

    feature_dim = 768
    if embedder_model == "custom" and model_path:
        config_path = os.path.join(model_path, "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                feature_dim = int(json.load(config_file).get("hidden_size", 768))
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return {
        "embedder_model": chosen_model,
        "feature_dim": feature_dim,
        "feature_output": "last_hidden_state",
        "feature_fingerprint": str(chosen_model),
    }


def load_audio_16k(file):
    # this is used by f0 and feature extractions that load preprocessed 16k files, so there's no need to resample
    try:
        audio, sr = librosa.load(file, sr=16000)
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq"
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio_infer(
    file,
    sample_rate,
):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File not found: {file}")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq"
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")
    return np.array(audio).flatten()


def format_title(title):
    formatted_title = unicodedata.normalize("NFC", title)
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title, flags=re.UNICODE)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title


def load_embedding(embedder_model, custom_embedder=None):
    embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")
    rvc_contentvec_base_url = (
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base"
    )
    embedding_list = {
        "contentvec": os.path.join(embedder_root, "contentvec"),
        "spin-v2": os.path.join(embedder_root, "spin-v2"),
        "spin-wavlm-512": os.path.join(embedder_root, "spin-wavlm-512"),
    }

    online_embedders = {
        "contentvec": f"{rvc_contentvec_base_url}/pytorch_model.bin",
        "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/pytorch_model.bin",
    }

    config_files = {
        "contentvec": f"{rvc_contentvec_base_url}/config.json",
        "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/config.json",
    }
    preprocessor_config_files = {
        "contentvec": f"{rvc_contentvec_base_url}/preprocessor_config.json",
    }

    if embedder_model == "custom":
        if os.path.exists(custom_embedder):
            model_path = custom_embedder
        else:
            print(f"Custom embedder not found: {custom_embedder}, using contentvec")
            model_path = embedding_list["contentvec"]
    else:
        model_path = embedding_list[embedder_model]
        if embedder_model == "spin-wavlm-512":
            from rvc.lib.tools.convert_spin_wavlm import ensure_converted

            ensure_converted(model_path)
            return SpinWavLMModel(model_path)
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        json_file = os.path.join(model_path, "config.json")
        preprocessor_json_file = os.path.join(
            model_path, "preprocessor_config.json"
        )
        os.makedirs(model_path, exist_ok=True)
        if not os.path.exists(bin_file):
            url = online_embedders[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=bin_file)
        if not os.path.exists(json_file):
            url = config_files[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=json_file)
        if (
            embedder_model in preprocessor_config_files
            and not os.path.exists(preprocessor_json_file)
        ):
            url = preprocessor_config_files[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=preprocessor_json_file)

    models = HubertModelWithFinalProj.from_pretrained(model_path)
    preprocessor_json_file = os.path.join(model_path, "preprocessor_config.json")
    if os.path.isfile(preprocessor_json_file):
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_path, local_files_only=True
        )
        models.audio_requires_normalization = bool(feature_extractor.do_normalize)
    else:
        models.audio_requires_normalization = False
    metadata = get_embedding_metadata(
        "contentvec"
        if embedder_model == "custom" and model_path == embedding_list["contentvec"]
        else embedder_model,
        custom_embedder,
    )
    models.feature_dim = metadata["feature_dim"]
    models.feature_output = metadata["feature_output"]
    models.feature_fingerprint = metadata["feature_fingerprint"]
    return models
