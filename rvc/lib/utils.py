import hashlib
import os
import sys
import tempfile
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


def get_embedding_metadata(embedder_model):
    if embedder_model not in {"contentvec", "spin-v2"}:
        raise ValueError(f"Unsupported embedder model: {embedder_model}")
    return {
        "embedder_model": embedder_model,
        "feature_dim": 768,
        "feature_output": "last_hidden_state",
        "feature_fingerprint": embedder_model,
    }


def load_audio_16k(file):
    try:
        audio, sr = sf.read(file, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        if sr != 16000:
            audio = soxr.resample(audio, sr, 16000, quality="HQ")
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)


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


CONTENTVEC_SHA256 = "d8dd400e054ddf4e6be75dab5a2549db748cc99e756a097c496c099f65a4854e"


def _sha256(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_file(url, destination_path, expected_sha256=None):
    directory = os.path.dirname(destination_path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=directory,
            prefix=f".{os.path.basename(destination_path)}.",
            suffix=".part",
        ) as temporary_file:
            temporary_path = temporary_file.name

        print(f"Downloading {url} to {directory}...")
        wget.download(url, out=temporary_path)

        if expected_sha256 is not None and _sha256(temporary_path) != expected_sha256:
            raise RuntimeError(
                f"Checksum verification failed for {destination_path}. "
                "The downloaded ContentVec checkpoint does not match Applio."
            )

        os.replace(temporary_path, destination_path)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_embedding(embedder_model):
    embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")
    rvc_contentvec_base_url = (
        "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec"
    )
    embedding_list = {
        "contentvec": os.path.join(embedder_root, "contentvec"),
        "spin-v2": os.path.join(embedder_root, "spin-v2"),
    }

    online_embedders = {
        "contentvec": f"{rvc_contentvec_base_url}/pytorch_model.bin",
        "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/pytorch_model.bin",
    }

    config_files = {
        "contentvec": f"{rvc_contentvec_base_url}/config.json",
        "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/config.json",
    }

    if embedder_model not in embedding_list:
        raise ValueError(f"Unsupported embedder model: {embedder_model}")
    model_path = embedding_list[embedder_model]
    bin_file = os.path.join(model_path, "pytorch_model.bin")
    json_file = os.path.join(model_path, "config.json")
    preprocessor_json_file = os.path.join(model_path, "preprocessor_config.json")
    os.makedirs(model_path, exist_ok=True)

    if embedder_model == "contentvec":
        if os.path.isfile(preprocessor_json_file):
            os.remove(preprocessor_json_file)
            print(f"Removed legacy ContentVec preprocessor config: {preprocessor_json_file}")

        checkpoint_is_valid = (
            os.path.isfile(bin_file)
            and os.path.getsize(bin_file) > 0
            and _sha256(bin_file) == CONTENTVEC_SHA256
        )
        if not checkpoint_is_valid:
            if os.path.exists(bin_file):
                print("ContentVec checkpoint SHA-256 mismatch; replacing it with Applio's checkpoint.")
            _download_file(
                online_embedders[embedder_model],
                bin_file,
                expected_sha256=CONTENTVEC_SHA256,
            )
    elif not os.path.exists(bin_file):
        _download_file(online_embedders[embedder_model], bin_file)

    if not os.path.exists(json_file):
        _download_file(config_files[embedder_model], json_file)

    models = HubertModelWithFinalProj.from_pretrained(model_path)
    if os.path.isfile(preprocessor_json_file):
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_path, local_files_only=True
        )
        models.audio_requires_normalization = bool(feature_extractor.do_normalize)
    else:
        models.audio_requires_normalization = False
    metadata = get_embedding_metadata(embedder_model)
    models.feature_dim = metadata["feature_dim"]
    models.feature_output = metadata["feature_output"]
    models.feature_fingerprint = metadata["feature_fingerprint"]
    return models
