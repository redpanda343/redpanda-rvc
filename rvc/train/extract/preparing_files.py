import os
from random import shuffle
from rvc.configs.config import Config
import json

import numpy as np

config = Config()
current_directory = os.getcwd()


def generate_config(sample_rate: int, model_path: str):
    config_save_path = os.path.join(model_path, "config.json")
    with open(
        os.path.join(model_path, "model_info.json"), "r", encoding="utf-8"
    ) as info_file:
        model_info = json.load(info_file)
    embedder_model = model_info.get("embedder_model", "contentvec")
    feature_dim = int(model_info.get("feature_dim", 768))
    config_name = (
        "32000_spin_wavlm_512.json"
        if int(sample_rate) == 32000 and embedder_model == "spin-wavlm-512"
        else f"{sample_rate}.json"
    )
    config_path = os.path.join("rvc", "configs", config_name)
    source_path = config_save_path if os.path.isfile(config_save_path) else config_path
    with open(source_path, "r", encoding="utf-8") as config_file:
        config_data = json.load(config_file)
    config_data["model"]["text_enc_hidden_dim"] = feature_dim
    with open(config_save_path, "w", encoding="utf-8") as config_file:
        json.dump(config_data, config_file, indent=4)
        config_file.write("\n")


def generate_spin_mute_feature(feature_path):
    import torch

    from rvc.lib.utils import load_audio_16k, load_embedding

    audio_path = os.path.join(
        current_directory, "logs", "mute", "sliced_audios_16k", "mute.wav"
    )
    model = load_embedding("spin-wavlm-512").float().eval()
    audio = torch.from_numpy(load_audio_16k(audio_path)).float().view(1, -1)
    with torch.no_grad():
        feature = model(audio)["last_hidden_state"].squeeze(0).cpu().numpy()
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
    np.save(feature_path, feature, allow_pickle=False)


def generate_filelist(model_path: str, sample_rate: int, include_mutes: int = 2):
    gt_wavs_dir = os.path.join(model_path, "sliced_audios")
    feature_dir = os.path.join(model_path, f"extracted")

    f0_dir, f0nsf_dir = None, None
    f0_dir = os.path.join(model_path, "f0")
    f0nsf_dir = os.path.join(model_path, "f0_voiced")

    try:
        model_info_path = os.path.join(model_path, "model_info.json")
        with open(model_info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
            embedder_name = model_info["embedder_model"]
            dataset_format = str(model_info.get("dataset_format", "wav")).lower()
            feature_dim = int(model_info.get("feature_dim", 768))
    except:
        embedder_name = "contentvec"
        dataset_format = "wav"
        feature_dim = 768

    if dataset_format not in {"wav", "flac"}:
        dataset_format = "wav"

    if embedder_name == "spin-v2":
        mute_base_path = os.path.join(current_directory, "logs", "mute_spin-v2")
        mute_feature_base_path = mute_base_path
    elif embedder_name == "spin-wavlm-512":
        mute_base_path = os.path.join(current_directory, "logs", "mute")
        mute_feature_base_path = os.path.join(
            current_directory, "logs", "mute_spin-wavlm-512"
        )
    else:
        mute_base_path = os.path.join(current_directory, "logs", "mute")
        mute_feature_base_path = mute_base_path

    options = []
    sids = []
    if dataset_format == "flac":
        audio_files = sorted(
            filename
            for filename in os.listdir(gt_wavs_dir)
            if filename.lower().endswith(".flac")
        )
        if not audio_files:
            raise RuntimeError(f"No FLAC training slices found in {gt_wavs_dir}")

        for audio_file in audio_files:
            name = os.path.splitext(audio_file)[0]
            feature_file = f"{audio_file}.npy"
            f0_file = f"{audio_file}.npy"
            required_paths = (
                os.path.join(feature_dir, feature_file),
                os.path.join(f0_dir, f0_file),
                os.path.join(f0nsf_dir, f0_file),
            )
            missing_paths = [path for path in required_paths if not os.path.isfile(path)]
            if missing_paths:
                raise RuntimeError(
                    "Missing FLAC extraction artifacts: " + ", ".join(missing_paths)
                )

            sid = name.split("_")[0]
            if sid not in sids:
                sids.append(sid)
            rel_audio = os.path.relpath(os.path.join(gt_wavs_dir, audio_file))
            rel_feat = os.path.relpath(required_paths[0])
            rel_f0 = os.path.relpath(required_paths[1])
            rel_f0nsf = os.path.relpath(required_paths[2])
            options.append(
                f"{rel_audio}|{rel_feat}|{rel_f0}|{rel_f0nsf}|{sid}".replace(
                    "\\", "/"
                )
            )
    else:
        gt_wavs_files = {
            name[: -len(".wav")]
            for name in os.listdir(gt_wavs_dir)
            if name.lower().endswith(".wav")
        }
        feature_files = {
            name[: -len(".npy")]
            for name in os.listdir(feature_dir)
            if name.lower().endswith(".npy")
            and not name.lower().endswith(".flac.npy")
        }
        f0_files = {
            name[: -len(".wav.npy")]
            for name in os.listdir(f0_dir)
            if name.lower().endswith(".wav.npy")
        }
        f0nsf_files = {
            name[: -len(".wav.npy")]
            for name in os.listdir(f0nsf_dir)
            if name.lower().endswith(".wav.npy")
        }
        names = gt_wavs_files & feature_files & f0_files & f0nsf_files

        for name in names:
            sid = name.split("_")[0]
            if sid not in sids:
                sids.append(sid)

            # Preserve the existing WAV paths and naming.
            rel_wav = os.path.relpath(f"{os.path.join(gt_wavs_dir, name)}.wav")
            rel_feat = os.path.relpath(f"{os.path.join(feature_dir, name)}.npy")
            rel_f0 = os.path.relpath(f"{os.path.join(f0_dir, name)}.wav.npy")
            rel_f0nsf = os.path.relpath(
                f"{os.path.join(f0nsf_dir, name)}.wav.npy"
            )
            options.append(
                f"{rel_wav}|{rel_feat}|{rel_f0}|{rel_f0nsf}|{sid}".replace(
                    "\\", "/"
                )
            )

    if include_mutes > 0:
        mute_audio_path = os.path.relpath(
            os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav")
        )
        mute_feature_path = os.path.relpath(
            os.path.join(mute_feature_base_path, "extracted", "mute.npy")
        )
        mute_f0_path = os.path.relpath(
            os.path.join(mute_base_path, "f0", "mute.wav.npy")
        )
        mute_f0nsf_path = os.path.relpath(
            os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")
        )

        absolute_mute_feature_path = os.path.abspath(mute_feature_path)
        if embedder_name == "spin-wavlm-512" and not os.path.isfile(
            absolute_mute_feature_path
        ):
            generate_spin_mute_feature(absolute_mute_feature_path)
        if not os.path.isfile(absolute_mute_feature_path):
            raise RuntimeError(f"Mute feature not found: {absolute_mute_feature_path}")
        mute_feature = np.load(absolute_mute_feature_path, mmap_mode="r")
        if mute_feature.ndim != 2 or mute_feature.shape[1] != feature_dim:
            raise RuntimeError(
                f"Mute feature has shape {mute_feature.shape}; expected "
                f"[frames, {feature_dim}]."
            )

        # adding x files per sid
        for sid in sids * include_mutes:
            options.append(
                f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}"
            )

    file_path = os.path.join(model_path, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data.update(
        {
            "speakers_id": len(sids),
        }
    )
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    shuffle(options)

    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))
