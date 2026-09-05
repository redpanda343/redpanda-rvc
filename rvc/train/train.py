import os
import sys

os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"
import datetime
import glob
import hashlib
import json
from collections import defaultdict
from random import randint, shuffle
from time import time as ttime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

from rvc.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.train.mel_processing import (
    MultiScaleMelSpectrogramLoss,
    mel_spectrogram_torch,
    spec_to_mel_torch,
)
from rvc.train.utils import (
    HParams,
    latest_checkpoint_path,
    load_checkpoint,
    load_wav_to_torch,
    plot_spectrogram_to_numpy,
    save_checkpoint,
    summarize,
)

# Zluda hijack
import rvc.lib.zluda
from rvc.lib.algorithm import commons
from rvc.train.process.extract_model import extract_model
from rvc.train.timbre_validation import ECAPATimbreValidator

# Parse command line arguments
model_name = sys.argv[1]
save_every_epoch = int(sys.argv[2])
total_epoch = int(sys.argv[3])
pretrainG = sys.argv[4]
pretrainD = sys.argv[5]
gpus = sys.argv[6]
batch_size = int(sys.argv[7])
sample_rate = int(sys.argv[8])


def _speaker_id(item):
    try:
        return int(item[4])
    except (TypeError, ValueError):
        return 0


def _timbre_reference_from_samples(samples, collate_fn, device):
    info = collate_fn(samples)
    phone, phone_lengths, pitch, pitchf, _, _, wave, wave_lengths, sid = info
    inference_inputs = (
        phone.to(device),
        phone_lengths.to(device),
        pitch.to(device),
        pitchf.to(device),
        sid.to(device),
    )
    return inference_inputs, wave, wave_lengths, sid


def build_timbre_reference(dataset, collate_fn, device, max_samples=4):
    candidates = []
    for index, item in enumerate(dataset.audiopaths_and_text):
        audio_path = item[0]
        if "mute" in os.path.basename(audio_path).lower():
            continue
        candidates.append((_speaker_id(item), audio_path, index))
    candidates.sort()

    selected = []
    selected_indices = set()
    selected_speakers = set()
    for speaker_id, _, index in candidates:
        if speaker_id in selected_speakers:
            continue
        sample = dataset[index]
        if sample[1].abs().mean().item() <= 1e-4:
            continue
        selected.append(sample)
        selected_indices.add(index)
        selected_speakers.add(speaker_id)
        if len(selected) == max_samples:
            break

    if len(selected) < max_samples:
        for _, _, index in candidates:
            if index in selected_indices:
                continue
            sample = dataset[index]
            if sample[1].abs().mean().item() <= 1e-4:
                continue
            selected.append(sample)
            if len(selected) == max_samples:
                break

    if not selected:
        raise RuntimeError("No non-silent training audio is available")

    return _timbre_reference_from_samples(selected, collate_fn, device)


def _dataset_signature(items):
    serialized = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validation_sort_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def build_timbre_enrollment(
    dataset,
    collate_fn,
    speaker_ids,
    seed,
    max_per_speaker=3,
):
    target_speakers = sorted({int(speaker_id) for speaker_id in speaker_ids})
    groups = defaultdict(list)
    for index, item in enumerate(dataset.audiopaths_and_text):
        speaker_id = _speaker_id(item)
        if speaker_id not in target_speakers:
            continue
        audio_path = item[0]
        if "mute" in os.path.basename(audio_path).lower():
            continue
        groups[speaker_id].append((audio_path, index))

    selected = []
    for speaker_id in target_speakers:
        candidates = sorted(
            groups[speaker_id],
            key=lambda item: _validation_sort_key(
                seed, f"enrollment:{speaker_id}:{item[0]}"
            ),
        )
        speaker_samples = []
        for _, index in candidates:
            try:
                sample = dataset[index]
            except Exception:
                continue
            if sample[1].abs().mean().item() <= 1e-4:
                continue
            speaker_samples.append(sample)
            if len(speaker_samples) == max_per_speaker:
                break
        if not speaker_samples:
            raise RuntimeError(f"No enrollment audio is available for speaker {speaker_id}")
        selected.extend(speaker_samples)

    info = collate_fn(selected)
    return info[6], info[7], info[8]


def _select_held_out_paths(
    dataset,
    seed,
    max_samples=16,
    max_per_speaker=4,
    min_samples_per_speaker=8,
):
    groups = defaultdict(list)
    for index, item in enumerate(dataset.audiopaths_and_text):
        audio_path = item[0]
        if "mute" in os.path.basename(audio_path).lower():
            continue
        groups[_speaker_id(item)].append((audio_path, index))

    speaker_order = sorted(
        groups,
        key=lambda speaker_id: _validation_sort_key(seed, f"speaker:{speaker_id}"),
    )
    candidates = {}
    targets = {}
    for speaker_id in speaker_order:
        group = sorted(
            groups[speaker_id],
            key=lambda item: _validation_sort_key(seed, item[0]),
        )
        available = len(group) - min_samples_per_speaker
        if available < 1:
            continue
        target = max(1, (len(group) + 19) // 20)
        candidates[speaker_id] = group
        targets[speaker_id] = min(target, max_per_speaker, available)

    selected = []
    cursors = defaultdict(int)
    counts = defaultdict(int)
    while len(selected) < max_samples:
        progress = False
        for speaker_id in speaker_order:
            if speaker_id not in candidates or counts[speaker_id] >= targets[speaker_id]:
                continue
            group = candidates[speaker_id]
            while cursors[speaker_id] < len(group):
                audio_path, index = group[cursors[speaker_id]]
                cursors[speaker_id] += 1
                try:
                    sample = dataset[index]
                except Exception:
                    continue
                if sample[1].abs().mean().item() <= 1e-4:
                    continue
                selected.append(audio_path)
                counts[speaker_id] += 1
                progress = True
                break
            if len(selected) == max_samples:
                break
        if not progress:
            break
    return selected


def prepare_held_out_timbre_reference(
    dataset,
    collate_fn,
    device,
    experiment_dir,
    rank,
    seed,
    minimum_training_samples,
):
    manifest_path = os.path.join(experiment_dir, "held_out_validation.json")
    signature = _dataset_signature(dataset.audiopaths_and_text)
    manifest_box = [None]
    if rank == 0:
        manifest = None
        try:
            with open(manifest_path, "r", encoding="utf-8") as file:
                candidate = json.load(file)
            if (
                isinstance(candidate, dict)
                and candidate.get("version") == 2
                and candidate.get("dataset_signature") == signature
                and candidate.get("seed") == seed
                and candidate.get("minimum_training_samples")
                == minimum_training_samples
                and isinstance(candidate.get("audio_paths"), list)
            ):
                manifest = candidate
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

        if manifest is None:
            held_out_paths = _select_held_out_paths(dataset, seed)
            allowed = max(0, len(dataset) - minimum_training_samples)
            held_out_paths = held_out_paths[:allowed]
            manifest = {
                "version": 2,
                "dataset_signature": signature,
                "seed": seed,
                "minimum_training_samples": minimum_training_samples,
                "audio_paths": held_out_paths,
            }
            try:
                temporary_path = f"{manifest_path}.{os.getpid()}.tmp"
                with open(temporary_path, "w", encoding="utf-8") as file:
                    json.dump(manifest, file, ensure_ascii=False, indent=2)
                os.replace(temporary_path, manifest_path)
            except OSError as error:
                print(f"Could not persist the held-out validation split: {error}")
        manifest_box[0] = manifest

    dist.broadcast_object_list(manifest_box, src=0)
    held_out_paths = set(manifest_box[0].get("audio_paths", []))
    if not held_out_paths:
        return None, False

    training_items = []
    training_lengths = []
    validation_items = []
    for item, length in zip(dataset.audiopaths_and_text, dataset.lengths):
        if item[0] in held_out_paths:
            validation_items.append(item)
        else:
            training_items.append(item)
            training_lengths.append(length)
    dataset.audiopaths_and_text = training_items
    dataset.lengths = training_lengths

    if rank != 0:
        return None, True

    samples = []
    for item in validation_items:
        try:
            sample = dataset.get_audio_text_pair(item)
        except Exception as error:
            print(f"Could not load held-out validation sample '{item[0]}': {error}")
            continue
        if sample[1].abs().mean().item() > 1e-4:
            samples.append(sample)

    if not samples:
        return None, False

    print(
        f"Held-out ECAPA validation uses {len(samples)} clips; {len(dataset)} clips remain for training."
    )
    return _timbre_reference_from_samples(samples, collate_fn, device), True


def _strtobool(val):
    return val.lower() in ("yes", "true", "t", "y", "1")


save_only_latest = _strtobool(sys.argv[9])
save_every_weights = _strtobool(sys.argv[10])
cache_data_in_gpu = _strtobool(sys.argv[11])
cleanup = _strtobool(sys.argv[12])
vocoder = sys.argv[13]
checkpointing = _strtobool(sys.argv[14])
save_every_steps = max(0, int(sys.argv[15])) if len(sys.argv) > 15 else 0
# experimental settings
randomized = True
d_lr_coeff = 1.0
g_lr_coeff = 1.0
d_step_per_g_step = 1
multiscale_mel_loss = False
bf16_adamw = False
disc_version = "v2"

if vocoder == "RefineGAN":
    disc_version = "v3"
    multiscale_mel_loss = True

current_dir = os.getcwd()

try:
    with open(
        os.path.join(current_dir, "assets", "config.json"),
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)
        precision = config["precision"]
        if (
            precision == "bf16"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            train_dtype = torch.bfloat16
        elif precision == "fp16" and torch.cuda.is_available():
            train_dtype = torch.float16
        else:
            train_dtype = torch.float32
except (FileNotFoundError, json.JSONDecodeError, KeyError):
    train_dtype = torch.float32

inference_export_dtype = (
    torch.float32 if train_dtype == torch.float32 else torch.float16
)

experiment_dir = os.path.join(current_dir, "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")
dataset_path = os.path.join(experiment_dir, "sliced_audios")
model_info_path = os.path.join(experiment_dir, "model_info.json")

try:
    with open(config_save_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config = HParams(**config)
except FileNotFoundError:
    print(
        f"Config file not found at {config_save_path}. Did you run preprocessing and feature extraction steps?"
    )
    sys.exit(1)

config.data.training_files = os.path.join(experiment_dir, "filelist.txt")

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
# TF32 settings, should improve performance in some cases
try:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception as e:
    print(f'Torch tf32: {e}')

global_step = 0
last_loss_gen_all = 0
training_file_path = os.path.join(experiment_dir, "training_data.json")

import logging

logging.getLogger("torch").setLevel(logging.ERROR)


class EpochRecorder:
    """
    Records the time elapsed per epoch.
    """

    def __init__(self):
        self.last_time = ttime()

    def record(self):
        """
        Records the elapsed time and returns a formatted string.
        """
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return f"time={current_time} | training_speed={elapsed_time_str}"


def main():
    """
    Main function to start the training process.
    """
    global training_file_path, last_loss_gen_all, gpus

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    # Check sample rate
    dataset_format = "wav"
    try:
        with open(model_info_path, "r", encoding="utf-8") as f:
            dataset_format = str(json.load(f).get("dataset_format", "wav")).lower()
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    if dataset_format not in {"wav", "flac"}:
        dataset_format = "wav"
    wavs = glob.glob(
        os.path.join(
            os.path.join(experiment_dir, "sliced_audios"),
            f"*.{dataset_format}",
        )
    )
    if wavs:
        _, sr = load_wav_to_torch(wavs[0])
        if sr != config.data.sample_rate:
            print(
                f"Error: Pretrained model sample rate ({config.data.sample_rate} Hz) does not match dataset audio sample rate ({sr} Hz)."
            )
            os._exit(1)
    else:
        print(f"No {dataset_format} file found.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpus = [int(item) for item in gpus.split("-")]
        n_gpus = len(gpus)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        gpus = [0]
        n_gpus = 1
    else:
        device = torch.device("cpu")
        gpus = [0]
        n_gpus = 1
        print("Training with CPU, this will take a long time.")

    def start():
        """
        Starts the training process with multi-GPU support or CPU.
        """
        children = []
        for rank, device_id in enumerate(gpus):
            subproc = mp.Process(
                target=run,
                args=(
                    rank,
                    n_gpus,
                    experiment_dir,
                    pretrainG,
                    pretrainD,
                    total_epoch,
                    save_every_weights,
                    config,
                    device,
                    device_id,
                ),
            )
            children.append(subproc)
            subproc.start()

        for i in range(n_gpus):
            children[i].join()

    if cleanup:
        print("Removing files from the prior training attempt...")

        # Clean up unnecessary files
        for root, dirs, files in os.walk(
            os.path.join(now_dir, "logs", model_name), topdown=False
        ):
            for name in files:
                file_path = os.path.join(root, name)
                file_name, file_extension = os.path.splitext(name)
                if (
                    file_extension == ".0"
                    or file_extension == ".pth"
                    or name.endswith(".spec.pt")
                    or (file_name.startswith("added") and file_extension == ".index")
                ):
                    os.remove(file_path)
            for name in dirs:
                if name == "eval":
                    folder_path = os.path.join(root, name)
                    for item in os.listdir(folder_path):
                        item_path = os.path.join(folder_path, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                    os.rmdir(folder_path)

        print("Cleanup done!")

    start()


def run(
    rank,
    n_gpus,
    experiment_dir,
    pretrainG,
    pretrainD,
    custom_total_epoch,
    custom_save_every_weights,
    config,
    device,
    device_id,
):
    """
    Runs the training loop on a specific GPU or CPU.

    Args:
        rank (int): The rank of the current process within the distributed training setup.
        n_gpus (int): The total number of GPUs available for training.
        experiment_dir (str): The directory where experiment logs and checkpoints will be saved.
        pretrainG (str): Path to the pre-trained generator model.
        pretrainD (str): Path to the pre-trained discriminator model.
        custom_total_epoch (int): The total number of epochs for training.
        custom_save_every_weights (int): The interval (in epochs) at which to save model weights.
        config (object): Configuration object containing training parameters.
        device (torch.device): The device to use for training (CPU or GPU).
    """
    global global_step

    if rank == 0:
        writer_eval = SummaryWriter(
            log_dir=os.path.join(experiment_dir, "eval"), max_queue=1000
        )
    else:
        writer_eval = None

    dist.init_process_group(
        backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )

    torch.manual_seed(config.train.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    # Create datasets and dataloaders
    from data_utils import (
        DistributedBucketSampler,
        TextAudioCollateMultiNSFsid,
        TextAudioLoaderMultiNSFsid,
    )

    train_dataset = TextAudioLoaderMultiNSFsid(config.data)
    collate_fn = TextAudioCollateMultiNSFsid()
    timbre_reference, timbre_is_held_out = prepare_held_out_timbre_reference(
        train_dataset,
        collate_fn,
        device,
        experiment_dir,
        rank,
        config.train.seed,
        max(8, batch_size * n_gpus * 3),
    )
    audio_reference = None
    if rank == 0 and timbre_is_held_out and timbre_reference is not None:
        audio_reference = tuple(value[:1] for value in timbre_reference[0])
        print("TensorBoard audio validation uses one held-out dataset clip.")
    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size,
        [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )

    # Validations
    if len(train_loader) < 3:
        print(
            "Not enough data present in the training set. Perhaps you forgot to slice the audio files in preprocess?"
        )
        os._exit(2333333)

    # defaults
    spk_dim = config.model.spk_embed_dim  # 109 default speakers

    try:
        with open(model_info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
            spk_dim = model_info["speakers_id"]
    except Exception as e:
        print(f"Could not load model info file: {e}. Using defaults.")

    dataset_spk_dim = spk_dim
    last_g = latest_checkpoint_path(experiment_dir, "G_*.pth")
    fresh_speaker_embeddings = last_g is None

    try:
        chk_path = last_g
        if chk_path:
            ckpt = torch.load(chk_path, map_location="cpu", weights_only=True)
            spk_dim = ckpt["model"]["emb_g.weight"].shape[0]
            del ckpt
    except Exception as e:
        print(f"Failed to load checkpoint: {e}. Using default number of speakers.")

    # update config before the model init
    print(f"Initializing the generator with {spk_dim} speakers.")
    config.model.spk_embed_dim = spk_dim

    # Initialize models and optimizers
    from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
    from rvc.lib.algorithm.synthesizers import Synthesizer

    net_g = Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0=True,
        sr=config.data.sample_rate,
        vocoder=vocoder,
        checkpointing=checkpointing,
        randomized=randomized,
    )

    net_d = MultiPeriodDiscriminator(
        config.model.use_spectral_norm,
        checkpointing=checkpointing,
        version=disc_version,
    )

    if torch.cuda.is_available():
        net_g = net_g.cuda(device_id)
        net_d = net_d.cuda(device_id)
    else:
        net_g = net_g.to(device)
        net_d = net_d.to(device)

    if bf16_adamw == True and train_dtype == torch.bfloat16:
        print("Using BFload16 AdamW optimizer")
        from rvc.train.anyprecision_optimizer import AnyPrecisionAdamW

        optimizer = AnyPrecisionAdamW
    else:
        print("Using AdamW optimizer")
        optimizer = torch.optim.AdamW

    optim_g = optimizer(
        net_g.parameters(),
        config.train.learning_rate * g_lr_coeff,
        betas=config.train.betas,
        eps=config.train.eps,
    )
    optim_d = optimizer(
        net_d.parameters(),
        config.train.learning_rate * d_lr_coeff,
        betas=config.train.betas,
        eps=config.train.eps,
    )
    if multiscale_mel_loss:
        fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=config.data.sample_rate)
        print("Using Multi-Scale Mel loss function")
    else:
        fn_mel_loss = torch.nn.L1Loss()
        print("Using Single-Scale Mel loss function")

    # Wrap models with DDP for multi-gpu processing
    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[device_id])
        net_d = DDP(net_d, device_ids=[device_id])

    if rank == 0 and train_dtype == torch.bfloat16:
        print("Using BFloat16 for training.")
    elif rank == 0 and train_dtype == torch.float16:
        print("Using Float16 for training.")

    # Load checkpoint if available
    scaler_dict = {}
    try:
        print("Starting training...")
        _, _, _, epoch_str, scaler_dict = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "D_*.pth"), net_d, optim_d
        )
        _, _, _, epoch_str, _ = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "G_*.pth"), net_g, optim_g
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)

    except Exception as e:
        epoch_str = 1
        global_step = 0

        if pretrainG not in ("", "None"):
            if rank == 0:
                print(f"Loaded pretrained (G) '{pretrainG}'")
            try:
                ckpt = torch.load(pretrainG, map_location="cpu", weights_only=True)[
                    "model"
                ]
                target_net_g = net_g.module if hasattr(net_g, "module") else net_g
                if fresh_speaker_embeddings:
                    if "emb_g.weight" not in ckpt:
                        raise KeyError(
                            "The pretrained generator has no speaker embedding."
                        )
                    ckpt["emb_g.weight"] = (
                        target_net_g.emb_g.weight.detach().cpu().clone()
                    )
                target_net_g.load_state_dict(ckpt)
                del ckpt
            except Exception as e:
                print(
                    "The parameters of the pretrain model such as the sample rate or architecture do not match the selected model."
                )
                print(e)
                sys.exit(1)

        if pretrainD not in ("", "None"):
            if rank == 0:
                print(f"Loaded pretrained (D) '{pretrainD}'")
            try:
                ckpt = torch.load(pretrainD, map_location="cpu", weights_only=True)[
                    "model"
                ]
                if hasattr(net_d, "module"):
                    net_d.module.load_state_dict(ckpt)
                else:
                    net_d.load_state_dict(ckpt)
                del ckpt
            except Exception as e:
                print(
                    "The parameters of the pretrain model such as the sample rate or architecture do not match the selected model."
                )
                print(e)
                sys.exit(1)

    # Initialize schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.train.lr_decay, last_epoch=epoch_str - 2
    )

    use_scaler = device.type == "cuda" and train_dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    if len(scaler_dict) > 0:
        scaler.load_state_dict(scaler_dict)

    cache = []

    timbre_validator = None
    if rank == 0:
        try:
            timbre_model_path = os.path.join(
                "rvc", "models", "pretraineds", "ecapa_tdnn", "pretrain.model"
            )
            timbre_validator = ECAPATimbreValidator(timbre_model_path)
            if timbre_reference is None:
                timbre_reference = build_timbre_reference(
                    train_dataset, collate_fn, device
                )
                timbre_is_held_out = False
                print("ECAPA validation is using the training-reference fallback.")
            if timbre_is_held_out:
                enrollment_wave, enrollment_lengths, enrollment_speakers = (
                    build_timbre_enrollment(
                        train_dataset,
                        collate_fn,
                        timbre_reference[3],
                        config.train.seed,
                    )
                )
            else:
                enrollment_wave = timbre_reference[1]
                enrollment_lengths = timbre_reference[2]
                enrollment_speakers = timbre_reference[3]
            timbre_validator.set_references(
                enrollment_wave,
                enrollment_lengths,
                enrollment_speakers,
                config.data.sample_rate,
            )
            print(
                f"ECAPA timbre validation enabled with {len(timbre_reference[3])} probes and {len(enrollment_speakers)} enrollment clips."
            )
        except Exception as error:
            print(f"ECAPA timbre validation disabled: {error}")
            timbre_validator = None

    for epoch in range(epoch_str, total_epoch + 1):
        train_and_evaluate(
            rank,
            epoch,
            config,
            [net_g, net_d],
            [optim_g, optim_d],
            [train_loader, None],
            [writer_eval],
            cache,
            custom_save_every_weights,
            custom_total_epoch,
            device,
            device_id,
            audio_reference,
            timbre_validator,
            timbre_reference,
            timbre_is_held_out,
            fn_mel_loss,
            scaler,
        )

        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    loaders,
    writers,
    cache,
    custom_save_every_weights,
    custom_total_epoch,
    device,
    device_id,
    audio_reference,
    timbre_validator,
    timbre_reference,
    timbre_is_held_out,
    fn_mel_loss,
    scaler,
):
    """
    Trains and evaluates the model for one epoch.

    Args:
        rank (int): Rank of the current process.
        epoch (int): Current epoch number.
        hps (Namespace): Hyperparameters.
        nets (list): List of models [net_g, net_d].
        optims (list): List of optimizers [optim_g, optim_d].
        loaders (list): List of dataloaders [train_loader, eval_loader].
        writers (list): List of TensorBoard writers [writer_eval].
        cache (list): List to cache data in GPU memory.
        use_cpu (bool): Whether to use CPU for training.
    """
    global global_step, loss_disc

    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader = loaders[0] if loaders is not None else None
    if writers is not None:
        writer = writers[0]

    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()
    freeze_discriminator_for_generator = device.type == "cuda" and not isinstance(
        net_d, DDP
    )

    use_amp = device.type == "cuda" and (
        train_dtype == torch.bfloat16 or train_dtype == torch.float16
    )

    # Data caching
    if device.type == "cuda" and cache_data_in_gpu:
        data_iterator = cache
        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                # phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
                cache.append((batch_idx, info))
        else:
            shuffle(cache)
    else:
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()
    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in data_iterator:
            if device.type == "cuda" and not cache_data_in_gpu:
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
            elif device.type != "cuda":
                info = [tensor.to(device) for tensor in info]
            # else iterator is going thru a cached list with a device already assigned

            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info

            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
                # Forward pass
                model_output = net_g(
                    phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
                )
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = (
                    model_output
                )
                # slice of the original waveform to match a generate slice
                if randomized:
                    wave = commons.slice_segments(
                        wave,
                        ids_slice * config.data.hop_length,
                        config.train.segment_size,
                        dim=3,
                    )
            for _ in range(d_step_per_g_step):  # default x1
                with torch.amp.autocast(
                    device_type="cuda", enabled=use_amp, dtype=train_dtype
                ):
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
                # Discriminator backward and update
                optim_d.zero_grad()
                if train_dtype == torch.float16:
                    scaler.scale(loss_disc).backward()
                    scaler.unscale_(optim_d)
                    grad_norm_d = commons.grad_norm(net_d.parameters())
                    scaler.step(optim_d)
                else:
                    loss_disc.backward()
                    grad_norm_d = commons.grad_norm(net_d.parameters())
                    optim_d.step()

            if freeze_discriminator_for_generator:
                optim_d.zero_grad(set_to_none=True)
                net_d.requires_grad_(False)

            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
                # Generator backward and update
                _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)

            if multiscale_mel_loss:
                loss_mel = fn_mel_loss(wave, y_hat) * config.train.c_mel / 3.0
                loss_kl = (
                    kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
                )
                loss_fm = feature_loss(fmap_r, fmap_g)
            else:
                wave_mel = spec_to_mel_torch(
                    spec,
                    config.data.filter_length,
                    config.data.n_mel_channels,
                    config.data.sample_rate,
                    config.data.mel_fmin,
                    config.data.mel_fmax,
                )
                wave_mel = commons.slice_segments(
                    wave_mel,
                    ids_slice,
                    config.train.segment_size // config.data.hop_length,
                    dim=3,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    config.data.filter_length,
                    config.data.n_mel_channels,
                    config.data.sample_rate,
                    config.data.hop_length,
                    config.data.win_length,
                    config.data.mel_fmin,
                    config.data.mel_fmax,
                )
                loss_mel = fn_mel_loss(wave_mel, y_hat_mel) * config.train.c_mel
                loss_kl = (
                    kl_loss(
                        z_p.float(),
                        logs_q.float(),
                        m_p.float(),
                        logs_p.float(),
                        z_mask.float(),
                    )
                    * config.train.c_kl
                )
                loss_fm = feature_loss(
                    [
                        [real_feature.float().detach() for real_feature in disc]
                        for disc in fmap_r
                    ],
                    [
                        [generated_feature.float() for generated_feature in disc]
                        for disc in fmap_g
                    ],
                )
            loss_gen, _ = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            optim_g.zero_grad()
            if train_dtype == torch.float16:
                scaler.scale(loss_gen_all).backward()
                scaler.unscale_(optim_g)
                grad_norm_g = commons.grad_norm(net_g.parameters())
                scaler.step(optim_g)
                scaler.update()
            else:
                loss_gen_all.backward()
                grad_norm_g = commons.grad_norm(net_g.parameters())
                optim_g.step()

            if freeze_discriminator_for_generator:
                net_d.requires_grad_(True)

            global_step += 1

            if rank == 0:
                (
                    loss_gen_all_value,
                    loss_disc_value,
                    loss_gen_value,
                    loss_fm_value,
                    loss_mel_value,
                    loss_kl_value,
                ) = (
                    torch.stack(
                        (
                            loss_gen_all.detach(),
                            loss_disc.detach(),
                            loss_gen.detach(),
                            loss_fm.detach(),
                            loss_mel.detach(),
                            loss_kl.detach(),
                        )
                    )
                    .float()
                    .cpu()
                    .tolist()
                )
                summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars={
                        "loss/g/total": loss_gen_all_value,
                        "loss/d/adv": loss_disc_value,
                        "learning_rate": optim_g.param_groups[0]["lr"],
                        "grad/norm_d": grad_norm_d,
                        "grad/norm_g": grad_norm_g,
                        "loss/g/adv": loss_gen_value,
                        "loss/g/fm": loss_fm_value,
                        "loss/g/mel": loss_mel_value,
                        "loss/g/kl": loss_kl_value,
                    },
                )

            if (
                rank == 0
                and save_every_steps > 0
                and global_step % save_every_steps == 0
            ):
                inference_model_path = os.path.join(
                    experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                )
                if not os.path.exists(inference_model_path):
                    inference_ckpt = (
                        net_g.module.state_dict()
                        if hasattr(net_g, "module")
                        else net_g.state_dict()
                    )
                    extract_model(
                        ckpt=inference_ckpt,
                        sr=config.data.sample_rate,
                        name=model_name,
                        model_path=inference_model_path,
                        epoch=epoch,
                        step=global_step,
                        hps=hps,
                        vocoder=vocoder,
                        export_dtype=inference_export_dtype,
                    )

            pbar.update(1)
        # end of batch train
    # end of tqdm

    # Logging and checkpointing
    if rank == 0:
        # used for tensorboard chart - all/mel
        mel = spec_to_mel_torch(
            spec,
            config.data.filter_length,
            config.data.n_mel_channels,
            config.data.sample_rate,
            config.data.mel_fmin,
            config.data.mel_fmax,
        )
        # used for tensorboard chart - slice/mel_org
        if randomized:
            y_mel = commons.slice_segments(
                mel,
                ids_slice,
                config.train.segment_size // config.data.hop_length,
                dim=3,
            )
        else:
            y_mel = mel
        # used for tensorboard chart - slice/mel_gen
        y_hat_mel = mel_spectrogram_torch(
            y_hat.float().squeeze(1),
            config.data.filter_length,
            config.data.n_mel_channels,
            config.data.sample_rate,
            config.data.hop_length,
            config.data.win_length,
            config.data.mel_fmin,
            config.data.mel_fmax,
        )

        validation_scalars = {}

        image_dict = {
            "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
            "all/mel": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }

        if epoch % save_every_epoch == 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            inference_model = net_g.module if hasattr(net_g, "module") else net_g
            inference_model.eval()
            rng_devices = [device_id] if device.type == "cuda" else []
            audio_o = None
            timbre_o = None
            try:
                with torch.random.fork_rng(devices=rng_devices):
                    torch.manual_seed(config.train.seed)
                    with torch.amp.autocast(
                        device_type="cuda", enabled=use_amp, dtype=train_dtype
                    ):
                        with torch.inference_mode():
                            if audio_reference is not None:
                                audio_o, *_ = inference_model.infer(*audio_reference)
                            if timbre_validator is not None:
                                try:
                                    timbre_o, *_ = inference_model.infer(
                                        *timbre_reference[0]
                                    )
                                except Exception as error:
                                    print(f"ECAPA reference generation failed: {error}")
            finally:
                inference_model.train()

            if timbre_validator is not None and timbre_o is not None:
                try:
                    speaker_ids = timbre_reference[3]
                    generated_lengths = (
                        timbre_reference[0][1].detach().cpu()
                        * config.data.hop_length
                    )
                    timbre_scores = timbre_validator.score_batch(
                        timbre_o.detach().cpu(),
                        generated_lengths,
                        speaker_ids,
                        config.data.sample_rate,
                    )
                    if timbre_scores["multi_speaker"]:
                        validation_scalars.update(
                            {
                                "validation/ecapa_cosine_mean": timbre_scores["mean"],
                                "validation/ecapa_margin_mean": timbre_scores[
                                    "margin_mean"
                                ],
                                "validation/ecapa_top1_accuracy_percent": timbre_scores[
                                    "top1_accuracy_percent"
                                ],
                                "validation/ecapa_eer_percent": timbre_scores[
                                    "eer_percent"
                                ],
                            }
                        )
                    else:
                        validation_scalars.update(
                            {
                                "validation/ecapa_cosine_mean": timbre_scores["mean"],
                                "validation/ecapa_cosine_min": timbre_scores["min"],
                            }
                        )
                except Exception as error:
                    print(f"ECAPA timbre validation failed: {error}")
            audio_dict = {}
            if audio_o is not None:
                audio_dict[f"gen/audio_{global_step:07d}"] = audio_o[0, :, :]
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=validation_scalars,
                audios=audio_dict,
                audio_sample_rate=config.data.sample_rate,
            )
        else:
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
            )

    # Save checkpoint
    model_add = []
    model_del = []
    done = False

    if rank == 0:
        # Print training progress
        record = f"{model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()}"
        print(record)

        # Save weights every N epochs
        if epoch % save_every_epoch == 0:
            checkpoint_suffix = f"{2333333 if save_only_latest else global_step}.pth"
            save_checkpoint(
                net_g,
                optim_g,
                config.train.learning_rate,
                epoch,
                os.path.join(experiment_dir, "G_" + checkpoint_suffix),
                scaler,
            )
            save_checkpoint(
                net_d,
                optim_d,
                config.train.learning_rate,
                epoch,
                os.path.join(experiment_dir, "D_" + checkpoint_suffix),
                scaler,
            )
            if custom_save_every_weights and save_every_steps == 0:
                model_add.append(
                    os.path.join(
                        experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                    )
                )

        # Check completion
        if epoch >= custom_total_epoch:
            print(
                f"Training has been successfully completed with {epoch} epoch, {global_step} steps and {round(loss_gen_all.item(), 3)} loss gen."
            )
            # Final model
            model_add.append(
                os.path.join(
                    experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                )
            )
            done = True

        # Clean-up old best epochs
        for m in model_del:
            os.remove(m)

        if model_add:
            ckpt = (
                net_g.module.state_dict()
                if hasattr(net_g, "module")
                else net_g.state_dict()
            )
            for m in model_add:
                if not os.path.exists(m):
                    extract_model(
                        ckpt=ckpt,
                        sr=config.data.sample_rate,
                        name=model_name,
                        model_path=m,
                        epoch=epoch,
                        step=global_step,
                        hps=hps,
                        vocoder=vocoder,
                        export_dtype=inference_export_dtype,
                    )

        if done:
            writer.close()
            os._exit(2333333)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
