import json
import os
import sys
from multiprocessing import cpu_count

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Parse command line arguments
exp_dir = str(sys.argv[1])
index_algorithm = str(sys.argv[2])

feature_dir = os.path.join(exp_dir, "extracted")
model_name = os.path.basename(exp_dir)

if not os.path.exists(feature_dir):
    print(
        f"Feature to generate index file not found at {feature_dir}. Did you run preprocessing and feature extraction steps?"
    )
    sys.exit(1)

index_filename_added = f"{model_name}.index"
index_filepath_added = os.path.join(exp_dir, index_filename_added)

if os.path.exists(index_filepath_added):
    pass
else:
    npys = []
    print(f"Generating index for '{model_name}', this may take a while...")
    dataset_format = "wav"
    model_info_path = os.path.join(exp_dir, "model_info.json")
    try:
        with open(model_info_path, "r", encoding="utf-8") as f:
            dataset_format = str(json.load(f).get("dataset_format", "wav")).lower()
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if dataset_format == "flac":
        feature_dir_abs = os.path.normcase(os.path.abspath(feature_dir))
        feature_paths = []
        filelist_path = os.path.join(exp_dir, "filelist.txt")
        with open(filelist_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 2:
                    continue
                feature_path = os.path.abspath(parts[1])
                if os.path.normcase(os.path.dirname(feature_path)) == feature_dir_abs:
                    feature_paths.append(feature_path)
        feature_paths = sorted(set(feature_paths))
    else:
        feature_paths = [
            os.path.join(feature_dir, name)
            for name in sorted(os.listdir(feature_dir))
        ]

    for feature_path in feature_paths:
        phone = np.load(feature_path)
        npys.append(phone)

    if not npys:
        print(
            f"Feature files in {feature_dir} could not be loaded correctly. Did you run preprocessing and feature extraction steps?"
        )
        sys.exit(1)

    big_npy = np.concatenate(npys, axis=0)

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 2e5 or index_algorithm == "KMeans":
        big_npy = (
            MiniBatchKMeans(
                n_clusters=10000,
                verbose=True,
                batch_size=256 * cpu_count(),
                compute_labels=False,
                init="random",
            )
            .fit(big_npy)
            .cluster_centers_
        )

    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)

    # index_added
    index_added = faiss.index_factory(768, f"IVF{n_ivf},Flat")
    index_ivf_added = faiss.extract_index_ivf(index_added)
    index_ivf_added.nprobe = 1
    index_added.train(big_npy)

    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index_added.add(big_npy[i : i + batch_size_add])

    faiss.write_index(index_added, index_filepath_added)
    print(f"Saved index file '{index_filepath_added}'")
