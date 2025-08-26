import os
import numpy as np

def load_data_npy(data_dir, file_name, use_mmap=False):
    filepath = os.path.join(data_dir, file_name)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")

    if not file_name.lower().endswith(".npy"):
        raise ValueError(f"Expected a .npy file, got {file_name}")

    if use_mmap:
        return np.load(filepath, mmap_mode="r")
    else:
        return np.load(filepath, allow_pickle=False)


