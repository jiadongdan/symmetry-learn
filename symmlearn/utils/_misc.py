import os
import psutil
import numpy as np

def get_free_ram():
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)
    used_gb = (mem.total - mem.available) / (1024**3)

    return {
        "total_GB": round(total_gb, 2),
        "used_GB": round(used_gb, 2),
        "free_GB": round(available_gb, 2)
    }

def peek_npy_info(filepath):
    # Open with memory mapping so it doesn't fully load into RAM
    arr = np.load(filepath, mmap_mode="r")

    # File size on disk
    file_size_gb = os.path.getsize(filepath) / (1024**3)

    # Estimate memory usage if fully loaded
    bytes_per_element = arr.dtype.itemsize
    ram_size_gb = (arr.size * bytes_per_element) / (1024**3)

    info = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "file_size_GB": round(file_size_gb, 2),
        "estimated_RAM_GB": round(ram_size_gb, 2)
    }

    return info


def load_data_npy(filepath):
    # check free RAM
    available_ram_gb = get_free_ram()['free_GB']
    estimated_data_gb = peek_npy_info(filepath)['estimated_RAM_GB']
    if estimated_data_gb < available_ram_gb / 2.:
        return np.load(filepath)  # fully load into RAM
    else:
        return np.load(filepath, mmap_mode='r')  # memory mapping

