import numpy as np


def estimate_file_size(image_size, channel, image_number, unit='GB'):
    """
    Estimate the size of a .npy file containing np.float32 data

    Parameters:
    image_size: int - image dimension (height and width)
    channel: int - number of channels
    image_number: int - number of images
    unit: str - output unit ('MB' or 'GB'), defaults to 'GB'

    Returns:
    float - estimated file size in specified unit (rounded to 2 decimal places)
    """
    # Calculate data size in bytes (np.float32 uses 4 bytes per element)
    data_size_bytes = image_number * channel * image_size * image_size * 4

    # Calculate .npy file header size
    # Construct header dictionary as numpy would
    header_dict = {
        'descr': '<f4',  # little-endian float32
        'fortran_order': False,
        'shape': (image_number, channel, image_size, image_size)
    }

    # Header string representation (simulating actual save format)
    header_str = str(header_dict)
    header_len = len(header_str) + 1  # +1 for newline character

    # .npy file format structure:
    # magic string(6 bytes) + version(2 bytes) + header length(2 bytes) + header data + padding
    magic_version_len = 6 + 2 + 2  # 10 bytes

    # Calculate aligned header length (aligned to 16 bytes)
    total_header_len_without_padding = magic_version_len + header_len
    padded_header_len = ((total_header_len_without_padding + 15) // 16) * 16

    # Header size in bytes
    header_size_bytes = padded_header_len

    # Total file size in bytes
    total_size_bytes = data_size_bytes + header_size_bytes

    # Convert to requested unit
    if unit.upper() == 'MB':
        total_size = total_size_bytes / (1024 * 1024)
        unit_str = 'MB'
    elif unit.upper() == 'GB':
        total_size = total_size_bytes / (1024 * 1024 * 1024)
        unit_str = 'GB'
    else:
        raise ValueError("Unit must be either 'MB' or 'GB'")

    # Format breakdown information for printing
    data_size_mb = data_size_bytes / (1024 * 1024)
    header_size_mb = header_size_bytes / (1024 * 1024)
    total_size_mb = total_size_bytes / (1024 * 1024)

    # Print results
    print("\n" + "=" * 60)
    print("NPY FILE SIZE ESTIMATION RESULTS")
    print("=" * 60)
    print(f"Array shape: ({image_number}, {channel}, {image_size}, {image_size})")
    print(f"Data type: np.float32")
    print(f"Data size: {data_size_mb:.2f} MB")
    print(f"Header size: {header_size_mb:.3f} MB")
    print(f"Total size: {total_size_mb:.2f} MB ({total_size:.2f} {unit_str})")
    print(f"Header overhead: {(header_size_bytes / total_size_bytes) * 100:.2f}%")
    print("=" * 60)

    return total_size