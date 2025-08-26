import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from skimage.transform import warp_polar
from scipy.ndimage import gaussian_filter1d

def standardize_image(image):
    """
    Standardize the image to have mean 0 and standard deviation 1.

    Parameters:
    image : numpy array
        The input 2D image.

    Returns:
    standardized_image : numpy array
        The standardized 2D image.
    """
    mean = np.mean(image)
    std = np.std(image)

    if std == 0:
        raise ValueError("Standard deviation is zero, can't standardize the image.")

    standardized_image = (image - mean) / std
    return standardized_image
def autocorrelation(image, mode='same', method='fft', standardize=True):
    # standardize the image
    if standardize is True:
        image = standardize_image(image)
    return correlate(image, image, mode=mode, method=method)

def radial_profile(data):
    i, j = np.unravel_index(np.argmax(data), shape=data.shape)
    line = warp_polar(data, center=(i, j)).mean(axis=0)[0:i]
    return line

def estimate_patch_size_from_img(image, sigma=3, standardize=True, debug=False):
    autocorr = autocorrelation(image=image, standardize=standardize)
    line_profile = radial_profile(autocorr)
    line_profile = gaussian_filter1d(line_profile, sigma=sigma)
    peaks, _ = find_peaks(line_profile)
    if debug:
        plt.plot(peaks, line_profile[peaks], "x")
        plt.plot(line_profile)
    if len(peaks) > 0:
        return peaks[0]
    else:
        raise ValueError("No peak detected in the radial profile.")