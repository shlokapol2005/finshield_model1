"""
dct_utils.py — Discrete Cosine Transform utilities for the frequency branch.

GAN-generated images exhibit distinctive statistical fingerprints in the
frequency domain.  DCT coefficients expose these patterns even when visual
artifacts are imperceptible to the human eye.

Pipeline:
  RGB image  →  per-channel 2D DCT  →  log-scale  →  normalise to [0, 1]
"""

import numpy as np
from scipy.fft import dctn


def compute_dct(image_np: np.ndarray) -> np.ndarray:
    """
    Compute 2D DCT on each channel of an RGB image.

    Args:
        image_np: numpy array of shape (H, W, 3), dtype uint8 or float32,
                  pixel values in [0, 255].

    Returns:
        dct_image: numpy array of shape (H, W, 3), float32,
                   log-scaled and normalised to [0, 1].
    """
    image = image_np.astype(np.float32)

    # Ensure 3 channels
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    dct_channels = []
    for c in range(image.shape[2]):
        # 2D DCT (type-II, orthogonal normalisation)
        coeffs = dctn(image[:, :, c], type=2, norm="ortho")

        # Log-scale to compress the huge dynamic range
        # abs() because DCT coefficients can be negative
        coeffs = np.log1p(np.abs(coeffs))

        dct_channels.append(coeffs)

    dct_image = np.stack(dct_channels, axis=-1)

    # Normalise each channel to [0, 1] independently
    for c in range(dct_image.shape[2]):
        ch = dct_image[:, :, c]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min > 1e-8:
            dct_image[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
        else:
            dct_image[:, :, c] = 0.0

    return dct_image.astype(np.float32)
