"""Utility phantom generators for FFT demos."""

from __future__ import annotations

import numpy as np

_ELLIPSES = [
    (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
    (-0.8, 0.6624, 0.8740, 0.0, -0.0184, 0.0),
    (-0.2, 0.1100, 0.3100, 0.22, 0.0, -18.0),
    (-0.2, 0.1600, 0.4100, -0.22, 0.0, 18.0),
    (0.1, 0.2100, 0.2500, 0.0, 0.35, 0.0),
    (0.1, 0.0460, 0.0460, 0.0, 0.1, 0.0),
    (0.1, 0.0460, 0.0460, 0.0, -0.1, 0.0),
    (0.1, 0.0460, 0.0230, -0.08, -0.605, 0.0),
    (0.1, 0.0230, 0.0230, 0.0, -0.606, 0.0),
    (0.1, 0.0230, 0.0460, 0.06, -0.605, 0.0),
]


def shepp_logan(shape: tuple[int, int]) -> np.ndarray:
    """Return a normalized Shepp-Logan phantom."""

    rows, cols = shape
    y = np.linspace(-1.0, 1.0, rows, endpoint=False)
    x = np.linspace(-1.0, 1.0, cols, endpoint=False)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    phantom = np.zeros(shape, dtype=np.float64)

    for amp, a, b, x0, y0, theta in _ELLIPSES:
        angle = np.deg2rad(theta)
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        x_rot = (xx - x0) * cos_t + (yy - y0) * sin_t
        y_rot = -(xx - x0) * sin_t + (yy - y0) * cos_t
        inside = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0
        phantom[inside] += amp

    phantom -= phantom.min()
    max_val = phantom.max()
    if max_val > 0:
        phantom /= max_val
    return phantom
