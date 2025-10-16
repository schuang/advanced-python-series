"""Simplified MRI reconstruction using NumPy FFTs."""

from __future__ import annotations

import numpy as np

from _phantoms import shepp_logan


def main() -> None:
    shape = (128, 128)
    phantom = shepp_logan(shape)
    kspace = np.fft.fft2(phantom)
    recon = np.fft.ifft2(kspace)

    max_err = np.max(np.abs(recon.real - phantom))

    print(f"Phantom shape: {phantom.shape}")
    print(f"Sequential recon max error: {max_err:.3e}")
    print(f"Phantom sample[0, :5]: {phantom[0, :5]}")
    print(f"Recon sample[0, :5]: {recon.real[0, :5]}")


if __name__ == "__main__":
    main()
