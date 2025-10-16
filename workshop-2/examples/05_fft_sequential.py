"""Sequential baseline FFT using NumPy (no MPI)."""

import numpy as np

# Problem dimensions match the MPI example for apples-to-apples comparison.
rows = 12
cols = 32

x = np.linspace(0, 2 * np.pi, cols, endpoint=False)
base_signal = np.sin(3 * x) + 0.3 * np.sin(9 * x)
data = np.vstack([np.roll(base_signal, shift) for shift in range(rows)])

# each row is a shifted version of the base signal:
# shift=0: [a, b, c, d] (no shift)
# shift=1: [d, a, b, c] (rotated right by 1)
# shift=2: [c, d, a, b] (rotated right by 2)
# ...

fft_result = np.fft.fft(data, axis=1)

print(f"Sequential FFT shape: {fft_result.shape}")
print(f"Sequential FFT sample[0, :5]: {fft_result[0, :5]}")
