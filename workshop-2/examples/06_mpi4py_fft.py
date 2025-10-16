"""2D FFT using mpi4py-fft."""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

try:
    from mpi4py_fft import PFFT, newDistArray
except ImportError as exc:  # pragma: no cover - optional dependency
    if rank == 0:
        print(
            "mpi4py-fft is not installed. Install with `python -m pip install mpi4py-fft`."
        )
    raise

# Global domain matches the NumPy-only and mpi4py demos.
global_shape = (12, 32)

# Plan a 2D complex FFT across all available ranks.
fft = PFFT(comm, global_shape, axes=(0, 1), dtype=np.complex128)

# Create distributed arrays: u (spatial domain) and uh (frequency domain).
u = newDistArray(fft, forward_output=False)
uh = newDistArray(fft, forward_output=True)

# Fill the local portion with reproducible random signal (complex values).
rng = np.random.default_rng(seed=42)  # Same seed for all ranks for consistency
u[:] = rng.standard_normal(u.shape) + 1j * rng.standard_normal(u.shape)

# Keep a copy of the input before transforming.
u_original = u.copy()

# Forward and backward transforms (note: backward is unnormalized by default).
fft.forward(u, uh)
fft.backward(uh, u)

# Compute reconstruction error (library handles normalization automatically).
local_error = float(np.max(np.abs(u - u_original)))
global_error = comm.allreduce(local_error, op=MPI.MAX)

if rank == 0:
    print(f"[mpi4py-fft] Global shape: {global_shape}, ranks: {comm.Get_size()}")
    print(f"[mpi4py-fft] Max reconstruction error: {global_error:.3e}")
