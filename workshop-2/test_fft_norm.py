"""Test to understand mpi4py-fft normalization."""

from __future__ import annotations

import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

global_shape = (4, 4)  # Small for easier debugging
fft = PFFT(comm, global_shape, axes=(0, 1), dtype=np.complex128)

u = newDistArray(fft, forward_output=False)
uh = newDistArray(fft, forward_output=True)

# Create a simple test signal - all ones
u[:] = 1.0 + 0j

# Gather all data to rank 0 for inspection
u_global = np.zeros(global_shape, dtype=np.complex128) if rank == 0 else None
comm.Reduce(u, u_global, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Input signal (rank 0 view): {u_global}")
    print(f"Sum of input: {np.sum(u_global)}")

# Forward transform
fft.forward(u, uh)

# Gather frequency domain
uh_global = np.zeros(global_shape, dtype=np.complex128) if rank == 0 else None  
comm.Reduce(uh, uh_global, op=MPI.SUM, root=0)

if rank == 0:
    print(f"\nAfter forward FFT:")
    print(f"Frequency domain DC component (should be sum of input): {uh_global[0,0]}")
    print(f"Expected: {global_shape[0] * global_shape[1]}")

# Backward transform
fft.backward(uh, u)

# Gather result
u_result = np.zeros(global_shape, dtype=np.complex128) if rank == 0 else None
comm.Reduce(u, u_result, op=MPI.SUM, root=0)

if rank == 0:
    print(f"\nAfter backward FFT (before manual normalization):")
    print(f"Result[0,0]: {u_result[0,0]}")
    print(f"Should be: {global_shape[0] * global_shape[1]} (unnormalized)")
    print(f"Or should be: 1.0 (if already normalized)")
