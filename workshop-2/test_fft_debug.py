"""Debug FFT normalization."""

from __future__ import annotations

import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

global_shape = (12, 32)
fft = PFFT(comm, global_shape, axes=(0, 1), dtype=np.complex128)

u = newDistArray(fft, forward_output=False)
uh = newDistArray(fft, forward_output=True)

# Simple test: fill with ones
u[:] = 1.0 + 0j
u_original = np.array(u, copy=True)

if rank == 0:
    print(f"Before transform: u[0,0] = {u[0,0]}")

fft.forward(u, uh)

if rank == 0:
    print(f"After forward: uh[0,0] = {uh[0,0]}")

fft.backward(uh, u)

if rank == 0:
    print(f"After backward: u[0,0] = {u[0,0]}")
    print(f"Expected after proper normalization: {u_original[0,0]}")
    
total_points = np.prod(global_shape)
if rank == 0:
    print(f"Total points = {total_points}")
    print(f"u[0,0] / total_points = {u[0,0] / total_points}")
