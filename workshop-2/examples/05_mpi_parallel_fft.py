"""Distributed row-wise FFT using mpi4py and NumPy."""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()   # number of processes
root = 0

# Problem dimensions (rows split across ranks).
rows = 12
cols = 32

if size > rows:
    if rank == root:
        raise RuntimeError(
            "Example expects ranks <= number of rows so each process has work."
        )

if rank == root:
    x = np.linspace(0, 2 * np.pi, cols, endpoint=False)
    base_signal = np.sin(3 * x) + 0.3 * np.sin(9 * x)
    data = np.vstack([np.roll(base_signal, shift) for shift in range(rows)])

    # split the data into `size` chunks along rows
    chunks = np.array_split(data, size, axis=0)
else:
    data = None
    chunks = None

local_block = comm.scatter(chunks, root=root)
local_fft = np.fft.fft(local_block, axis=1)

gathered = comm.gather(local_fft, root=root)

if rank == root:
    fft_result = np.vstack(gathered)
    reference = np.fft.fft(data, axis=1)
    max_error = np.max(np.abs(fft_result - reference))
    print(f"[Rank {rank}] Combined FFT shape: {fft_result.shape}")
    print(f"[Rank {rank}] Parallel vs sequential max error: {max_error:.3e}")
else:
    print(f"[Rank {rank}] Processed rows: {local_block.shape[0]}")
