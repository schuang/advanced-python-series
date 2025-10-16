"""Simplified MRI reconstruction split across MPI ranks."""

import numpy as np
from mpi4py import MPI

from _phantoms import shepp_logan

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

shape = (128, 128)

if size > shape[0]:
    if rank == root:
        raise RuntimeError(
            "Number of ranks must be <= image rows for this row-wise demo."
        )

if rank == root:
    phantom = shepp_logan(shape)
    kspace = np.fft.fft2(phantom)
    reference = np.fft.ifft2(kspace)
    chunks = np.array_split(kspace, size, axis=0)
else:
    phantom = None
    reference = None
    chunks = None

local_kspace = comm.scatter(chunks, root=root)

# First stage: inverse FFT along columns on each rank.
stage_one = np.fft.ifft(local_kspace, axis=1)

partials = comm.gather(stage_one, root=root)

if rank == root:
    assembled = np.vstack(partials)
    recon = np.fft.ifft(assembled, axis=0)
    max_err = np.max(np.abs(recon - reference))
    image_err = np.max(np.abs(recon.real - phantom))
    print(f"[Rank 0] Parallel MRI recon max error vs sequential: {max_err:.3e}")
    print(f"[Rank 0] Parallel MRI recon max error vs phantom: {image_err:.3e}")
    print(f"[Rank 0] Recon sample[0, :5]: {recon.real[0, :5]}")
else:
    print(f"[Rank {rank}] Processed rows: {local_kspace.shape[0]}")
