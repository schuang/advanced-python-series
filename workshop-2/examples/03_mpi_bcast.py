"""Broadcast an object from the root rank to all processes."""

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
root = 0

if rank == root:
    config = {"timestep": 0.05, "iterations": 3, "description": "demo broadcast"}
else:
    config = None

config = comm.bcast(config, root=root)
print(f"[Rank {rank}] Received config: {config}")
