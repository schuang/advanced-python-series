"""MPI hello world"""

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = MPI.Get_processor_name()

if rank == 0:
    print(f"[Rank {rank}/{size}] Greetings from the root process on {hostname}.")
else:
    print(f"[Rank {rank}/{size}] Hello from {hostname}.")
