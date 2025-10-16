"""Scatter chunks of data, compute local results, and gather them back."""

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0
chunk_length = 2

if rank == root:
    full_data = list(range(size * chunk_length))
    chunks = [
        full_data[i * chunk_length : (i + 1) * chunk_length] for i in range(size)
    ]
else:
    chunks = None

local_chunk = comm.scatter(chunks, root=root)
local_sum = sum(local_chunk)
print(f"[Rank {rank}] Local chunk {local_chunk} -> local sum {local_sum}")

all_sums = comm.gather(local_sum, root=root)

if rank == root:
    total = sum(all_sums)
    print(f"[Rank {rank}] Gathered partial sums {all_sums} -> total {total}")
