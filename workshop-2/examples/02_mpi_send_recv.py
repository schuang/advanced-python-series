"""Point-to-point send/receive across two ranks."""

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 2:
    if rank == 0:
        raise RuntimeError("Launch with at least 2 processes to run this example.")

if rank == 0:
    payload = {"data": [1, 2, 3], "sender": rank}
    comm.send(payload, dest=1, tag=11)
    reply = comm.recv(source=1, tag=22)
    print(f"[Rank {rank}] Sent {payload} -> Received acknowledgement: {reply}")
elif rank == 1:
    received = comm.recv(source=0, tag=11)
    reply = {"status": "ok", "receiver": rank}
    comm.send(reply, dest=0, tag=22)
    print(f"[Rank {rank}] Received {received} -> Sent {reply}")
else:
    # Additional ranks idle but participate in the barrier to keep mpiexec happy.
    print(f"[Rank {rank}] Idle in this example.")

comm.Barrier()
