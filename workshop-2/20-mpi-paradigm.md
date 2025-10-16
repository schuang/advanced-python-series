# Message Passing Interface (MPI)

## What Is MPI?

- MPI stands for Message Passing Interface, a standard for processes to exchange data by sending messages. 

- Each process runs its own program instance and shares results through explicit communication. 
  
- Designed for high-performance clusters, MPI works on laptops, workstations, and supercomputers. 

- Different languages (C, C++, Fortran, Python, R, Julia, etc.) follow the same communication rules.

## What Is a Process?

- A process is an independent program instance with its own memory space and resources. 
- MPI launches many processes that run the same code on different slices of the data. Processes communicate by passing messages instead of sharing memory directly.

## What Is a Rank?

- A rank is the ID number MPI assigns to each process inside a communicator. 
- Ranks let you address a specific process when sending messages. 
- Rank 0 can act as coordinator or root, while other ranks (1, 2, …) handle parallel work, but this is not always the case.

## What Is `MPI_COMM_WORLD`?

- This is the default communicator created at startup, containing every MPI process in the job. 
- It provides `Get_rank()` and `Get_size()` for global coordination. This is a good starting point; create custom communicators later for subgroups.

## Where MPI Came From

- In the early 1990s, the community unified competing libraries into MPI-1 (1994). 
- MPI-2 (1997) added dynamic processes and parallel I/O. 
- MPI-3 (2012) added nonblocking collectives, shared windows, and more. 
- The standard API covers C, C++, and Fortran. 
- Vendors implement the same calls for portability. 
- Popular implementations: MPICH, Open MPI, MVAPICH2
- Python uses `mpi4py` to wrap the C bindings with a clean, NumPy-friendly layer.

## Why MPI Scales Everywhere

- MPI provides a common language for distributed-memory systems. 
- One specification runs portably from laptops to supercomputers. 
- Well-written MPI code moves from laptop tests to supercomputer runs without rewrites. 
- `mpi4py` keeps Python productive while tapping into the same MPI foundation.

## Core Vocabulary

- Communicators define who can talk to whom; start with `MPI.COMM_WORLD`. 
- Rank identifies each process; size reports how many are participating. 
- Custom communicators carve out subgroups for focused collaboration.

## Point-to-Point Communication

- One sender talks to one receiver; each "point" is a process (rank). Blocking `Send` and `Recv` complete only when buffers are safe to reuse. 
- Nonblocking `Isend` and `Irecv` overlap communication with computation; finish via `Wait` or `Test`. 
- Tags label message streams so receives can pull exactly what they expect.

## Point-to-Point Examples

- Boundary exchange: rank 0 sends edge cells to rank 1 (`comm.send(boundary, dest=1)`), then receives updates back. 
- Ping-pong timing: two ranks trade a token (`Send`/`Recv`) to measure latency. 
- Task farm: workers `Recv` work units from rank 0, process them, then `Send` results home.

## Collective Communication

- Broadcast: one root rank shares identical data with everyone. 
- Scatter: root distributes distinct chunks to each rank.
- Gather: ranks collect results back to the root; `Allgather` returns a copy to every rank. 
- Reduce and Allreduce: combine values (sum, max, etc.) across ranks, optionally sharing the result with all. 
- Barrier: everyone waits until all ranks arrive before continuing.

