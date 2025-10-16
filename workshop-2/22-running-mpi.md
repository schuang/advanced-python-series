# Running simple MPI programs

## Launching MPI Programs

Run scripts with `mpirun` or `mpiexec` to spin up multiple processes:

```bash
mpirun -n 4 python examples/01_mpi_hello_world.py
```

- Make sure the MPI runtime is on `PATH` and `LD_LIBRARY_PATH`. 
- Activate the Python environment that has `mpi4py`. 
- Reserve nodes and cores (scheduler directives) before launching on clusters.

## Hands-On Flow

- Start with environment checks and practice launching scripts via `mpiexec` or `mpirun`. 
- Watch for rank-tagged print statements to confirm parallel execution.

## Writing `mpi4py` Programs

- Hello world: import `MPI`, grab `COMM_WORLD`, print rank, size, and hostname. 
- Point-to-point: use `comm.send` and `comm.recv`
  -  (Optional exercise) Switch to `comm.isend` and `comm.irecv` with `Wait` to overlap work. 
- Collectives: `comm.bcast`, `comm.scatter`, `comm.gather`, `comm.reduce` map directly onto MPI theory. 
- Keep data types consistent across ranks—prefer NumPy arrays or explicit dtype hints.

 

## Hello World Walkthrough

- Explore `examples/01_mpi_hello_world.py` to map ranks to hosts. 
- Print from both root and non-root processes to see the full communicator. 
- There is no need to run `MPI_Init` and `MPI_Finalize` as in standard C/C++/Fortran MPI; `mpi4py` automatically handles them.

## Send/Recv Patterns

- Step through `examples/02_mpi_send_recv.py` to show the handshake. 
- Emphasize matching tags and datatypes for deterministic ordering.

## Broadcast and Collectives

- Use `examples/03_mpi_bcast.py` to share data from root to every rank. 
- Broadcast beats repeated point-to-point sends in performance.

## Scatter and Gather

- Dive into `examples/04_mpi_scatter_gather.py` to partition work and collect results. 
- Connect the pattern to chunking large arrays and aggregating statistics.

## Debugging and Validation

- Prefix log messages with rank IDs to trace execution order. 
- Drop `comm.Barrier()` between phases to expose ordering or race issues. 
- Time sections with `MPI.Wtime()` to capture communication overhead. 

## MPI in Practice

- The MPI standard defines semantics, datatypes, and communication patterns; language bindings expose the calls. 
- Implementations (MPICH, Open MPI, Intel MPI, MVAPICH2) follow the same spec, so the **same user code** can run with different MPI implementations on different parallel computers.
- MPI offers performance, scalability, and portability across mixed hardware generations. 
- `mpi4py` maps MPI communicators and datatypes into Python objects, leaning on NumPy buffers or pickles instead of raw pointers. 
- Exceptions propagate MPI errors in Python, giving friendlier feedback during development.
