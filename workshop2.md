# Workshop 2: Scaling Your Science with Parallel Computing

**Part of the series:** *From Scripts to Software: Practical Python for Reproducible Research*

Now that you have a solid foundation in writing sustainable and object-oriented Python code, we will explore how to scale up your research by breaking the single-processor barrier. This workshop will introduce you to the world of parallel computing, empowering you to tackle larger research problems.

## Part 1: The "Why" of Parallelism

As your datasets grow and your simulations become more complex, you will inevitably hit the limits of what a single CPU core can do. Moore's Law is no longer giving us faster clock speeds, but more cores. The key to unlocking the full potential of modern hardware is parallelism.

We will discuss:
*   The difference between **distributed memory** (like in a cluster) and **shared memory** (like in your laptop).
*   The types of problems that are easy to parallelize (**embarrassingly parallel**) and those that are more challenging (**tightly coupled**).
*   **Amdahl's Law**, which gives us a theoretical limit on the speedup we can achieve with parallelism.
*   Using MPI (Message Passing Interface) for distributed memory parallelism with `mpi4py`.
*   Using PETSc (the Portable, Extensible Toolkit for Scientific Computation) for high-performance scientific computing with `petsc4py`.

## Part 2: The MPI Paradigm

The Message Passing Interface (MPI) is the de facto standard for programming on distributed memory systems. It provides a powerful and portable way to write parallel programs that can run on anything from a multi-core laptop to a massive supercomputer.

We will cover the core concepts of MPI:
*   The **communicator**, which is a group of processes that can talk to each other.
*   The **rank**, which is the unique ID of a process within a communicator.
*   The **size**, which is the total number of processes in a communicator.

## Part 3: Hands-On with `mpi4py`

`mpi4py` is the most popular Python binding for MPI. We will use it to explore the core MPI communication patterns.

*   **Hello World in Parallel:** A simple example to get started.
    *   See example: [01_mpi_hello_world.py](workshop-2-examples/01_mpi_hello_world.py)

*   **Point-to-Point Communication:** Sending and receiving data between two specific processes.
    *   See example: [02_mpi_send_recv.py](workshop-2-examples/02_mpi_send_recv.py)

*   **Collective Communication (Broadcast):** Broadcasting data from one process to all others.
    *   See example: [03_mpi_bcast.py](workshop-2-examples/03_mpi_bcast.py)

*   **Collective Communication (Scatter/Gather):** Scattering data from one process to all others, and then gathering the results back.
    *   See example: [04_mpi_scatter_gather.py](workshop-2-examples/04_mpi_scatter_gather.py)

## Part 4: Applying Parallelism to the Golden Examples

Finally, we will apply these concepts to our two "golden examples."

*   **Heat Equation (Domain Decomposition):** We will see how to split our simulation grid across multiple processes and use halo/ghost cell exchange to communicate boundary conditions. This is a fundamental pattern in scientific HPC.
    *   See example: [05_heat_equation_mpi.py](workshop-2-examples/05_heat_equation_mpi.py)

*   **Deep Learning (Data Parallelism):** We will explore the concept of data parallelism, where we split our dataset across multiple processes, train our model on each subset, and then combine the results. This is the most common way to do distributed deep learning.
    *   See example: [06_deep_learning_data_parallel.py](workshop-2-examples/06_deep_learning_data_parallel.py)
