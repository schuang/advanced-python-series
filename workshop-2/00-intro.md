---
title: "**Scaling your science with parallel computing**"
author: "Shao-Ching Huang"
date: 2025-10-17
---

Workshop series:

1. Foundations for sustainable research software (October 9)

2. **Scaling your science with parallel computing (October 16)**

3. Accelerating your code with GPUs (October 30)




Resources:

- Workshop materials [[link]](https://github.com/schuang/advanced-python-series)

- UCLA Office of Advanced Research Computing [[link]](https://oarc.ucla.edu/contact)

- Hoffman2 Cluster [[link]](https://www.hoffman2.idre.ucla.edu)


# Introuction


## Recap from workshop 1

- Structure research code with **functions** and **classes** that follow single-responsibility principles, use clear signatures, and handle errors deliberately.

- Lean on object-oriented patterns (instance, class, and static methods, inheritance and abstraction) to encapsulate scientific workflows.

- Use modern Python features like dataclasses, decorators, and type aliases to simplify programming, reduce boilerplate while keeping intent explicit.

- Package projects with a `src/` layout, `pyproject.toml`, semantic versioning, and tagged releases so collaborators can install the exact code you share.

- Lock down reproducibility by recording dependencies, capturing environments (conda, Docker, Apptainer), and looping through the write-test-release development cycle.



## Goals for today


- Workshop 2 in the Advanced Python Series: shift your Python workflows from a single laptop to HPC clusters and supercomputers

- Today's focus: understanding when and why to parallelize, MPI fundamentals, hands-on practice with mpi4py, PETSc workflows, and real-world examples


  - Introduce distributed-memory parallel computing using MPI

  - Use `mpi4py` to access MPI functionality from Python

  - Introduce PETSc for scaling research applications onto HPC clusters (PDE solvers, climate models, MRI reconstruction, genome analytics, multi-omics pipelines, etc.)

  - Use `petsc4py` to access PETSc functionality from Python (built on `mpi4py`)

  - Walk through sample code demonstrating these methods in real applications

- Prerequisites: Python and MPI environment ready; familiarity with Python, NumPy, and basic MPI concepts is helpful
 


## Why parallel computing matters

- Parallel computing lets you tackle simulations and datasets that exceed single-machine memory or time limits (higher-resolution PDEs, larger ML models, whole-genome analyses). 

- Distributing work across many cores or nodes reduces wall-clock time, enabling quicker iterations and more experiments. 
  
- This makes possible ensemble runs, uncertainty quantification, parameter sweeps, and higher-fidelity models that reveal phenomena single-node runs cannot.

- Using cluster resources effectively (many moderate-cost nodes) can be cheaper and more energy-efficient than a single oversized machine. 

- Scalable pipelines and reproducible deployments on HPC clusters support multi-user workflows and reliable, automated processing in research environments.

- Datasets and simulations now outgrow a single CPU core. 

- Hardware roadmaps add cores, cache, and accelerators rather than higher clocks. Parallel workflows turn extra hardware into faster, reproducible results. 

- Everyday laptops and desktops ship with multi-core CPUs, so modernizing code to exploit those cores is increasingly necessary.


## Key Concepts

- Shared-memory vs distributed-memory (single workstation vs cluster).

- Embarrassingly parallel vs tightly coupled workloads.

- Speedup estimates
  - Overall speedup is limited by the fraction of the program that must run serially (Amdahl's Law): even if the parallel portion is sped up arbitrarily, the serial part bounds the maximum achievable speedup.



**Embarassingly parallel example**

Climate researchers run 1,000 weather ensemble members independently to bracket uncertainty. Each simulation uses different initial conditions, needs **zero communication**, and can saturate every core on a workstation or a cluster queue.

**Tightly coupled example**

The same research group runs a 3D Navier–Stokes solver where every timestep must exchange boundary values with neighboring subdomains. Latency and bandwidth now determine how fast those tightly coupled processes can march forward together.


## Parallel Models

### Shared memory

- threads share RAM on one machine

- Python tools: `threading`, `concurrent.futures`, `numpy`, `numba`.

### Distributed memory

- processes own RAM and exchange messages

- Requires explicit communication among processes

- a.k.a. MPI-style parallelism

- Python tools: `mpi4py`, `petsc4py`, MPI-enabled `Dask`.



### Hybrid model

- Shared-memory and distributed-memory models are not mutually exclusive — they are frequently combined in real applications. 

- Use shared-memory parallelism (threads, OpenMP, or multiprocessing with shared arrays) to parallelize work within a single compute node (many cores), and 

- Use distributed-memory parallelism (MPI, distributed Dask, PETSc) to scale across multiple nodes in an HPC cluster. 

- Hybrid approaches (e.g., MPI between nodes + multithreading or vectorized libraries within a node) are common and often give the best performance on modern clusters.


We will focus on MPI-style parallelism in this workshop. 

`mpi4py` is our Python's bridge to distributed-memory computing.



## Examples of MPI applications

- 3D MRI/CT reconstruction pipelines that distribute FFTs and solvers.
- Cardiac and vascular simulations using PETSc domain decomposition.
- Large molecular dynamics workloads (NAMD, GROMACS) with MPI force exchange.
- Large-scale (fine resolution, large region) earthquake simulations.
- Cryo-EM processing pipelines (RELION, cryoSPARC) accelerating EM refinement.
- MPI-enabled sequence alignment and multi-omics analytics (HipMer, mpiBLAST).

And many other domain-specific, tightly coupled or data-parallel workflows.


## Takeaways

Use the right mix: combine shared-memory parallelism inside a node with distributed-memory (MPI) across nodes when appropriate. Match the parallel strategy to where the data lives and how tasks communicate (data locality matters). Start with a simple model, profile early, and iterate. Profiling catches I/O and communication bottlenecks before large-scale runs.

Prefer high-level, well-supported tools: `numpy`, `numba`, or OpenMP for per-node performance; `mpi4py`, `petsc4py`, or Dask for scaling across nodes. Test and scale incrementally: verify correctness and performance on a single node before moving to multi-node runs.
