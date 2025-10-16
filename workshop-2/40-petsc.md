# PETSc

## What is PETSc

The **Portable, Extensible Toolkit for Scientific Computation (PETSc)** is a comprehensive software library for the parallel numerical solution of partial differential equations (PDEs) and related problems in scientific computing. Developed at Argonne National Laboratory, PETSc provides a unified framework that abstracts away the complexities of distributed-memory parallelism while exposing high-performance numerical algorithms.

### From MPI to PETSc: Why the Abstraction Matters

With plain MPI, implementing a parallel PDE solver requires manually managing:

**1. Domain decomposition and ghost cells:**

```python
# Plain MPI: Manual halo exchange for a 2D Laplacian stencil
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Determine local subdomain boundaries
local_nx, local_ny = compute_local_size(rank, size)
ghost_cells = allocate_with_halo(local_nx, local_ny, halo_width=1)

# Exchange boundary data with 4 neighbors (N, S, E, W)
send_north = ghost_cells[1, :]
send_south = ghost_cells[-2, :]
comm.Sendrecv(send_north, dest=rank_north, recvbuf=recv_south, source=rank_south)
comm.Sendrecv(send_south, dest=rank_south, recvbuf=recv_north, source=rank_north)
# ... repeat for east/west boundaries ...
```

**With PETSc DMDA (Distributed Multi-dimensional Array):**

```python
# PETSc: Automatic halo exchange
from petsc4py import PETSc

da = PETSc.DMDA().create(dim=2, sizes=(nx, ny), stencil_width=1, comm=comm)
da.setFromOptions()

# One-line halo exchange:
da.globalToLocal(global_vec, local_vec)  # All neighbor communication handled internally
```

**2. Sparse matrix assembly with correct global indexing:**

```python
# Plain MPI: Track global indices, communicate off-processor entries
local_rows = rows_for_rank(rank, size)
A_local = scipy.sparse.lil_matrix((len(local_rows), global_N))

for i in local_rows:
    # Compute stencil entries for row i
    for j in stencil_neighbors(i):
        A_local[local_to_global(i), j] = compute_coefficient(i, j)

# Convert to parallel format, communicate off-processor data...
# (Requires custom CSR distribution + MPI_Alltoallv communication)
```

**With PETSc Mat:**

```python
# PETSc: Distributed sparse matrix with automatic communication
A = PETSc.Mat().createAIJ([global_N, global_N], comm=comm)
A.setFromOptions()
A.setUp()

istart, iend = A.getOwnershipRange()  # My row range
for i in range(istart, iend):
    # Set values; PETSc handles off-processor communication
    A.setValues(i, neighbor_indices, neighbor_coefficients)

A.assemblyBegin()  # Start communication
A.assemblyEnd()    # Complete assembly (MPI under the hood)
```

**3. Iterative solver implementation:**

```python
# Plain MPI: Implement CG from scratch with parallel dot products
r = b - A @ x
p = r.copy()
for iteration in range(max_iter):
    alpha = comm.allreduce(np.dot(r, r), op=MPI.SUM) / comm.allreduce(np.dot(p, A @ p), op=MPI.SUM)
    x += alpha * p
    r_new = r - alpha * (A @ p)
    beta = comm.allreduce(np.dot(r_new, r_new), op=MPI.SUM) / comm.allreduce(np.dot(r, r), op=MPI.SUM)
    p = r_new + beta * p
    r = r_new
    # Missing: preconditioning, convergence checks, restarts...
```

**With PETSc KSP:**

```python
# PETSc: Production solver with one setup call
ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A)
ksp.setType('cg')                      # Or gmres, bicgstab, etc.
ksp.getPC().setType('gamg')            # Algebraic multigrid preconditioner
ksp.setFromOptions()                   # Override via -ksp_type, -pc_type at runtime

ksp.solve(b, x)                        # Solves Ax = b in parallel
```

### Key Abstractions and Their Benefits

| PETSc Component | Abstraction | Plain MPI Equivalent | Complexity Reduction |
|-----------------|-------------|---------------------|---------------------|
| **Vec** | Distributed vector with ghost cells | Manual partitioning + halo exchange code | ~100 lines → 5 lines |
| **Mat** | Distributed sparse matrix | Custom CSR distribution + communication | ~200 lines → 10 lines |
| **DM** (DMDA/DMPlex) | Structured/unstructured grid topology | Manual neighbor tracking, ghost regions | ~300 lines → 15 lines |
| **KSP** | Krylov subspace solvers + preconditioners | CG/GMRES from scratch + parallel dot products | ~500 lines → 20 lines |
| **SNES** | Nonlinear solvers (Newton, quasi-Newton) | Custom Newton with line search, Jacobian assembly | ~800 lines → 30 lines |
| **TAO** | Optimization (bound-constrained, PDE-constrained) | Custom gradient descent/L-BFGS + parallel reductions | ~1000 lines → 40 lines |

### Algorithmic Richness

PETSc ships with production-quality implementations of algorithms that would take months to develop from MPI primitives:

**Linear solvers (KSP):**

- Direct: LU, Cholesky (via external packages: MUMPS, SuperLU_DIST)
- Krylov: CG, GMRES, BiCGStab, MINRES, TFQMR, Richardson
- Specialized: QMR for complex symmetric, LGMRES with restart

**Preconditioners (PC):**

- Algebraic multigrid: GAMG (PETSc-native), BoomerAMG (HYPRE)
- Domain decomposition: Block Jacobi, additive/multiplicative Schwarz
- Incomplete factorizations: ILU(k), ICC(k)
- Physics-based: Fieldsplit (segregated fluid-structure, Stokes saddle point)

**Nonlinear solvers (SNES):**

- Newton methods with line search or trust region
- Quasi-Newton: Broyden, L-BFGS
- Nonlinear CG, Anderson mixing (for fixed-point problems)

**Optimization (TAO):**

- Unconstrained: Newton, L-BFGS, conjugate gradient
- Bound-constrained: BLMVM, TRON (trust region Newton)
- PDE-constrained: Reduced-space methods, adjoint-driven optimization

**Time integration (TS):**

- Explicit: Forward Euler, Runge-Kutta (2-5 stages)
- Implicit: Backward Euler, BDF, Theta methods
- IMEX: Implicit-explicit splittings for stiff-nonstiff systems

### Runtime Configurability

Unlike custom MPI code where algorithm choices are hard-coded, PETSc allows runtime tuning without recompilation:

```bash
# Try different solver combinations from command line:
mpiexec -n 16 python my_solver.py -ksp_type gmres -pc_type gamg
mpiexec -n 16 python my_solver.py -ksp_type bcgs -pc_type ilu
mpiexec -n 16 python my_solver.py -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps

# Monitor convergence:
mpiexec -n 16 python my_solver.py -ksp_monitor -ksp_view

# Profile performance:
mpiexec -n 16 python my_solver.py -log_view
```

This design philosophy—compose algorithms from command-line options—accelerates experimentation and makes production codes adaptable to different hardware (CPUs, GPUs) and problem scales without source changes.

### The Complete Scientific Computing Stack

PETSc provides an **end-to-end HPC toolkit** that covers the full spectrum from problem setup to solution:

**Data structures and topology:**

- `Vec` / `Mat`: Distributed linear algebra primitives
- `DM` (DMDA for structured grids, DMPlex for unstructured meshes): Automatic domain decomposition, ghost cell management, and global-to-local mappings

**Solvers and algorithms:**

- `KSP`: 40+ iterative linear solvers with 30+ preconditioners
- `SNES`: Nonlinear equation systems (Newton, quasi-Newton, nonlinear CG)
- `TS`: Time integration for ODEs and time-dependent PDEs
- `TAO`: Optimization (unconstrained, bound-constrained, PDE-constrained)
- `SLEPc` (extension): Eigenvalue problems for modal analysis

**Scalability:**

The MPI-native design means codes written with `petsc4py` scale seamlessly from **laptop prototyping** (single process) to **leadership-class supercomputers** (100,000+ cores) without modification. The same Python script that solves a 1000-element problem on your workstation can tackle 100-million-element problems on a production cluster by simply changing `mpiexec -n`.

**Real-world impact:**

- **Climate models:** DMDA manages structured atmospheric grids; KSP solves pressure Poisson equations.
- **Subsurface flow & geomechanics:** DMPlex handles unstructured tetrahedral meshes for reservoir simulation.
- **Cardiac electrophysiology:** DMPlex + SNES for reaction-diffusion systems on anatomical heart geometries.
- **Fusion plasma (MHD):** SNES solves coupled magnetohydrodynamic equations in tokamak simulations.
- **Aircraft structural analysis:** SLEPc computes flutter modes and eigenfrequencies.
- **Medical imaging:** TAO performs MRI parameter estimation and image reconstruction.
- **Power grid optimization:** TAO handles optimal power flow with tens of thousands of constraints.

## Quick Start

Install MPI, PETSc, SLEPc, and their Python bindings. Initialize once: `from petsc4py import PETSc`; `PETSc.Sys.Initialize()` (auto via import in new versions). Core layers to remember: `Vec`/`Mat`, `KSP`, `SNES`, `TAO`, `DM`.

## Core Patterns

Assemble distributed matrices and vectors collaboratively (e.g., structural stiffness). Select solvers and preconditioners per workload (AMG for groundwater, block PC for Navier–Stokes). Tune behavior with runtime flags (`-ksp_type`, `-pc_type`, `-snes_type`, `-tao_type`), no recompiles. `petsc4py` mirrors the C API while speaking NumPy buffers and `mpi4py` communicators. Hands-on: `examples/08_petsc_tao_t2_fit.py` uses TAO to recover MRI T2 decay parameters.

## Practical Guidance

Prototype small meshes before full-scale (coarse cardiac grid to full anatomy). Log convergence and timings every run. Swap solver pipelines via options without recompiling.

## Discussion

What domain decompositions fit your models (DMDA vs DMPlex)? What are your favorite PETSc examples or docs to bookmark? When do you promote from `KSP` to `SNES` or `TAO` (MRI inverse problems, coupled climate)?

## Architecture Overview

`Vec` and `Mat`: distributed linear algebra for CFD, seismic inversion. `KSP`: Krylov methods with preconditioners (GAMG, block PC) for combustion, ocean flow. `SNES`: nonlinear engines for phase-field materials, biomechanics elasticity. `SLEPc`: eigenvalues for aircraft modes, photonic bands. `TAO`: optimization for medical imaging, climate assimilation, power grids. `DM`: DMDA (structured) and DMPlex (unstructured) handle grids; MPI backbone manages halo exchange.

