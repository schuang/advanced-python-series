# Vibrating Bridge Mode (Eigenvalue) using PETsc/SLEPc

## Eigenvalue Problems

The eigenvalue problem seeks non-zero \(x\) and scalar \(\lambda\) satisfying

$$ A x = \lambda x $$

where \(\lambda\) is an eigenvalue (scaling factor) and \(x\) the eigenvector (direction). In structural dynamics, eigenvalues and eigenvectors predict vibration modes. For bridge dynamics, \(\lambda\) gives the squared vibration frequency while \(x\) describes the deflection shape for that mode. The vector \(x\) carries one entry per degree of freedom, so its length equals the number of mesh nodes or DOFs. Realistic meshes often exceed millions of unknowns, requiring eigenproblems to be distributed across MPI ranks. The lowest eigenpair approximates the fundamental frequency and shape, which matters for safety analysis of bridges, towers, and aircraft. PETSc with SLEPc provides scalable eigensolvers, accessible in Python through `petsc4py` and `slepc4py`.

## Problem Setup

We model the bridge deck as a rectangular plate with fixed edges. Finite differences discretize the Laplacian, assembling a stiffness matrix \(K\), while the cell areas fill a lumped mass matrix \(M\). This yields the generalized eigenproblem

$$
K \, x = \lambda M \, x, \quad f = \frac{\sqrt{\lambda}}{2\pi}.
$$

Here \(K\) and \(M\) form a generalized mass–spring system where \(\lambda\) corresponds to squared angular frequency \(\omega^2\); taking $\sqrt{\lambda}$ recovers $\omega$. Converting angular frequency to Hertz via $f = \omega / (2\pi)$ gives $\sqrt{\lambda} / (2\pi)$ as an approximation of the bridge's vibration frequency.

The script `examples/09_petsc_eigen_bridge.py` follows these steps:
1. Partition interior grid points across MPI ranks.
2. Assemble `Mat` objects for \(K\) (5-point stencil) and \(B\) (mass).
3. Configure SLEPc `EPS` to request the smallest real eigenpair.
4. Gather the eigenvector, reshape back to the 2D grid, and print a cross-section of the mode shape.

## Key SLEPc Steps

```python
from petsc4py import PETSc
from slepc4py import SLEPc

A = PETSc.Mat().createAIJ(size=n, nnz=5)      # stiffness
B = PETSc.Mat().createAIJ(size=n, nnz=5)      # mass
# ... assemble finite-difference Laplacian with boundary conditions ...

eps = SLEPc.EPS().create()
eps.setOperators(A, B)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
eps.solve()

nev = eps.getConverged()
vr, vi = A.createVecs()
eps.getEigenpair(0, vr, vi)
```

The eigenvector `vr` can be reshaped into the 2D grid to visualize the bridge deck deflection. Larger grids scale across MPI ranks automatically.

## Requirements

You need PETSc compiled with SLEPc support (`--with-slepc=1` or via package managers that bundle SLEPc). Install `slepc4py` alongside `petsc4py`. Run the demo with `mpirun -n 4 python examples/09_petsc_eigen_bridge.py`. Optionally, add `matplotlib` or `pyvista` to visualize the mode surface.

## Installation Notes

For setup steps, see **25-environment.md** for combined MPI/PETSc/SLEPc installation instructions, platform specifics, and verification commands. Use the same PETSc/SLEPc build across all nodes (or rely on cluster modules) to avoid mismatched libraries.

## Extending the Example

Request multiple eigenpairs with `eps.setDimensions(nevs=5)`. Explore damping via quadratic eigenvalue problems. Swap finite differences for PETSc DMPlex finite-element assembly.
