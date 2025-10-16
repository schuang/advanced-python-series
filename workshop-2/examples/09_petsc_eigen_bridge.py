"""Vibrating bridge mode via PETSc/SLEPc eigensolver."""

import math
from typing import Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc


def build_matrices(
    nx: int, ny: int, length: float = 100.0, width: float = 10.0
) -> Tuple[PETSc.Mat, PETSc.Mat]:
    """Assemble stiffness (A) and mass (B) matrices for a rectangular plate."""

    comm = PETSc.COMM_WORLD
    n = nx * ny
    A = PETSc.Mat().createAIJ([n, n], nnz=5, comm=comm)
    B = PETSc.Mat().createAIJ([n, n], nnz=1, comm=comm)

    hx = length / (nx + 1)
    hy = width / (ny + 1)
    sx = 1.0 / (hx * hx)
    sy = 1.0 / (hy * hy)

    A.setUp()
    B.setUp()

    start, end = A.getOwnershipRange()

    for row in range(start, end):
        ix = row // ny
        iy = row % ny

        diag = 2.0 * (sx + sy)
        A.setValue(row, row, diag)

        if ix > 0:
            A.setValue(row, row - ny, -sx)
        if ix < nx - 1:
            A.setValue(row, row + ny, -sx)
        if iy > 0:
            A.setValue(row, row - 1, -sy)
        if iy < ny - 1:
            A.setValue(row, row + 1, -sy)

        mass = hx * hy
        B.setValue(row, row, mass)

    A.assemblyBegin()
    A.assemblyEnd()
    B.assemblyBegin()
    B.assemblyEnd()

    return A, B


def solve_bridge_mode(nx: int, ny: int) -> None:
    """Solve for the smallest vibration mode of the rectangular plate."""

    comm = PETSc.COMM_WORLD
    mpicomm = comm.tompi4py()
    rank = mpicomm.Get_rank()

    A, B = build_matrices(nx, ny)

    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(A, B)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    # Use a target-based selection for shift-and-invert spectral transform
    # (required when ST type is sinvert).
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(0.0)
    eps.setDimensions(nev=3)
    eps.setTolerances(tol=1e-9, max_it=200)
    # Use a shift-and-invert spectral transformation with an iterative linear
    # solver to avoid requiring a parallel direct LU factorization (which may
    # not be available in minimal PETSc builds). These settings are portable
    # across most PETSc/SLEPc installations.
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    # Use a Krylov solver for the inner linear solves with a simple preconditioner
    ksp = st.getKSP()
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.BJACOBI)

    eps.solve()

    nconv = eps.getConverged()

    if nconv == 0:
        if rank == 0:
            print("No eigenpairs converged.")
        return

    vr, vi = A.createVecs()
    eps.getEigenpair(0, vr, vi)
    eigval = eps.getEigenvalue(0)

    frequency = math.sqrt(eigval) / (2.0 * math.pi)

    local_mode = vr.getArray().copy()
    gathered = mpicomm.gather(local_mode, root=0)

    if rank == 0:
        mode = np.concatenate(gathered)
        mode /= np.max(np.abs(mode))
        mode_grid = mode.reshape(nx, ny)

        print("--- SLEPc Bridge Mode ---")
        print(f"Grid: {nx} x {ny} interior points")
        print(f"Smallest eigenvalue (lambda): {eigval:.6e}")
        print(f"Approximate frequency (Hz): {frequency:.3f}")
        print("Mode shape (central cross-section):")
        mid = ny // 2
        print(mode_grid[:, mid])


def main() -> None:
    solve_bridge_mode(nx=40, ny=12)


if __name__ == "__main__":
    main()
