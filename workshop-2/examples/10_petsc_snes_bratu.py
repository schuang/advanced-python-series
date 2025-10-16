"""Bratu-type nonlinear heat equation solved with PETSc SNES."""

import math

from petsc4py import PETSc


def create_dm(nx: int, ny: int) -> PETSc.DMDA:
    return PETSc.DMDA().create(
        [nx, ny],
        dof=1,
        stencil_width=1,
        boundary_type=(PETSc.DMDA.BoundaryType.NONE, PETSc.DMDA.BoundaryType.NONE),
        stencil_type=PETSc.DMDA.StencilType.STAR,
    )


def form_function(snes: PETSc.SNES, x: PETSc.Vec, f: PETSc.Vec) -> None:
    dm = snes.getDM()
    local_x = dm.createLocalVec()
    dm.globalToLocal(x, local_x)

    f.set(0.0)

    arr_x = dm.getVecArray(local_x)
    arr_f = dm.getVecArray(f)
    f.set(0.0)

    (xs, xe), (ys, ye) = dm.getRanges()
    nx, ny = dm.sizes

    hx = 1.0 / (nx - 1)
    hy = 1.0 / (ny - 1)
    hx2 = 1.0 / (hx * hx)
    hy2 = 1.0 / (hy * hy)

    lam = snes.getApplicationContext()["lambda"]

    for i in range(xs, xe):
        for j in range(ys, ye):
            u = arr_x[i, j]
            boundary = i == 0 or j == 0 or i == nx - 1 or j == ny - 1
            if boundary:
                arr_f[i, j] = u  # Dirichlet T=0
                continue

            ue = arr_x[i + 1, j]
            uw = arr_x[i - 1, j]
            un = arr_x[i, j + 1]
            us = arr_x[i, j - 1]

            lap = (
                (ue - 2.0 * u + uw) * hx2
                + (un - 2.0 * u + us) * hy2
            )
            source = lam * math.exp(u)
            arr_f[i, j] = -lap - source


def form_jacobian(
    snes: PETSc.SNES,
    x: PETSc.Vec,
    J: PETSc.Mat,
    P: PETSc.Mat,
) -> None:
    dm = snes.getDM()
    local_x = dm.createLocalVec()
    dm.globalToLocal(x, local_x)

    arr_x = dm.getVecArray(local_x)

    (xs, xe), (ys, ye) = dm.getRanges()
    nx, ny = dm.sizes

    hx = 1.0 / (nx - 1)
    hy = 1.0 / (ny - 1)
    hx2 = 1.0 / (hx * hx)
    hy2 = 1.0 / (hy * hy)

    lam = snes.getApplicationContext()["lambda"]

    P.zeroEntries()
    if J.handle != P.handle:
        J.zeroEntries()

    row = PETSc.Mat.Stencil()
    col = PETSc.Mat.Stencil()
    row.k = col.k = 0
    row.c = col.c = 0

    for i in range(xs, xe):
        for j in range(ys, ye):
            row.i, row.j = i, j
            row.k = 0
            boundary = i == 0 or j == 0 or i == nx - 1 or j == ny - 1
            if boundary:
                P.setValueStencil(row, row, 1.0)
                if J.handle != P.handle:
                    J.setValueStencil(row, row, 1.0)
                continue

            u = arr_x[i, j]
            diag = 2.0 * (hx2 + hy2) - lam * math.exp(u)

            neighbors = [
                ((i, j), diag),
                ((i + 1, j), -hx2),
                ((i - 1, j), -hx2),
                ((i, j + 1), -hy2),
                ((i, j - 1), -hy2),
            ]

            for (ii, jj), value in neighbors:
                col.i, col.j = ii, jj
                col.k = 0
                P.setValueStencil(row, col, value)
                if J.handle != P.handle:
                    J.setValueStencil(row, col, value)

    P.assemblyBegin(PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)
    P.assemblyEnd(PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)
    if J.handle != P.handle:
        J.assemblyBegin(PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)
        J.assemblyEnd(PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)


def main() -> None:
    opts = PETSc.Options()
    nx = opts.getInt("nx", 64)
    ny = opts.getInt("ny", 32)
    lam = opts.getReal("lambda", 6.0)

    dm = create_dm(nx, ny)
    snes = PETSc.SNES().create()
    snes.setDM(dm)

    ctx = {"lambda": lam}
    snes.setApplicationContext(ctx)

    snes.setFunction(form_function, dm.createGlobalVec())
    jac = dm.createMatrix()
    # petsc4py expects the Python callback first, then the Jacobian and P mats
    # (callback, J, P). Keep this order to match PETSc's SNES API.
    snes.setJacobian(form_jacobian, jac, jac)

    x = dm.createGlobalVec()
    x.set(0.0)

    # Set up monitoring to display convergence history
    snes.setMonitor(lambda snes, its, norm: PETSc.Sys.Print(f"  {its} SNES Function norm {norm:.6e}"))

    snes.setFromOptions()
    snes.solve(None, x)

    its = snes.getIterationNumber()
    reason = snes.getConvergedReason()

    PETSc.Sys.Print(f"SNES converged in {its} iterations with reason {reason}")
    PETSc.Sys.Print(f"Max temperature rise: {x.max()[1]:.4e}")


if __name__ == "__main__":
    main()
