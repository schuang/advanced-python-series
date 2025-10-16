"""Solve a power-network Laplacian using PETSc."""

import argparse
from dataclasses import dataclass

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI


@dataclass
class Grid:
    nx: int
    ny: int

    @property
    def size(self) -> int:
        return self.nx * self.ny

    def node_id(self, ix: int, iy: int) -> int:
        return iy * self.nx + ix

    def neighbors(self, ix: int, iy: int):
        if ix > 0:
            yield ix - 1, iy
        if ix < self.nx - 1:
            yield ix + 1, iy
        if iy > 0:
            yield ix, iy - 1
        if iy < self.ny - 1:
            yield ix, iy + 1


def build_laplacian(grid: Grid, conductance: float) -> PETSc.Mat:
    n = grid.size
    A = PETSc.Mat().createAIJ([n, n], nnz=5, comm=PETSc.COMM_WORLD)
    A.setUp()

    start, end = A.getOwnershipRange()

    for row in range(start, end):
        ix = row % grid.nx
        iy = row // grid.nx
        degree = 0.0

        for jx, jy in grid.neighbors(ix, iy):
            col = grid.node_id(jx, jy)
            A.setValue(row, col, -conductance)
            degree += conductance

        A.setValue(row, row, degree)

    A.assemblyBegin()
    A.assemblyEnd()
    return A


def apply_dirichlet(mat: PETSc.Mat, rhs: PETSc.Vec, node: int, value: float) -> None:
    """Impose v[node] = value by modifying matrix row/col and rhs."""
    mat.zeroRowsColumns(node, diag=1.0, b=rhs, x=rhs)
    rhs.setValue(node, value)
    rhs.assemblyBegin()
    rhs.assemblyEnd()


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Laplacian on a grid graph.")
    parser.add_argument("--nx", type=int, default=40, help="Grid width (streets)")
    parser.add_argument("--ny", type=int, default=20, help="Grid height (avenues)")
    parser.add_argument("--load", type=float, default=0.6, help="Total load magnitude (per unit)")
    parser.add_argument("--g", type=float, default=1.0, help="Conductance per line")
    args, unknown = parser.parse_known_args()

    # petsc4py's Options object does not provide a `process` method in
    # newer versions. Insert any unknown args into PETSc's options database
    # so that PETSc-related command-line flags (e.g. -ksp_type) are seen.
    if unknown:
        # unknown is a list like ['--foo', 'val', '-ksp_type', 'cg']
        # PETSc expects a single string; join with spaces.
        try:
            PETSc.Options().insertString(' '.join(unknown))
        except Exception:
            # Fallback: try to insert each token individually.
            opts = PETSc.Options()
            for token in unknown:
                opts.insertString(token)

    grid = Grid(args.nx, args.ny)

    A = build_laplacian(grid, args.g)

    b = PETSc.Vec().createMPI(grid.size, comm=PETSc.COMM_WORLD)
    b.set(0.0)

    # Inject generation at central plant (top-left quadrant) and sink loads elsewhere.
    rank = PETSc.COMM_WORLD.getRank()

    gen_node = grid.node_id(grid.nx // 4, grid.ny // 4)
    load_nodes = [
        grid.node_id(3 * grid.nx // 4, grid.ny // 4),
        grid.node_id(grid.nx // 3, 2 * grid.ny // 3),
        grid.node_id(3 * grid.nx // 4, 3 * grid.ny // 4),
    ]

    gen_power = args.load
    load_share = -args.load / len(load_nodes)

    if b.getOwnershipRange()[0] <= gen_node < b.getOwnershipRange()[1]:
        b.setValue(gen_node, gen_power)

    for node in load_nodes:
        if b.getOwnershipRange()[0] <= node < b.getOwnershipRange()[1]:
            b.setValue(node, load_share)

    b.assemblyBegin()
    b.assemblyEnd()

    # Apply Dirichlet boundary: main substation at 1.0 per unit, far corner at 0.0.
    apply_dirichlet(A, b, gen_node, 1.0)
    reference_node = grid.node_id(grid.nx - 1, grid.ny - 1)
    apply_dirichlet(A, b, reference_node, 0.0)

    x = PETSc.Vec().createMPI(grid.size, comm=PETSc.COMM_WORLD)
    x.set(0.0)

    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("gamg")
    
    # Set up monitoring to display convergence history
    ksp.setMonitor(lambda ksp, its, rnorm: PETSc.Sys.Print(f"  {its} KSP Residual norm {rnorm:.6e}"))
    
    ksp.setFromOptions()
    ksp.solve(b, x)

    if rank == 0:
        its = ksp.getIterationNumber()
        reason = ksp.getConvergedReason()
        print(f"KSP converged in {its} iterations (reason={reason})")

    # Compute line currents for selected edges. Use Vec.getValue to fetch
    # global entries (safer for parallel ownership) rather than indexing the
    # local array which only contains the ownership range for this rank.
    # Gather the distributed vector to rank 0 as a full NumPy array so we can
    # sample arbitrary global indices safely.
    local_start, local_end = x.getOwnershipRange()
    local_arr = x.getArray(readonly=True)
    # Each rank sends a tuple (start, array) to root; we'll reconstruct full array on root.
    local_payload = (local_start, np.array(local_arr, copy=True))
    gathered = MPI.COMM_WORLD.gather(local_payload, root=0)

    if rank == 0:
        # Reconstruct full array of size grid.size
        full = np.empty(grid.size, dtype=np.float64)
        for start, arr_part in gathered:
            full[start : start + arr_part.size] = arr_part

        print("Sampled node voltages (per unit):")
        samples = [
            ("Plant", full[gen_node]),
            ("Downtown", full[load_nodes[0]]),
            ("Midtown", full[load_nodes[1]]),
            ("Uptown", full[load_nodes[2]]),
            ("Reference", full[reference_node]),
        ]
        for name, val in samples:
            print(f"  {name:<10s} {val: .4f}")

        print("Sampled line currents (conductance * voltage drop):")

        def current(n1: int, n2: int) -> float:
            return args.g * (full[n1] - full[n2])

        edges_to_report = [
            (gen_node, load_nodes[0]),
            (gen_node, load_nodes[1]),
            (load_nodes[1], load_nodes[2]),
            (load_nodes[0], reference_node),
        ]
        for n1, n2 in edges_to_report:
            print(f"  ({n1:4d}->{n2:4d}) current = {current(n1, n2): .4f} per unit")


if __name__ == "__main__":
    main()
