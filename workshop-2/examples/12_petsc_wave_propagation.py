"""MPI-distributed 2D wave propagation (earthquake-inspired) using petsc4py."""

import argparse
import math

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D acoustic wave propagation with PETSc.")
    parser.add_argument("--nx", type=int, default=200, help="Grid points in X (east-west streets)")
    parser.add_argument("--ny", type=int, default=120, help="Grid points in Y (north-south streets)")
    parser.add_argument("--length-x", type=float, default=40_000.0, help="Domain length in X (meters)")
    parser.add_argument("--length-y", type=float, default=24_000.0, help="Domain length in Y (meters)")
    parser.add_argument("--c", type=float, default=2_500.0, help="Wave speed (m/s)")
    parser.add_argument("--dt", type=float, default=0.002, help="Time step (seconds)")
    parser.add_argument("--steps", type=int, default=1100, help="Number of time steps to run")
    parser.add_argument("--source-amp", type=float, default=1.0, help="Source amplitude")
    parser.add_argument("--source-freq", type=float, default=1.0, help="Source dominant frequency (Hz)")
    parser.add_argument("--output-interval", type=int, default=200, help="Steps between progress prints")
    parser.add_argument("--absorb-width", type=int, default=15, help="Absorbing boundary width (grid cells)")
    parser.add_argument("--absorb-strength", type=float, default=4.0, help="Absorbing layer strength")
    args, petsc_opts = parser.parse_known_args()

    # petsc4py Options has no `process` method in newer versions. Insert any
    # unknown args into PETSc's options database so PETSc flags are recognized.
    if petsc_opts:
        try:
            PETSc.Options().insertString(' '.join(petsc_opts))
        except Exception:
            opts = PETSc.Options()
            for token in petsc_opts:
                opts.insertString(token)
    return args


def create_dm(nx: int, ny: int) -> PETSc.DMDA:
    dm = PETSc.DMDA().create(
        [nx, ny],
        dof=1,
        stencil_width=1,
        boundary_type=(PETSc.DMDA.BoundaryType.NONE, PETSc.DMDA.BoundaryType.NONE),
        stencil_type=PETSc.DMDA.StencilType.STAR,
        comm=PETSc.COMM_WORLD,
    )
    return dm


def build_damping(dm: PETSc.DMDA, nx: int, ny: int, width: int, strength: float) -> PETSc.Vec:
    damping = dm.createGlobalVec()

    # Use the Vec.getArray() interface and ownership ranges so we avoid
    # DM-specific restore calls that may not exist in all petsc4py versions.
    start, end = damping.getOwnershipRange()
    local_size = end - start
    local_arr = damping.getArray(readonly=False)

    # Map global linear index -> 2D (i,j) assuming row-major ordering with
    # global_index = j * nx + i
    for local_idx in range(local_size):
        global_idx = start + local_idx
        i = global_idx % nx
        j = global_idx // nx
        dist = min(i, nx - 1 - i, j, ny - 1 - j)
        if dist < width:
            ratio = (width - dist) / max(width, 1)
            factor = math.exp(-strength * ratio * ratio)
        else:
            factor = 1.0
        local_arr[local_idx] = factor

    return damping


def compute_laplacian(
    dm: PETSc.DMDA,
    u_global: PETSc.Vec,
    lap_global: PETSc.Vec,
    dx: float,
    dy: float,
) -> None:
    local = dm.createLocalVec()
    dm.globalToLocal(u_global, local)
    arr_local = dm.getVecArray(local)
    # Create a local vector to hold laplacian values (includes ghost points)
    local_lap = dm.createLocalVec()
    arr_lap = dm.getVecArray(local_lap)
    (xs, xe), (ys, ye) = dm.getRanges()
    nx, ny = dm.getSizes()

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    for i in range(xs, xe):
        for j in range(ys, ye):
            u_c = arr_local[i, j]
            # Dirichlet boundary (outside domain -> 0)
            u_e = arr_local[i + 1, j] if i + 1 < nx else 0.0
            u_w = arr_local[i - 1, j] if i - 1 >= 0 else 0.0
            u_n = arr_local[i, j + 1] if j + 1 < ny else 0.0
            u_s = arr_local[i, j - 1] if j - 1 >= 0 else 0.0
            lap = (u_e - 2.0 * u_c + u_w) * inv_dx2 + (u_n - 2.0 * u_c + u_s) * inv_dy2
            arr_lap[i, j] = lap

    # Move local lap values to the global lap vector
    dm.localToGlobal(local_lap, lap_global, addv=PETSc.InsertMode.INSERT_VALUES)


def add_source(
    vec: PETSc.Vec,
    dm: PETSc.DMDA,
    node: tuple[int, int],
    amplitude: float,
    freq: float,
    t: float,
    t0: float,
    sigma: float,
) -> None:
    vec.set(0.0)
    value = amplitude * math.exp(-((t - t0) ** 2) / (2.0 * sigma * sigma)) * math.sin(2.0 * math.pi * freq * t)
    ix, iy = node
    nx, _ = dm.getSizes()
    global_index = iy * nx + ix
    start, end = vec.getOwnershipRange()
    if start <= global_index < end:
        vec.setValue(global_index, value)
    vec.assemblyBegin()
    vec.assemblyEnd()


def main() -> None:
    args = parse_args()
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()

    if args.nx < 10 or args.ny < 10:
        raise ValueError("nx and ny must each be >= 10.")

    dx = args.length_x / (args.nx - 1)
    dy = args.length_y / (args.ny - 1)

    stability = args.c * args.dt * math.sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy))
    if rank == 0:
        PETSc.Sys.Print(
            f"Grid: {args.nx} x {args.ny}, dx={dx:.2f} m, dy={dy:.2f} m, dt={args.dt:.4f} s, CFL={stability:.3f}"
        )
        if stability >= 1.0:
            PETSc.Sys.Print("Warning: time step may be unstable (CFL >= 1). Consider reducing dt or increasing resolution.")

    dm = create_dm(args.nx, args.ny)

    u_prev = dm.createGlobalVec()
    u_curr = dm.createGlobalVec()
    u_next = dm.createGlobalVec()
    lap = dm.createGlobalVec()
    source_vec = dm.createGlobalVec()
    damping = build_damping(dm, args.nx, args.ny, args.absorb_width, args.absorb_strength)

    u_prev.set(0.0)
    u_curr.set(0.0)

    dt2c2 = (args.c * args.dt) ** 2
    dt_sq = args.dt ** 2
    t0 = 1.5 / max(args.source_freq, 1e-6)
    sigma = t0 / 3.0

    source_node = (args.nx // 5, args.ny // 2)
    receiver_node = (4 * args.nx // 5, args.ny // 2)
    receiver_index = receiver_node[1] * args.nx + receiver_node[0]

    (_, _), (ys, ye) = dm.getRanges()

    max_amplitude = 0.0
    receiver_trace = []

    for step in range(args.steps):
        time = step * args.dt

        compute_laplacian(dm, u_curr, lap, dx, dy)
        add_source(
            source_vec,
            dm,
            source_node,
            args.source_amp,
            args.source_freq,
            time,
            t0,
            sigma,
        )

        # Use DM local vectors to safely access a 2D array view (including
        # ghost points) for each global vector. This avoids mismatched global
        # array views that can cause segmentation faults across petsc4py
        # versions/builds.
        local_curr = dm.createLocalVec()
        local_prev = dm.createLocalVec()
        local_next = dm.createLocalVec()
        local_lap = dm.createLocalVec()
        local_source = dm.createLocalVec()
        local_damp = dm.createLocalVec()

        dm.globalToLocal(u_curr, local_curr)
        dm.globalToLocal(u_prev, local_prev)
        dm.globalToLocal(u_next, local_next)
        dm.globalToLocal(lap, local_lap)
        dm.globalToLocal(source_vec, local_source)
        dm.globalToLocal(damping, local_damp)

        arr_curr = dm.getVecArray(local_curr)
        arr_prev = dm.getVecArray(local_prev)
        arr_next = dm.getVecArray(local_next)
        arr_lap = dm.getVecArray(local_lap)
        arr_source = dm.getVecArray(local_source)
        arr_damp = dm.getVecArray(local_damp)
        (xs, xe), (ys, ye) = dm.getRanges()

        for i in range(xs, xe):
            for j in range(ys, ye):
                arr_next[i, j] = (
                    2.0 * arr_curr[i, j]
                    - arr_prev[i, j]
                    + dt2c2 * arr_lap[i, j]
                    + dt_sq * arr_source[i, j]
                )
                arr_next[i, j] *= arr_damp[i, j]
        
        # Push local_next back to its global Vec
        dm.localToGlobal(local_next, u_next)
        # Local vectors and arrays will be garbage collected automatically

        # Shift time levels.
        u_prev, u_curr, u_next = u_curr, u_next, u_prev

        amp = u_curr.norm(norm_type=PETSc.NormType.NORM_INFINITY)
        max_amplitude = max(max_amplitude, amp)

        start, end = u_curr.getOwnershipRange()
        if start <= receiver_index < end:
            receiver_trace.append(u_curr.getValue(receiver_index))
        else:
            receiver_trace.append(None)

        if args.output_interval > 0 and (step % args.output_interval == 0):
            if rank == 0:
                PETSc.Sys.Print(f"Step {step:5d} / {args.steps}, time {time:7.3f} s, |u|_inf = {amp:.3e}")

    # Consolidate receiver trace to rank 0.
    trace = None
    if rank == 0:
        trace = np.zeros(args.steps)
    local_trace = np.array([val if val is not None else 0.0 for val in receiver_trace], dtype=np.float64)
    MPI.COMM_WORLD.Reduce(local_trace, trace, op=MPI.SUM, root=0)

    if rank == 0:
        PETSc.Sys.Print(f"Simulation complete. Max amplitude = {max_amplitude:.3e}")
        max_receiver = np.max(np.abs(trace))
        PETSc.Sys.Print(f"Receiver peak displacement at {receiver_node}: {max_receiver:.3e}")
        PETSc.Sys.Print("Receiver sample (first 10 values):")
        PETSc.Sys.Print(trace[:10])
        PETSc.Sys.Print("Receiver sample (peak region around max):")
        max_idx = np.argmax(np.abs(trace))
        start_idx = max(0, max_idx - 5)
        end_idx = min(len(trace), max_idx + 6)
        PETSc.Sys.Print(f"  Steps {start_idx}-{end_idx-1}: {trace[start_idx:end_idx]}")
        PETSc.Sys.Print(f"  Max at step {max_idx}, time = {max_idx * args.dt:.3f} s")


if __name__ == "__main__":
    main()
