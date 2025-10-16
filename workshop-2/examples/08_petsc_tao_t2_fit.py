"""PETSc TAO example: fit MRI T2 decay parameters via least squares.

This version uses PETSc Scatter.toAll to replicate the (very small)
parameter vector onto each rank, computes objective and gradient centrally
and writes the local gradient slice back into the PETSc gradient Vec.
"""

import numpy as np
from petsc4py import PETSc


class T2Objective:
    """Least-squares objective for T2 decay fitting."""

    def __init__(self, times: np.ndarray, signal: np.ndarray) -> None:
        self.times = times
        self.signal = signal

    def __call__(self, tao: PETSc.TAO, x: PETSc.Vec, g: PETSc.Vec) -> float:
        # local ownership range
        start, end = x.getOwnershipRange()

        # replicate the small distributed parameter vector to a local sequential
        # Vec on each rank (VecScatter to all)
        vs, seq = PETSc.Scatter.toAll(x)
        vs.scatter(x, seq, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        params = seq.getArray(readonly=True).copy()

        amp, t2 = params
        t2 = max(t2, 1e-6)

        exp_term = np.exp(-self.times / t2)
        pred = amp * exp_term
        residual = pred - self.signal

        # objective
        f = 0.5 * np.dot(residual, residual)

        # full gradient
        full_grad = np.empty_like(params)
        full_grad[0] = np.dot(residual, exp_term)
        full_grad[1] = np.dot(residual, amp * exp_term * self.times / (t2 * t2))

        # write local slice into PETSc gradient Vec
        g_local = g.getArray()
        g_local[:] = full_grad[start:end]

        # cleanup
        vs.destroy()
        seq.destroy()

        return f


def main() -> None:
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()

    # Synthetic spin-echo measurements: S(t) = A * exp(-t / T2)
    times_ms = np.array([10, 20, 40, 60, 80, 120, 160, 200, 240, 280], dtype=np.float64)
    true_amp = 1.0
    true_t2 = 85.0  # ms
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0.0, 0.02, size=times_ms.size)
    signal = true_amp * np.exp(-times_ms / true_t2) + noise

    objective = T2Objective(times_ms, signal)

    # Initial guess: amplitude 0.8, T2 50 ms.
    x = PETSc.Vec().createMPI(2, comm=comm)
    x.setValues(range(2), [0.8, 50.0])
    x.assemble()

    tao = PETSc.TAO().create(comm=comm)
    tao.setType("lmvm")
    tao.setObjectiveGradient(objective, None)
    tao.setFromOptions()
    tao.solve(x)

    # reconstruct global solution via VecScatter-to-all
    vs, seq = PETSc.Scatter.toAll(x)
    vs.scatter(x, seq, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
    solution = seq.getArray().copy()
    vs.destroy()
    seq.destroy()

    final_obj = tao.getObjective()

    if final_obj is None and rank == 0:
        amp_est, t2_est = solution
        t2_est = max(t2_est, 1e-12)
        exp_term = np.exp(-times_ms / t2_est)
        pred = amp_est * exp_term
        residual = pred - signal
        final_obj = 0.5 * np.dot(residual, residual)

    if rank == 0:
        amp_est, t2_est = solution
        print("--- PETSc TAO MRI T2 Fit ---")
        print("True parameters: amplitude = {:.3f}, T2 = {:.1f} ms".format(true_amp, true_t2))
        print("Estimated parameters: amplitude = {:.3f}, T2 = {:.1f} ms".format(amp_est, t2_est))
        print(f"Final objective value: {final_obj:.3e}")


if __name__ == "__main__":
    main()
