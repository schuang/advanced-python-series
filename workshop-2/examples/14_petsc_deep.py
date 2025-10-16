"""PETSc TAO-based data-parallel deep classifier (two hidden layers)."""

import argparse
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep logistic classifier with PETSc TAO.")
    parser.add_argument("--samples", type=int, default=10_000, help="Total samples across all ranks.")
    parser.add_argument("--features", type=int, default=16, help="Number of input features.")
    parser.add_argument("--hidden1", type=int, default=32, help="Hidden layer 1 width.")
    parser.add_argument("--hidden2", type=int, default=16, help="Hidden layer 2 width.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum TAO iterations.")
    parser.add_argument("--batch", type=int, default=512, help="Mini-batch size per rank.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--tao_type", type=str, default="lmvm", help="TAO solver type (lmvm, nls, tron, etc.).")
    args = parser.parse_args()
    return args


def generate_data(
    rank: int,
    size: int,
    samples_total: int,
    features: int,
    hidden1: int,
    hidden2: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if samples_total % size != 0:
        raise ValueError("--samples must be divisible by MPI world size.")
    samples_local = samples_total // size
    rng_global = np.random.default_rng(seed)
    true_W1 = rng_global.normal(scale=1.0, size=(features, hidden1))
    true_b1 = rng_global.normal(scale=0.1, size=(hidden1,))
    true_W2 = rng_global.normal(scale=1.0, size=(hidden1, hidden2))
    true_b2 = rng_global.normal(scale=0.1, size=(hidden2,))
    true_W3 = rng_global.normal(scale=1.0, size=(hidden2, 1))
    true_b3 = rng_global.normal(scale=0.1, size=(1,))

    rng_local = np.random.default_rng(seed + rank * 17)
    X = rng_local.normal(size=(samples_local, features))
    h1 = np.tanh(X @ true_W1 + true_b1)
    h2 = np.tanh(h1 @ true_W2 + true_b2)
    logits = (h2 @ true_W3 + true_b3).squeeze(-1)
    probs = sigmoid(logits)
    y = rng_local.binomial(1, probs).astype(np.float64)

    truth = {
        "W1": true_W1,
        "b1": true_b1,
        "W2": true_W2,
        "b2": true_b2,
        "W3": true_W3,
        "b3": true_b3,
    }
    return X.astype(np.float64), y, truth


def pack_params(weights: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            weights["W1"].ravel(),
            weights["b1"].ravel(),
            weights["W2"].ravel(),
            weights["b2"].ravel(),
            weights["W3"].ravel(),
            weights["b3"].ravel(),
        ]
    )


def unpack_params(theta: np.ndarray, dims: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    params = {}
    offset = 0
    for name, shape in dims.items():
        size = np.prod(shape, dtype=int)
        params[name] = theta[offset : offset + size].reshape(shape)
        offset += size
    return params


def forward_backward(
    params: dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float, dict[str, np.ndarray]]:
    batch = X.shape[0]
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    W3, b3 = params["W3"], params["b3"]

    z1 = X @ W1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2
    a2 = np.tanh(z2)
    logits = (a2 @ W3 + b3).squeeze(-1)
    pred = sigmoid(logits)

    loss = -np.mean(y * np.log(pred + 1e-12) + (1.0 - y) * np.log(1.0 - pred + 1e-12))

    delta3 = (pred - y).reshape(batch, 1)
    grad_W3 = (a2.T @ delta3) / batch
    grad_b3 = delta3.mean(axis=0)
    delta2 = (delta3 @ W3.T) * (1.0 - a2**2)
    grad_W2 = (a1.T @ delta2) / batch
    grad_b2 = delta2.mean(axis=0)
    delta1 = (delta2 @ W2.T) * (1.0 - a1**2)
    grad_W1 = (X.T @ delta1) / batch
    grad_b1 = delta1.mean(axis=0)

    grads = {
        "W1": grad_W1,
        "b1": grad_b1,
        "W2": grad_W2,
        "b2": grad_b2,
        "W3": grad_W3,
        "b3": grad_b3,
    }
    return loss, grads


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    X, y, truth = generate_data(rank, size, args.samples, args.features, args.hidden1, args.hidden2, args.seed)
    samples_local = X.shape[0]
    samples_total = args.samples

    if rank == 0:
        PETSc.Sys.Print(f"Neural Network Configuration:")
        PETSc.Sys.Print(f"  Architecture: {args.features} -> {args.hidden1} -> {args.hidden2} -> 1")
        PETSc.Sys.Print(f"  Total samples: {samples_total} ({samples_local} per rank)")
        PETSc.Sys.Print(f"  Batch size: {args.batch}")
        PETSc.Sys.Print(f"  MPI ranks: {size}")
        PETSc.Sys.Print(f"  TAO solver: {args.tao_type}")
        PETSc.Sys.Print(f"  Max iterations: {args.epochs}")
        PETSc.Sys.Print("")

    dims = {
        "W1": (args.features, args.hidden1),
        "b1": (args.hidden1,),
        "W2": (args.hidden1, args.hidden2),
        "b2": (args.hidden2,),
        "W3": (args.hidden2, 1),
        "b3": (1,),
    }
    total_params = sum(np.prod(shape, dtype=int) for shape in dims.values())

    # Create vectors - using numpy arrays wrapped in PETSc vectors
    init_params = np.random.default_rng(args.seed + 123).normal(scale=0.05, size=total_params)
    w_np = init_params.copy()
    grad_np = np.zeros(total_params, dtype=np.float64)

    w = PETSc.Vec().createWithArray(w_np, comm=PETSc.COMM_SELF)
    grad = PETSc.Vec().createWithArray(grad_np, comm=PETSc.COMM_SELF)

    batch = max(1, min(args.batch, samples_local))

    def objective_gradient(tao, x, G):
        theta = x.getArray(readonly=True)
        params = unpack_params(theta, dims)

        grad_local = np.zeros_like(theta)
        loss_accum = 0.0
        count = 0

        for start in range(0, samples_local, batch):
            end = min(start + batch, samples_local)
            xb = X[start:end]
            yb = y[start:end]
            loss, grads = forward_backward(params, xb, yb)
            weight = end - start
            loss_accum += loss * weight
            count += weight
            grad_batch = pack_params(grads)
            grad_local += grad_batch * weight

        total_loss = comm.allreduce(loss_accum, op=MPI.SUM) / samples_total
        grad_sum = np.empty_like(grad_local)
        comm.Allreduce(grad_local, grad_sum, op=MPI.SUM)
        grad_sum /= samples_total

        if G is not None:
            g_arr = G.getArray()
            g_arr[:] = grad_sum

        return total_loss

    tao = PETSc.TAO().create(comm=PETSc.COMM_SELF)
    tao.setType(args.tao_type)
    tao.setObjectiveGradient(objective_gradient, grad)
    
    # Set up monitoring to display convergence history
    def monitor_progress(tao):
        its, f, res, cnorm, step, reason = tao.getSolutionStatus()
        if rank == 0:
            PETSc.Sys.Print(f"  Iteration {its:3d}: Loss = {f:.6e}, |grad| = {res:.6e}")
    
    tao.setMonitor(monitor_progress)
    
    # Set maximum iterations from command-line argument
    tao.setMaximumIterations(args.epochs)
    
    tao.setFromOptions()

    if rank == 0:
        PETSc.Sys.Print("Starting TAO optimization...")
    
    tao.solve(w)

    if rank == 0:
        PETSc.Sys.Print(f"TAO converged in {tao.getIterationNumber()} iterations with reason {tao.getConvergedReason()}")

    theta_final = w.getArray(readonly=True)
    final_params = unpack_params(theta_final, dims)

    a1_local = np.tanh(X @ final_params["W1"] + final_params["b1"])
    a2_local = np.tanh(a1_local @ final_params["W2"] + final_params["b2"])
    logits_local = (a2_local @ final_params["W3"] + final_params["b3"]).squeeze(-1)
    probs_local = sigmoid(logits_local)
    preds_local = (probs_local >= 0.5).astype(np.float64)
    correct_local = np.sum(preds_local == y)

    total_correct = comm.allreduce(correct_local, op=MPI.SUM)
    accuracy = total_correct / samples_total

    if rank == 0:
        PETSc.Sys.Print(f"Training accuracy: {accuracy:.3f}")
        truth_flat = pack_params(truth)
        cos_sim = np.dot(theta_final, truth_flat) / (
            np.linalg.norm(theta_final) * np.linalg.norm(truth_flat) + 1e-12
        )
        PETSc.Sys.Print(f"Cosine similarity to synthetic ground-truth parameters: {cos_sim:.4f}")


if __name__ == "__main__":
    main()
