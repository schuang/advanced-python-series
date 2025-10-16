"""Data-parallel logistic regression with PETSc and MPI."""

import argparse

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPI data-parallel logistic regression with PETSc.")
    parser.add_argument("--samples", type=int, default=200_000, help="Total number of samples across all ranks.")
    parser.add_argument("--features", type=int, default=64, help="Number of features per sample.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of passes over the data.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--batch", type=int, default=2048, help="Mini-batch size per rank.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    args = parser.parse_args()
    return args


def generate_data(
    rank: int,
    size: int,
    samples_total: int,
    features: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples_local = samples_total // size
    rng = np.random.default_rng(seed + rank * 97)
    # Use same ground-truth weights for all ranks (broadcast from rank 0).
    if rank == 0:
        true_w = np.random.default_rng(seed).normal(scale=1.0, size=features)
    else:
        true_w = np.empty(features, dtype=np.float64)
    MPI.COMM_WORLD.Bcast(true_w, root=0)

    X = rng.normal(size=(samples_local, features)).astype(np.float64)
    logits = X @ true_w
    probs = sigmoid(logits)
    y = rng.binomial(1, probs).astype(np.float64)
    return X, y, true_w


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if args.samples % size != 0:
        if rank == 0:
            raise ValueError("--samples must be divisible by number of ranks.")
        return

    X, y, true_w = generate_data(rank, size, args.samples, args.features, args.seed)
    samples_local = X.shape[0]

    # PETSc vector for the model parameters (local copy per rank wrapped for BLAS ops).
    w_np = np.zeros(args.features, dtype=np.float64)
    w = PETSc.Vec().createWithArray(w_np, comm=PETSc.COMM_SELF)

    batch = max(1, min(args.batch, samples_local))
    lr = args.lr

    for epoch in range(1, args.epochs + 1):
        perm = np.random.permutation(samples_local)
        X_shuff = X[perm]
        y_shuff = y[perm]

        total_loss_local = 0.0
        total_correct_local = 0
        total_seen_local = 0

        for start in range(0, samples_local, batch):
            end = min(start + batch, samples_local)
            xb = X_shuff[start:end]
            yb = y_shuff[start:end]

            preds = sigmoid(xb @ w_np)
            errors = preds - yb
            grad_local = xb.T @ errors  # shape (features,)

            loss_local = np.mean(
                -(yb * np.log(preds + 1e-12) + (1.0 - yb) * np.log(1.0 - preds + 1e-12))
            )
            correct_local = np.sum((preds >= 0.5) == yb)

            total_loss_local += loss_local * (end - start)
            total_correct_local += correct_local
            total_seen_local += (end - start)

            grad_global = np.empty_like(grad_local)
            comm.Allreduce(grad_local, grad_global, op=MPI.SUM)
            grad_global /= args.samples  # average over all samples

            grad_vec = PETSc.Vec().createWithArray(grad_global, comm=PETSc.COMM_SELF)
            w.axpy(-lr, grad_vec)  # w = w + (-lr)*grad

        total_loss = comm.allreduce(total_loss_local, op=MPI.SUM) / args.samples
        total_correct = comm.allreduce(total_correct_local, op=MPI.SUM)
        accuracy = total_correct / args.samples

        if rank == 0:
            PETSc.Sys.Print(
                f"Epoch {epoch:02d}/{args.epochs}: loss={total_loss:.4f}, accuracy={accuracy:.3f}"
            )

    if rank == 0:
        cosine_sim = np.dot(w_np, true_w) / (
            np.linalg.norm(w_np) * np.linalg.norm(true_w) + 1e-12
        )
        PETSc.Sys.Print(f"Cosine similarity to true weights: {cosine_sim:.4f}")


if __name__ == "__main__":
    main()
