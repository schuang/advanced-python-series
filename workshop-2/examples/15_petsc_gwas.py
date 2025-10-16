"""Distributed GWAS chi-square scan using PETSc and MPI."""

import argparse
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPI/PETSc genome-wide association scan.")
    parser.add_argument("--samples", type=int, default=5_000, help="Number of individuals.")
    parser.add_argument("--variants", type=int, default=20_000, help="Number of variants (SNPs).")
    parser.add_argument("--maf-min", type=float, default=0.05, help="Minimum minor-allele frequency.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--report", type=int, default=5, help="Top-N variants to print.")
    args = parser.parse_args()
    return args


def compute_local_range(global_n: int, size: int, rank: int) -> tuple[int, int]:
    base = global_n // size
    rem = global_n % size
    start = rank * base + min(rank, rem)
    local = base + (1 if rank < rem else 0)
    end = start + local
    return start, end


def simulate_genotypes(
    num_samples: int,
    num_variants: int,
    start_idx: int,
    seed: int,
    maf_min: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed + start_idx * 97)
    mafs = rng.uniform(maf_min, 0.5, size=num_variants)
    genotypes = np.empty((num_samples, num_variants), dtype=np.int8)
    for j, maf in enumerate(mafs):
        genotypes[:, j] = rng.binomial(2, maf, size=num_samples)
    return genotypes


def chi_square_stat(
    genotypes: np.ndarray,
    phenotype: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cases_mask = phenotype == 1
    controls_mask = ~cases_mask
    cases = cases_mask.sum()
    controls = controls_mask.sum()
    two_cases = 2.0 * cases
    two_controls = 2.0 * controls

    minor_cases = genotypes[cases_mask].sum(axis=0, dtype=np.int64)
    minor_controls = genotypes[controls_mask].sum(axis=0, dtype=np.int64)

    major_cases = two_cases - minor_cases
    major_controls = two_controls - minor_controls

    observed = np.stack(
        [minor_cases, major_cases, minor_controls, major_controls], axis=0
    ).astype(np.float64)

    totals = np.array([
        two_cases,
        two_controls,
    ])
    allele_totals = np.stack(
        [minor_cases + minor_controls, major_cases + major_controls], axis=0
    ).astype(np.float64)

    expected_cases_minor = totals[0] * allele_totals[0] / (totals.sum())
    expected_cases_major = totals[0] * allele_totals[1] / (totals.sum())
    expected_controls_minor = totals[1] * allele_totals[0] / (totals.sum())
    expected_controls_major = totals[1] * allele_totals[1] / (totals.sum())

    expected = np.stack(
        [
            expected_cases_minor,
            expected_cases_major,
            expected_controls_minor,
            expected_controls_major,
        ],
        axis=0,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nan_to_num((observed - expected) ** 2 / expected).sum(axis=0)

    maf_local = allele_totals[0] / (2.0 * (cases + controls))
    return chi2, maf_local


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.variants < size:
        if rank == 0:
            PETSc.Sys.Print("Number of variants must be >= number of ranks.")
        return

    phen_rng = np.random.default_rng(args.seed)
    phenotype = phen_rng.binomial(1, 0.5, size=args.samples).astype(np.int8)
    comm.Bcast(phenotype, root=0)

    start_idx, end_idx = compute_local_range(args.variants, size, rank)
    local_variants = end_idx - start_idx

    geno = simulate_genotypes(
        args.samples,
        local_variants,
        start_idx,
        args.seed,
        args.maf_min,
    )

    chi2_local, maf_local = chi_square_stat(geno, phenotype)

    chi2_vec = PETSc.Vec().createMPI(args.variants, bsize=local_variants, comm=PETSc.COMM_WORLD)
    maf_vec = PETSc.Vec().createMPI(args.variants, bsize=local_variants, comm=PETSc.COMM_WORLD)

    chi2_arr = chi2_vec.getArray()
    chi2_arr[:] = chi2_local
    maf_arr = maf_vec.getArray()
    maf_arr[:] = maf_local
    chi2_vec.assemblyBegin()
    chi2_vec.assemblyEnd()
    maf_vec.assemblyBegin()
    maf_vec.assemblyEnd()

    max_val, max_idx = chi2_vec.max()
    max_idx = int(max_idx)
    owner_rank = -1
    top_maf = 0.0
    start_range, end_range = chi2_vec.getOwnershipRange()
    if start_range <= max_idx < end_range:
        owner_rank = rank
        top_maf = maf_local[max_idx - start_range]
    owner_rank = comm.allreduce(owner_rank, op=MPI.MAX)
    top_maf = comm.bcast(top_maf, root=owner_rank)

    if rank == 0:
        PETSc.Sys.Print(
            f"Top chi-square variant: index={max_idx}, chi2={max_val:.3f}, maf={top_maf:.3f}"
        )

    if args.report > 0:
        local_pairs = list(zip(range(start_idx, end_idx), chi2_local, maf_local))
        local_pairs.sort(key=lambda x: x[1], reverse=True)
        top_local = local_pairs[: args.report]
        gathered = comm.gather(top_local, root=0)
        if rank == 0:
            flattened = [item for sublist in gathered for item in sublist]
            flattened.sort(key=lambda x: x[1], reverse=True)
            PETSc.Sys.Print(f"Top {args.report} variants by chi-square:")
            for idx, chi2_val, maf_val in flattened[: args.report]:
                PETSc.Sys.Print(f"  variant={idx:7d} chi2={chi2_val:8.3f} maf={maf_val:.3f}")


if __name__ == "__main__":
    main()
