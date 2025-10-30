#!/usr/bin/env python3
"""
Monte Carlo Pi Estimation - CPU Version with Numba JIT

This example demonstrates:
- CPU acceleration with @jit decorator
- Monte Carlo method for estimating π
- Numba's JIT compilation for loop-heavy code

Algorithm:
1. Generate random points (x, y) in [0, 1] × [0, 1]
2. Check if x² + y² ≤ 1 (inside quarter circle)
3. π ≈ 4 × (points inside circle) / (total points)

Requirements:
- numba
- numpy

Usage:
    python monte-carlo-pi-cpu.py
"""

from numba import jit
import numpy as np
import time


@jit(nopython=True)
def monte_carlo_pi_cpu(n_samples):
    """
    Estimate pi using Monte Carlo method on CPU.

    Args:
        n_samples: Number of random samples to generate

    Returns:
        float: Estimate of π
    """
    count_inside = 0

    for i in range(n_samples):
        x = np.random.random()
        y = np.random.random()

        # Check if point is inside quarter circle
        if x*x + y*y <= 1.0:
            count_inside += 1

    return 4.0 * count_inside / n_samples


def pure_python_monte_carlo_pi(n_samples):
    """
    Pure Python version for comparison (no Numba).

    Args:
        n_samples: Number of random samples to generate

    Returns:
        float: Estimate of π
    """
    count_inside = 0

    for i in range(n_samples):
        x = np.random.random()
        y = np.random.random()

        # Check if point is inside quarter circle
        if x*x + y*y <= 1.0:
            count_inside += 1

    return 4.0 * count_inside / n_samples


def main():
    """Main function to run Monte Carlo Pi estimation."""
    print("Monte Carlo Pi Estimation - CPU Version\n")
    print("=" * 60)

    # Different sample sizes to test
    sample_sizes = [100_000, 1_000_000, 10_000_000]

    print("\nNumba JIT Compilation:")
    print("-" * 60)

    for n_samples in sample_sizes:
        print(f"\nSample size: {n_samples:,}")

        # Warm up JIT compilation
        if n_samples == sample_sizes[0]:
            print("  Warming up JIT compiler...")
            _ = monte_carlo_pi_cpu(10_000)

        # Run Numba version
        start = time.time()
        pi_estimate = monte_carlo_pi_cpu(n_samples)
        numba_time = time.time() - start

        error = abs(pi_estimate - np.pi)
        error_percent = (error / np.pi) * 100

        print(f"  Numba JIT version:")
        print(f"    π estimate: {pi_estimate:.6f}")
        print(f"    Actual π:   {np.pi:.6f}")
        print(f"    Error:      {error:.6f} ({error_percent:.3f}%)")
        print(f"    Time:       {numba_time:.3f} seconds")
        print(f"    Throughput: {n_samples/numba_time/1e6:.2f} M samples/sec")

    # Compare with pure Python for smaller size
    print("\n\nComparison with Pure Python:")
    print("-" * 60)
    n_compare = 100_000
    print(f"Sample size: {n_compare:,}\n")

    # Pure Python
    start = time.time()
    pi_python = pure_python_monte_carlo_pi(n_compare)
    python_time = time.time() - start

    # Numba (already warmed up)
    start = time.time()
    pi_numba = monte_carlo_pi_cpu(n_compare)
    numba_time = time.time() - start

    print(f"  Pure Python time:  {python_time:.3f} seconds")
    print(f"  Numba JIT time:    {numba_time:.3f} seconds")
    print(f"  Speedup:           {python_time/numba_time:.1f}x")

    print("\n" + "=" * 60)
    print("Note: Speedup is typically 50-100x for Numba JIT on CPU")
    print("=" * 60)


if __name__ == "__main__":
    main()
