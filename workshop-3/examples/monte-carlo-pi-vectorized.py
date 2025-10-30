#!/usr/bin/env python3
"""
Monte Carlo Pi Estimation - Vectorized GPU Version

This example demonstrates:
- @vectorize decorator for automatic GPU parallelization
- Simpler code than explicit CUDA kernels
- Automatic handling of thread management

Algorithm:
1. Generate random points (x, y) arrays
2. Use @vectorize to check each point on GPU
3. Sum results to get count inside circle
4. π ≈ 4 × (points inside) / (total points)

Requirements:
- numba
- numpy
- NVIDIA GPU with CUDA support

Usage:
    python monte-carlo-pi-vectorized.py
"""

from numba import vectorize, cuda
import numpy as np
import time


@vectorize(['int32(float32, float32)'], target='cuda')
def check_inside_circle(x, y):
    """
    Check if point is inside unit circle.
    Automatically parallelized across GPU threads.

    Args:
        x: x-coordinate (scalar, but applies to arrays)
        y: y-coordinate (scalar, but applies to arrays)

    Returns:
        int: 1 if inside circle, 0 otherwise
    """
    return 1 if (x*x + y*y <= 1.0) else 0


def monte_carlo_pi_vectorized(n_samples):
    """
    Vectorized GPU implementation of Monte Carlo Pi estimation.

    Args:
        n_samples: Number of random samples

    Returns:
        float: Estimate of π
    """
    # Generate random points on CPU
    x = np.random.random(n_samples).astype(np.float32)
    y = np.random.random(n_samples).astype(np.float32)

    # GPU automatically processes all points in parallel
    inside = check_inside_circle(x, y)

    return 4.0 * inside.sum() / n_samples


def main():
    """Main function to run vectorized Monte Carlo Pi estimation."""
    # Check CUDA availability
    if not cuda.is_available():
        print("CUDA is not available. This example requires an NVIDIA GPU.")
        return

    print("Monte Carlo Pi Estimation - Vectorized GPU Version\n")
    print("=" * 60)

    print("\nCUDA devices detected:")
    for i, device in enumerate(cuda.gpus):
        with device:
            print(f"  Device {i}: {device.name.decode()}")

    print("\n" + "-" * 60)

    # Different sample sizes to test
    sample_sizes = [1_000_000, 10_000_000, 50_000_000, 100_000_000]

    print("\nVectorized @vectorize with target='cuda':")
    print("-" * 60)

    for n_samples in sample_sizes:
        print(f"\nSample size: {n_samples:,}")

        # Run multiple times for stable timing
        n_runs = 3
        times = []
        estimates = []

        for run in range(n_runs):
            start = time.time()
            pi_estimate = monte_carlo_pi_vectorized(n_samples)
            elapsed = time.time() - start
            times.append(elapsed)
            estimates.append(pi_estimate)

        avg_time = np.mean(times)
        avg_estimate = np.mean(estimates)
        error = abs(avg_estimate - np.pi)
        error_percent = (error / np.pi) * 100

        print(f"  π estimate: {avg_estimate:.6f}")
        print(f"  Actual π:   {np.pi:.6f}")
        print(f"  Error:      {error:.6f} ({error_percent:.3f}%)")
        print(f"  Time:       {avg_time:.3f} seconds (avg of {n_runs} runs)")
        print(f"  Throughput: {n_samples/avg_time/1e6:.2f} M samples/sec")

    print("\n" + "=" * 60)
    print("Advantages of @vectorize:")
    print("  ✓ Simpler code than explicit CUDA kernels")
    print("  ✓ Automatic thread management")
    print("  ✓ Easy to switch between CPU and GPU (change target)")
    print("  ✓ Good performance for element-wise operations")
    print("\nNote: Random generation on CPU may be bottleneck")
    print("      For best performance, use explicit CUDA kernels")
    print("=" * 60)


if __name__ == "__main__":
    main()
