#!/usr/bin/env python3
"""
Monte Carlo Pi Estimation - GPU Version with Numba CUDA

This example demonstrates:
- GPU kernel programming with @cuda.jit
- Parallel random number generation on GPU
- Reduction operations across GPU threads
- Optimal kernel launch configuration

Algorithm:
1. Each GPU thread generates multiple random samples
2. Count points inside quarter circle
3. Reduce counts across all threads
4. π ≈ 4 × (total points inside) / (total points)

Requirements:
- numba
- numpy
- NVIDIA GPU with CUDA support

Usage:
    python monte-carlo-pi-gpu.py
"""

from numba import cuda
import numpy as np
import math
import time


@cuda.jit
def monte_carlo_pi_kernel(n_samples_per_thread, counts, seed_offset):
    """
    GPU kernel for Monte Carlo pi estimation.
    Each thread generates samples and counts hits.

    Args:
        n_samples_per_thread: Number of samples each thread processes
        counts: Output array to store per-thread hit counts
        seed_offset: Offset for random seed to ensure different sequences
    """
    idx = cuda.grid(1)

    if idx >= counts.size:
        return

    # Simple LCG (Linear Congruential Generator) for random numbers
    # Note: This is a simple RNG, not cryptographically secure
    # Each thread gets unique seed based on its index
    seed = idx + seed_offset * 1000000

    thread_count = 0

    for i in range(n_samples_per_thread):
        # Generate random x
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        x = seed / 2147483648.0

        # Generate random y
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        y = seed / 2147483648.0

        # Check if inside circle
        if x*x + y*y <= 1.0:
            thread_count += 1

    # Store thread result
    counts[idx] = thread_count


def monte_carlo_pi_gpu(n_samples, threads_per_block=256, blocks_per_grid=512):
    """
    Wrapper for GPU Monte Carlo pi estimation.

    Args:
        n_samples: Total number of samples to generate
        threads_per_block: Threads per block (default 256)
        blocks_per_grid: Number of blocks (default 512)

    Returns:
        float: Estimate of π
    """
    total_threads = threads_per_block * blocks_per_grid
    samples_per_thread = math.ceil(n_samples / total_threads)
    actual_samples = samples_per_thread * total_threads

    # Allocate device memory for counts
    d_counts = cuda.device_array(total_threads, dtype=np.int32)

    # Use time-based seed offset for different runs
    seed_offset = int(time.time() * 1000) % 1000000

    # Launch kernel
    monte_carlo_pi_kernel[blocks_per_grid, threads_per_block](
        samples_per_thread, d_counts, seed_offset
    )

    # Copy results and sum
    counts = d_counts.copy_to_host()
    total_inside = counts.sum()

    # Calculate pi estimate
    pi_estimate = 4.0 * total_inside / actual_samples

    return pi_estimate


def main():
    """Main function to run GPU Monte Carlo Pi estimation."""
    # Check CUDA availability
    if not cuda.is_available():
        print("CUDA is not available. This example requires an NVIDIA GPU.")
        return

    print("Monte Carlo Pi Estimation - GPU Version\n")
    print("=" * 60)

    print("\nCUDA devices detected:")
    for i, device in enumerate(cuda.gpus):
        with device:
            print(f"  Device {i}: {device.name.decode()}")

    print("\n" + "-" * 60)

    # Different sample sizes to test
    sample_sizes = [1_000_000, 10_000_000, 100_000_000, 500_000_000]

    # Kernel configuration
    threads_per_block = 256
    blocks_per_grid = 512
    total_threads = threads_per_block * blocks_per_grid

    print(f"\nKernel Configuration:")
    print(f"  Threads per block: {threads_per_block}")
    print(f"  Blocks per grid:   {blocks_per_grid}")
    print(f"  Total GPU threads: {total_threads:,}")

    print("\n" + "-" * 60)

    for n_samples in sample_sizes:
        print(f"\nSample size: {n_samples:,}")

        # Run multiple times for stable timing
        n_runs = 3
        times = []

        for run in range(n_runs):
            start = time.time()
            pi_estimate = monte_carlo_pi_gpu(
                n_samples,
                threads_per_block,
                blocks_per_grid
            )
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        error = abs(pi_estimate - np.pi)
        error_percent = (error / np.pi) * 100

        print(f"  π estimate: {pi_estimate:.6f}")
        print(f"  Actual π:   {np.pi:.6f}")
        print(f"  Error:      {error:.6f} ({error_percent:.3f}%)")
        print(f"  Time:       {avg_time:.3f} seconds (avg of {n_runs} runs)")
        print(f"  Throughput: {n_samples/avg_time/1e6:.2f} M samples/sec")

    print("\n" + "=" * 60)
    print("Note: GPU excels at large sample sizes (100M+ samples)")
    print("      Typical speedup: 100-1000x vs pure Python")
    print("=" * 60)


if __name__ == "__main__":
    main()
