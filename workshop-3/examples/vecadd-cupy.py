#!/usr/bin/env python3
"""
CuPy Vector Addition Example
Demonstrates CuPy as a drop-in replacement for NumPy with GPU acceleration.

This example shows:
- Minimal code changes (numpy -> cupy)
- Automatic GPU execution
- Memory transfer between CPU and GPU
- Performance comparison with NumPy
"""

import numpy as np
import cupy as cp
import time


def main():
    # Vector size
    n = 50_000_000

    print("=" * 60)
    print("CuPy Vector Addition Example")
    print("=" * 60)

    # ========================================================================
    # NumPy (CPU) Version
    # ========================================================================
    print("\n" + "=" * 60)
    print("NumPy (CPU) Version")
    print("=" * 60)

    # Create arrays on CPU
    a_cpu = np.random.randn(n).astype(np.float32)
    b_cpu = np.random.randn(n).astype(np.float32)

    # Time CPU execution
    start = time.time()
    c_cpu = a_cpu + b_cpu
    cpu_time = time.time() - start

    print(f"Array size: {n:,} elements ({n*4/1e6:.1f} MB per array)")
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"First 5 elements: {c_cpu[:5]}")

    # ========================================================================
    # CuPy (GPU) Version
    # ========================================================================
    print("\n" + "=" * 60)
    print("CuPy (GPU) Version")
    print("=" * 60)

    # Create arrays on GPU (just change np to cp!)
    a_gpu = cp.random.randn(n).astype(cp.float32)
    b_gpu = cp.random.randn(n).astype(cp.float32)

    # Warm-up (first call includes compilation overhead)
    _ = a_gpu + b_gpu
    cp.cuda.Stream.null.synchronize()

    # Time GPU execution
    start = time.time()
    c_gpu = a_gpu + b_gpu
    cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
    gpu_time = time.time() - start

    print(f"Array size: {n:,} elements ({n*4/1e6:.1f} MB per array)")
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"First 5 elements: {cp.asnumpy(c_gpu[:5])}")

    # ========================================================================
    # Performance Comparison
    # ========================================================================
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    print(f"CPU time: {cpu_time:.4f}s")
    print(f"GPU time: {gpu_time:.4f}s")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x faster")

    # ========================================================================
    # Including Memory Transfer Overhead
    # ========================================================================
    print("\n" + "=" * 60)
    print("Including Memory Transfer Overhead")
    print("=" * 60)

    # Start from NumPy arrays, transfer to GPU, compute, transfer back
    start = time.time()
    a_transfer = cp.asarray(a_cpu)  # CPU -> GPU
    b_transfer = cp.asarray(b_cpu)  # CPU -> GPU
    c_transfer = a_transfer + b_transfer
    result = cp.asnumpy(c_transfer)  # GPU -> CPU
    cp.cuda.Stream.null.synchronize()
    total_time = time.time() - start

    print(f"Total time (with transfers): {total_time:.4f}s")
    print(f"Pure GPU compute: {gpu_time:.4f}s")
    print(f"Transfer overhead: {total_time - gpu_time:.4f}s")
    print(f"Speedup (with transfers): {cpu_time/total_time:.1f}x")

    # ========================================================================
    # Correctness Check
    # ========================================================================
    print("\n" + "=" * 60)
    print("Correctness Check")
    print("=" * 60)
    # Compare small portion (avoid memory issues)
    test_size = 1000
    a_test = np.random.randn(test_size).astype(np.float32)
    b_test = np.random.randn(test_size).astype(np.float32)

    c_numpy = a_test + b_test
    c_cupy = cp.asnumpy(cp.asarray(a_test) + cp.asarray(b_test))

    print(f"Results match: {np.allclose(c_numpy, c_cupy)}")
    print(f"Max difference: {np.max(np.abs(c_numpy - c_cupy))}")

    # ========================================================================
    # Best Practice Example
    # ========================================================================
    print("\n" + "=" * 60)
    print("Best Practice: Keep Data on GPU")
    print("=" * 60)

    # Good: Keep data on GPU for multiple operations
    a_gpu = cp.asarray(a_cpu)  # Transfer once
    b_gpu = cp.asarray(b_cpu)

    start = time.time()
    # Many operations without transfer
    c_gpu = a_gpu + b_gpu
    d_gpu = c_gpu * 2.0
    e_gpu = cp.sin(d_gpu)
    result = cp.asnumpy(e_gpu)  # Transfer once at end
    cp.cuda.Stream.null.synchronize()
    optimized_time = time.time() - start

    print(f"Multiple operations with minimal transfers: {optimized_time:.4f}s")
    print("This is the recommended pattern for best performance!")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This example requires:")
        print("  - NVIDIA GPU with CUDA support")
        print("  - CuPy installed: pip install cupy-cuda12x")
        raise
