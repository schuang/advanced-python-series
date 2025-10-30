#!/usr/bin/env python3
"""
Vector Addition with Numba CUDA

This example demonstrates GPU kernel programming with Numba:
- Defining CUDA kernels with @cuda.jit
- GPU memory management
- Thread indexing with cuda.grid()
- Kernel launch configuration (blocks and threads)

Requirements:
- numba
- numpy
- NVIDIA GPU with CUDA support

Usage:
    python vecadd-numba.py
"""

from numba import cuda
import numpy as np
import math
import time


@cuda.jit
def vector_add_kernel(a, b, c):
    """
    GPU kernel for element-wise vector addition.
    Each thread computes one element of the output.
    """
    # Calculate global thread index
    idx = cuda.grid(1)

    # Boundary check
    if idx < c.size:
        c[idx] = a[idx] + b[idx]


def vector_add_gpu(a, b):
    """
    Wrapper function to launch GPU kernel.

    Args:
        a: First input array (numpy array)
        b: Second input array (numpy array)

    Returns:
        c: Result array (numpy array)
    """
    # Allocate device memory
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(a)

    # Configure kernel launch
    threads_per_block = 256
    blocks_per_grid = math.ceil(a.size / threads_per_block)

    print(f"Launching kernel with {blocks_per_grid} blocks and {threads_per_block} threads per block")

    # Launch kernel
    vector_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

    # Copy result back to host
    c = d_c.copy_to_host()
    return c


def main():
    """Main function to run vector addition example."""
    # Check CUDA availability
    if not cuda.is_available():
        print("CUDA is not available. This example requires an NVIDIA GPU.")
        return

    print("CUDA devices detected:")
    for i, device in enumerate(cuda.gpus):
        with device:
            print(f"  Device {i}: {device.name.decode()}")

    print()

    # Test with different sizes
    sizes = [1_000, 100_000, 1_000_000, 10_000_000]

    for n in sizes:
        print(f"\nTesting with array size: {n:,}")

        # Create input arrays
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)

        # Perform vector addition on GPU
        start = time.time()
        c_gpu = vector_add_gpu(a, b)
        gpu_time = time.time() - start

        # Verify correctness with CPU computation
        c_cpu = a + b
        max_error = np.max(np.abs(c_gpu - c_cpu))

        print(f"  GPU time: {gpu_time*1000:.3f} ms")
        print(f"  Max error: {max_error:.2e}")
        print(f"  Correctness: {'✓ PASS' if max_error < 1e-5 else '✗ FAIL'}")


if __name__ == "__main__":
    main()
