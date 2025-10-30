#!/usr/bin/env python3
"""
Numba Multi-Core CPU Parallelization Examples

This example demonstrates:
- Automatic parallelization with parallel=True
- Explicit parallel loops with prange
- Parallel reductions
- Thread control
- Performance comparisons

Requirements:
- numba
- numpy

Usage:
    python numba-cpu-parallel.py
"""

from numba import jit, prange, set_num_threads, get_num_threads
import numpy as np
import time
import os


# ============================================================================
# Example 1: Automatic Loop Parallelization
# ============================================================================

@jit(nopython=True)
def sum_squares_sequential(arr):
    """Sequential computation"""
    n = arr.shape[0]
    result = np.zeros(n)
    for i in range(n):
        result[i] = arr[i] ** 2
    return result


@jit(nopython=True, parallel=True)
def sum_squares_parallel(arr):
    """Parallel computation across CPU cores"""
    n = arr.shape[0]
    result = np.zeros(n)
    for i in range(n):
        result[i] = arr[i] ** 2
    return result


def example_automatic_parallelization():
    """Demonstrate automatic parallelization with parallel=True"""
    print("\n" + "=" * 70)
    print("Example 1: Automatic Loop Parallelization (parallel=True)")
    print("=" * 70)

    n = 100_000_000
    data = np.random.randn(n)

    # Warmup
    _ = sum_squares_sequential(data[:1000])
    _ = sum_squares_parallel(data[:1000])

    # Sequential
    start = time.time()
    result_seq = sum_squares_sequential(data)
    seq_time = time.time() - start

    # Parallel
    start = time.time()
    result_par = sum_squares_parallel(data)
    par_time = time.time() - start

    print(f"\nArray size: {n:,}")
    print(f"Sequential time: {seq_time:.3f}s")
    print(f"Parallel time:   {par_time:.3f}s")
    print(f"Speedup:         {seq_time/par_time:.2f}x")
    print(f"Results match:   {np.allclose(result_seq, result_par)}")


# ============================================================================
# Example 2: Explicit Parallel Loops with prange
# ============================================================================

@jit(nopython=True, parallel=True)
def parallel_computation(a, b):
    """Explicitly parallel loop using prange"""
    n = a.shape[0]
    result = np.zeros(n)

    # prange explicitly parallelizes this loop
    for i in prange(n):
        # Each iteration runs on different CPU core
        result[i] = np.sqrt(a[i]**2 + b[i]**2)

    return result


def example_prange():
    """Demonstrate explicit parallel loops with prange"""
    print("\n" + "=" * 70)
    print("Example 2: Explicit Parallel Loops (prange)")
    print("=" * 70)

    n = 50_000_000
    a = np.random.randn(n)
    b = np.random.randn(n)

    # Warmup
    _ = parallel_computation(a[:1000], b[:1000])

    # Time the execution
    start = time.time()
    result = parallel_computation(a, b)
    elapsed = time.time() - start

    print(f"\nArray size: {n:,}")
    print(f"Time:       {elapsed:.3f}s")
    print(f"Throughput: {n/elapsed/1e6:.1f} M operations/second")


# ============================================================================
# Example 3: Parallel Reductions
# ============================================================================

@jit(nopython=True, parallel=True)
def parallel_sum(arr):
    """Parallel sum reduction"""
    total = 0.0
    # Numba parallelizes this reduction automatically
    for i in prange(arr.shape[0]):
        total += arr[i]
    return total


@jit(nopython=True, parallel=True)
def parallel_dot_product(a, b):
    """Parallel dot product"""
    result = 0.0
    for i in prange(a.shape[0]):
        result += a[i] * b[i]
    return result


def example_parallel_reduction():
    """Demonstrate parallel reduction operations"""
    print("\n" + "=" * 70)
    print("Example 3: Parallel Reductions")
    print("=" * 70)

    n = 100_000_000
    arr = np.random.randn(n)

    # Warmup
    _ = parallel_sum(arr[:1000])

    # Parallel sum
    start = time.time()
    par_sum = parallel_sum(arr)
    par_time = time.time() - start

    # NumPy sum (single-threaded for large arrays)
    start = time.time()
    np_sum = np.sum(arr)
    np_time = time.time() - start

    print(f"\nArray size: {n:,}")
    print(f"Parallel sum: {par_sum:.6f} ({par_time:.3f}s)")
    print(f"NumPy sum:    {np_sum:.6f} ({np_time:.3f}s)")
    print(f"Speedup:      {np_time/par_time:.2f}x")
    print(f"Match:        {np.isclose(par_sum, np_sum)}")


# ============================================================================
# Example 4: Nested Parallel Loops
# ============================================================================

@jit(nopython=True, parallel=True)
def parallel_matrix_op(A, B):
    """Element-wise operation on matrices with nested parallel loops"""
    m, n = A.shape
    C = np.zeros((m, n))

    # Outer loop parallelized
    for i in prange(m):
        for j in range(n):
            C[i, j] = np.sqrt(A[i, j]**2 + B[i, j]**2)

    return C


def example_nested_parallel():
    """Demonstrate nested parallel loops for 2D operations"""
    print("\n" + "=" * 70)
    print("Example 4: Nested Parallel Loops (2D Operations)")
    print("=" * 70)

    m, n = 10000, 10000
    A = np.random.randn(m, n).astype(np.float32)
    B = np.random.randn(m, n).astype(np.float32)

    # Warmup
    _ = parallel_matrix_op(A[:100, :100], B[:100, :100])

    start = time.time()
    C = parallel_matrix_op(A, B)
    elapsed = time.time() - start

    print(f"\nMatrix size: {m}×{n} = {m*n:,} elements")
    print(f"Time:        {elapsed:.3f}s")
    print(f"Throughput:  {m*n/elapsed/1e6:.1f} M ops/sec")


# ============================================================================
# Example 5: Comparing Sequential vs Parallel Performance
# ============================================================================

def benchmark_sizes():
    """Show when parallel execution helps"""
    print("\n" + "=" * 70)
    print("Example 5: Sequential vs Parallel Performance")
    print("=" * 70)

    sizes = [1000, 10000, 100000, 1000000, 10000000]

    @jit(nopython=True)
    def compute_seq(arr):
        result = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            result[i] = np.sqrt(arr[i]**2 + arr[i]**3)
        return result

    @jit(nopython=True, parallel=True)
    def compute_par(arr):
        result = np.zeros_like(arr)
        for i in prange(arr.shape[0]):
            result[i] = np.sqrt(arr[i]**2 + arr[i]**3)
        return result

    print(f"\n{'Size':<12} {'Sequential':<12} {'Parallel':<12} {'Speedup':<12}")
    print("-" * 50)

    for size in sizes:
        arr = np.random.randn(size)

        # Warmup
        _ = compute_seq(arr)
        _ = compute_par(arr)

        # Sequential
        start = time.time()
        _ = compute_seq(arr)
        seq_time = time.time() - start

        # Parallel
        start = time.time()
        _ = compute_par(arr)
        par_time = time.time() - start

        speedup = seq_time / par_time if par_time > 0 else 0

        print(f"{size:<12} {seq_time:<12.6f} {par_time:<12.6f} {speedup:<12.2f}")


# ============================================================================
# Example 6: Thread Control
# ============================================================================

def example_thread_control():
    """Demonstrate controlling number of threads"""
    print("\n" + "=" * 70)
    print("Example 6: Thread Control")
    print("=" * 70)

    # Check default thread count
    print(f"\nDefault threads: {get_num_threads()}")
    print(f"CPU cores:       {os.cpu_count()}")

    @jit(nopython=True, parallel=True)
    def parallel_work(arr):
        result = np.zeros_like(arr)
        for i in prange(arr.shape[0]):
            result[i] = np.sqrt(arr[i]**2)
        return result

    arr = np.random.randn(50_000_000)

    # Warmup
    _ = parallel_work(arr[:1000])

    print(f"\n{'Threads':<10} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 35)

    base_time = None

    # Test with different thread counts
    for nthreads in [1, 2, 4, 8]:
        if nthreads > os.cpu_count():
            break

        set_num_threads(nthreads)

        start = time.time()
        result = parallel_work(arr)
        elapsed = time.time() - start

        if base_time is None:
            base_time = elapsed
            speedup_str = "1.0x"
        else:
            speedup = base_time / elapsed
            speedup_str = f"{speedup:.2f}x"

        print(f"{nthreads:<10} {elapsed:<12.3f} {speedup_str:<10}")

    # Reset to default
    set_num_threads(os.cpu_count())


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all CPU parallelization examples"""
    print("\n" + "=" * 70)
    print("Numba Multi-Core CPU Parallelization Examples")
    print("=" * 70)

    print(f"\nSystem Information:")
    print(f"  CPU cores: {os.cpu_count()}")
    print(f"  Default Numba threads: {get_num_threads()}")

    # Run all examples
    example_automatic_parallelization()
    example_prange()
    example_parallel_reduction()
    example_nested_parallel()
    benchmark_sizes()
    example_thread_control()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("  ✓ parallel=True enables automatic parallelization")
    print("  ✓ prange for explicit parallel loops")
    print("  ✓ Works best for large arrays (>1M elements)")
    print("  ✓ Speedup typically 2-8x on modern CPUs")
    print("  ✓ No GIL limitations, true parallel execution")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
