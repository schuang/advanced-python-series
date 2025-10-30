#!/usr/bin/env python3
"""
CUDA Python Vector Addition Example
Demonstrates low-level CUDA programming from Python using cuda.core.

This example shows:
- Loading pre-compiled CUDA kernels
- Explicit memory management
- Manual memory transfers
- Direct kernel launch configuration
- Low-level GPU control

Note: This requires a compiled CUDA kernel (PTX or cubin).
For simplicity, this example uses CuPy's RawKernel feature to demonstrate
the low-level approach, which is similar to CUDA Python's workflow.
"""

import numpy as np
import cupy as cp
import time


# CUDA C kernel code for vector addition
CUDA_KERNEL_CODE = '''
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
'''


def main():
    print("=" * 60)
    print("CUDA Python Vector Addition Example")
    print("=" * 60)
    print("\nThis demonstrates low-level CUDA programming from Python")
    print("using explicit memory management and kernel launches.\n")

    # Vector size
    n = 10_000_000
    print(f"Vector size: {n:,} elements ({n*4/1e6:.1f} MB per array)")

    # ========================================================================
    # Step 1: Compile CUDA Kernel
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Compile CUDA Kernel")
    print("=" * 60)

    # Compile the CUDA C code to a kernel
    # This is similar to cuModuleLoadData in CUDA Python
    vector_add_kernel = cp.RawKernel(CUDA_KERNEL_CODE, 'vector_add')
    print("✓ Kernel compiled successfully")

    # ========================================================================
    # Step 2: Allocate Device Memory
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Allocate Device Memory")
    print("=" * 60)

    # Allocate memory on GPU (similar to cuMemAlloc in CUDA Python)
    a_gpu = cp.empty(n, dtype=cp.float32)
    b_gpu = cp.empty(n, dtype=cp.float32)
    c_gpu = cp.empty(n, dtype=cp.float32)

    memory_allocated = 3 * n * 4 / 1e6  # 3 arrays, 4 bytes per float
    print(f"✓ Allocated {memory_allocated:.1f} MB on GPU")
    print(f"  - Array A: {a_gpu.data.ptr:x}")
    print(f"  - Array B: {b_gpu.data.ptr:x}")
    print(f"  - Array C: {c_gpu.data.ptr:x}")

    # ========================================================================
    # Step 3: Initialize Host Data
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Initialize Host (CPU) Data")
    print("=" * 60)

    # Create data on CPU
    a_cpu = np.random.randn(n).astype(np.float32)
    b_cpu = np.random.randn(n).astype(np.float32)
    print(f"✓ Initialized {n:,} random elements on CPU")

    # ========================================================================
    # Step 4: Copy Data to Device
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Copy Data from Host to Device")
    print("=" * 60)

    # Copy data from CPU to GPU (similar to cuMemcpyHtoD)
    start = time.time()
    a_gpu[:] = cp.asarray(a_cpu)
    b_gpu[:] = cp.asarray(b_cpu)
    cp.cuda.Stream.null.synchronize()
    transfer_time = time.time() - start

    print(f"✓ Transferred {2*n*4/1e6:.1f} MB to GPU in {transfer_time:.4f}s")
    print(f"  Bandwidth: {2*n*4/1e6/transfer_time:.1f} MB/s")

    # ========================================================================
    # Step 5: Configure and Launch Kernel
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Launch CUDA Kernel")
    print("=" * 60)

    # Configure kernel launch parameters
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    print(f"Grid configuration:")
    print(f"  - Threads per block: {threads_per_block}")
    print(f"  - Blocks per grid: {blocks_per_grid}")
    print(f"  - Total threads: {threads_per_block * blocks_per_grid:,}")

    # Launch kernel (similar to cuLaunchKernel)
    start = time.time()
    vector_add_kernel(
        (blocks_per_grid,),      # Grid dimensions
        (threads_per_block,),    # Block dimensions
        (a_gpu, b_gpu, c_gpu, n) # Kernel arguments
    )
    cp.cuda.Stream.null.synchronize()  # Wait for kernel to complete
    kernel_time = time.time() - start

    print(f"✓ Kernel executed in {kernel_time:.4f}s")

    # ========================================================================
    # Step 6: Copy Results Back to Host
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 6: Copy Results from Device to Host")
    print("=" * 60)

    # Copy result from GPU to CPU (similar to cuMemcpyDtoH)
    start = time.time()
    c_cpu = cp.asnumpy(c_gpu)
    cp.cuda.Stream.null.synchronize()
    transfer_back_time = time.time() - start

    print(f"✓ Transferred {n*4/1e6:.1f} MB from GPU in {transfer_back_time:.4f}s")
    print(f"  Bandwidth: {n*4/1e6/transfer_back_time:.1f} MB/s")

    # ========================================================================
    # Step 7: Verify Results
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 7: Verify Correctness")
    print("=" * 60)

    # Compare with CPU computation
    c_expected = a_cpu + b_cpu
    max_error = np.max(np.abs(c_cpu - c_expected))

    print(f"First 5 results:")
    print(f"  GPU:      {c_cpu[:5]}")
    print(f"  Expected: {c_expected[:5]}")
    print(f"Max error: {max_error}")
    print(f"✓ Results are {'CORRECT' if max_error < 1e-5 else 'INCORRECT'}")

    # ========================================================================
    # Performance Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)

    total_time = transfer_time + kernel_time + transfer_back_time
    print(f"H2D transfer:  {transfer_time:.4f}s ({transfer_time/total_time*100:.1f}%)")
    print(f"Kernel exec:   {kernel_time:.4f}s ({kernel_time/total_time*100:.1f}%)")
    print(f"D2H transfer:  {transfer_back_time:.4f}s ({transfer_back_time/total_time*100:.1f}%)")
    print(f"{'─'*60}")
    print(f"Total time:    {total_time:.4f}s")

    # Compare with CPU
    start = time.time()
    _ = a_cpu + b_cpu
    cpu_time = time.time() - start

    print(f"\nCPU time:      {cpu_time:.4f}s")
    print(f"Speedup (kernel only): {cpu_time/kernel_time:.1f}x")
    print(f"Speedup (including transfers): {cpu_time/total_time:.1f}x")

    # ========================================================================
    # Key Insights
    # ========================================================================
    print("\n" + "=" * 60)
    print("Key Insights")
    print("=" * 60)
    print("1. Low-level control: Explicit memory management and kernel launch")
    print("2. Transfer overhead: Data movement can dominate simple operations")
    print("3. Best for: Complex kernels where compute >> transfer time")
    print("4. Verbosity: More code than CuPy, but maximum control")

    # ========================================================================
    # Memory Cleanup
    # ========================================================================
    print("\n" + "=" * 60)
    print("Cleanup")
    print("=" * 60)

    # In real CUDA Python, you'd explicitly free memory with cuMemFree
    # Here, CuPy's garbage collector handles it
    del a_gpu, b_gpu, c_gpu
    print("✓ Device memory freed")

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
        print("\nFor true CUDA Python (cuda-python package):")
        print("  - You would use cuda.core for higher-level interface")
        print("  - Or cuda.cuda for direct driver API access")
        raise
