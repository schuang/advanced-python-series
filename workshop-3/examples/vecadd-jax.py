#!/usr/bin/env python3
"""
JAX Vector Addition Example
Demonstrates JAX's JIT compilation and performance characteristics.

This example shows:
- JAX as a NumPy-compatible array library
- JIT compilation with jax.jit()
- Performance comparison with NumPy
- Immutable arrays and functional programming
- Device management (CPU/GPU)
"""

import numpy as np
import jax
import jax.numpy as jnp
import time


def vector_add_numpy(a, b):
    """NumPy vector addition"""
    return a + b


def vector_add_jax(a, b):
    """JAX vector addition (not compiled)"""
    return a + b


# JIT-compiled version
@jax.jit
def vector_add_jax_jit(a, b):
    """JAX vector addition (JIT compiled)"""
    return a + b


def main():
    # Vector size
    n = 50_000_000

    print("=" * 60)
    print("JAX Vector Addition Example")
    print("=" * 60)

    # Check available devices
    print("\n" + "=" * 60)
    print("Available Devices")
    print("=" * 60)
    devices = jax.devices()
    print(f"JAX version: {jax.__version__}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Devices: {devices}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device.device_kind} - {device}")

    # ========================================================================
    # NumPy (CPU) Version - Baseline
    # ========================================================================
    print("\n" + "=" * 60)
    print("NumPy (CPU) Version")
    print("=" * 60)

    # Create arrays on CPU
    a_numpy = np.random.randn(n).astype(np.float32)
    b_numpy = np.random.randn(n).astype(np.float32)

    # Time CPU execution
    start = time.time()
    c_numpy = vector_add_numpy(a_numpy, b_numpy)
    numpy_time = time.time() - start

    print(f"Array size: {n:,} elements ({n*4/1e6:.1f} MB per array)")
    print(f"NumPy time: {numpy_time:.4f} seconds")
    print(f"First 5 elements: {c_numpy[:5]}")

    # ========================================================================
    # JAX (Uncompiled) Version
    # ========================================================================
    print("\n" + "=" * 60)
    print("JAX (Uncompiled) Version")
    print("=" * 60)

    # Convert to JAX arrays (will use default device)
    a_jax = jnp.array(a_numpy)
    b_jax = jnp.array(b_numpy)

    print(f"Arrays on device: {a_jax.device}")

    # Time uncompiled JAX execution
    start = time.time()
    c_jax_uncompiled = vector_add_jax(a_jax, b_jax)
    # Block until computation is done (JAX is asynchronous)
    c_jax_uncompiled.block_until_ready()
    jax_uncompiled_time = time.time() - start

    print(f"JAX (uncompiled) time: {jax_uncompiled_time:.4f} seconds")
    print(f"First 5 elements: {np.array(c_jax_uncompiled[:5])}")

    # ========================================================================
    # JAX (JIT Compiled) Version - First Call
    # ========================================================================
    print("\n" + "=" * 60)
    print("JAX (JIT Compiled) Version - First Call")
    print("=" * 60)
    print("Note: First call includes compilation overhead\n")

    # First call: includes compilation time
    start = time.time()
    c_jax_jit = vector_add_jax_jit(a_jax, b_jax)
    c_jax_jit.block_until_ready()
    jax_jit_first_time = time.time() - start

    print(f"JAX (JIT, first call) time: {jax_jit_first_time:.4f} seconds")
    print(f"  (includes compilation overhead)")
    print(f"First 5 elements: {np.array(c_jax_jit[:5])}")

    # ========================================================================
    # JAX (JIT Compiled) Version - Subsequent Calls
    # ========================================================================
    print("\n" + "=" * 60)
    print("JAX (JIT Compiled) Version - Subsequent Calls")
    print("=" * 60)
    print("Note: Uses cached compiled version\n")

    # Subsequent calls: use cached compiled version
    start = time.time()
    c_jax_jit = vector_add_jax_jit(a_jax, b_jax)
    c_jax_jit.block_until_ready()
    jax_jit_time = time.time() - start

    print(f"JAX (JIT, cached) time: {jax_jit_time:.4f} seconds")
    print(f"First 5 elements: {np.array(c_jax_jit[:5])}")

    # ========================================================================
    # Performance Comparison
    # ========================================================================
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    print(f"NumPy (CPU):           {numpy_time:.4f}s (baseline)")
    print(f"JAX (uncompiled):      {jax_uncompiled_time:.4f}s ({numpy_time/jax_uncompiled_time:.2f}x)")
    print(f"JAX (JIT, first call): {jax_jit_first_time:.4f}s ({numpy_time/jax_jit_first_time:.2f}x)")
    print(f"JAX (JIT, cached):     {jax_jit_time:.4f}s ({numpy_time/jax_jit_time:.2f}x)")
    print(f"\nJIT compilation overhead: {jax_jit_first_time - jax_jit_time:.4f}s")

    # ========================================================================
    # Including Data Transfer (NumPy to JAX)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Including Data Transfer Overhead")
    print("=" * 60)

    # Start from NumPy arrays, convert to JAX, compute, convert back
    start = time.time()
    a_transfer = jnp.array(a_numpy)  # NumPy -> JAX
    b_transfer = jnp.array(b_numpy)  # NumPy -> JAX
    c_transfer = vector_add_jax_jit(a_transfer, b_transfer)
    result = np.array(c_transfer)  # JAX -> NumPy
    total_time = time.time() - start

    print(f"Total time (with transfers): {total_time:.4f}s")
    print(f"Pure JAX compute: {jax_jit_time:.4f}s")
    print(f"Transfer overhead: {total_time - jax_jit_time:.4f}s")
    print(f"Speedup (with transfers): {numpy_time/total_time:.2f}x")

    # ========================================================================
    # Correctness Check
    # ========================================================================
    print("\n" + "=" * 60)
    print("Correctness Check")
    print("=" * 60)

    # Compare all versions
    test_size = 1000
    a_test = np.random.randn(test_size).astype(np.float32)
    b_test = np.random.randn(test_size).astype(np.float32)

    c_numpy_test = vector_add_numpy(a_test, b_test)
    c_jax_test = np.array(vector_add_jax_jit(jnp.array(a_test), jnp.array(b_test)))

    print(f"NumPy vs JAX match: {np.allclose(c_numpy_test, c_jax_test)}")
    print(f"Max difference: {np.max(np.abs(c_numpy_test - c_jax_test))}")

    # ========================================================================
    # JAX Array Immutability
    # ========================================================================
    print("\n" + "=" * 60)
    print("JAX Array Immutability")
    print("=" * 60)

    print("\nJAX arrays are immutable - you cannot modify them in place.\n")

    x = jnp.array([1, 2, 3, 4, 5])
    print(f"Original array: {x}")

    # This would raise an error in JAX:
    # x[0] = 10  # Error! Cannot modify JAX array

    # Instead, use functional updates:
    x_new = x.at[0].set(10)
    print(f"After x.at[0].set(10):")
    print(f"  Original: {x}")
    print(f"  New:      {x_new}")

    # ========================================================================
    # Best Practice: Multiple Operations
    # ========================================================================
    print("\n" + "=" * 60)
    print("Best Practice: Keep Data on Device")
    print("=" * 60)

    # Good: Keep data as JAX arrays for multiple operations
    @jax.jit
    def multi_operation(a, b):
        """Multiple operations fused by JIT"""
        c = a + b
        d = c * 2.0
        e = jnp.sin(d)
        return e

    a_jax = jnp.array(a_numpy)
    b_jax = jnp.array(b_numpy)

    # Warm-up
    _ = multi_operation(a_jax, b_jax).block_until_ready()

    # Time multiple operations
    start = time.time()
    result = multi_operation(a_jax, b_jax)
    result.block_until_ready()
    multi_time = time.time() - start

    print(f"Multiple operations (JIT-fused): {multi_time:.4f}s")
    print("JIT compilation automatically fuses operations for efficiency!")

    # ========================================================================
    # Vectorization with vmap
    # ========================================================================
    print("\n" + "=" * 60)
    print("Vectorization with vmap")
    print("=" * 60)

    # Function for single pair of numbers
    def add_single(x, y):
        return x + y

    # Vectorize over batch dimension
    add_batch = jax.vmap(add_single)

    # Create batches
    batch_size = 1000
    a_batch = jnp.arange(batch_size)
    b_batch = jnp.arange(batch_size) * 2

    result_batch = add_batch(a_batch, b_batch)
    print(f"Vectorized addition of {batch_size} pairs")
    print(f"Input a: {a_batch[:5]} ...")
    print(f"Input b: {b_batch[:5]} ...")
    print(f"Result:  {result_batch[:5]} ...")

    # ========================================================================
    # Device Placement (if GPU available)
    # ========================================================================
    if jax.default_backend() == 'gpu':
        print("\n" + "=" * 60)
        print("Device Placement (GPU Detected)")
        print("=" * 60)

        # Explicitly place on GPU
        with jax.default_device(jax.devices('gpu')[0]):
            a_gpu = jnp.array(a_numpy[:1000000])
            b_gpu = jnp.array(b_numpy[:1000000])
            c_gpu = vector_add_jax_jit(a_gpu, b_gpu)
            print(f"Arrays on device: {c_gpu.device}")
            print("✓ Computation performed on GPU")

    # ========================================================================
    # Key Differences from NumPy
    # ========================================================================
    print("\n" + "=" * 60)
    print("Key Differences from NumPy")
    print("=" * 60)

    print("\n1. Immutable arrays:")
    print("   NumPy: x[0] = 5  # OK")
    print("   JAX:   x = x.at[0].set(5)  # Must create new array")

    print("\n2. Asynchronous execution:")
    print("   JAX operations return immediately")
    print("   Use .block_until_ready() to wait for completion")

    print("\n3. JIT compilation:")
    print("   First call: compilation overhead")
    print("   Subsequent calls: use cached compiled version")

    print("\n4. Device management:")
    print("   NumPy: CPU only")
    print("   JAX: CPU, GPU, or TPU (automatic)")

    print("\n5. Transformations:")
    print("   JAX provides grad(), jit(), vmap(), pmap()")
    print("   NumPy has none of these")

    # ========================================================================
    # When to Use JAX vs Other Frameworks
    # ========================================================================
    print("\n" + "=" * 60)
    print("When to Use JAX")
    print("=" * 60)

    print("\n✓ Use JAX when you need:")
    print("  - Automatic differentiation (gradients)")
    print("  - JIT compilation for performance")
    print("  - Functional transformations (vmap, pmap)")
    print("  - Code that works on CPU, GPU, and TPU")
    print("  - Composable transformations")

    print("\n✗ Consider alternatives when:")
    print("  - Simple NumPy operations (use CuPy for GPU)")
    print("  - Heavy use of side effects")
    print("  - Don't need gradients or transformations")
    print("  - Maximum GPU control needed (use CUDA Python)")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("\nKey JAX Features Demonstrated:")
    print("  1. NumPy-compatible API (jax.numpy)")
    print("  2. JIT compilation with @jax.jit")
    print("  3. Automatic device management (CPU/GPU/TPU)")
    print("  4. Immutable arrays (functional programming)")
    print("  5. Vectorization with vmap()")
    print("  6. Asynchronous execution")

    print("\nPerformance Summary:")
    print(f"  NumPy baseline:    {numpy_time:.4f}s")
    print(f"  JAX JIT (cached):  {jax_jit_time:.4f}s ({numpy_time/jax_jit_time:.2f}x speedup)")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This example requires:")
        print("  - JAX installed: pip install jax")
        print("  - For GPU support: pip install jax[cuda12]")
        print("  - For CPU-only: pip install jax (default)")
        raise
