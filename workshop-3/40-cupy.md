# CuPy

CuPy is an array library designed for GPU-accelerated computing with Python. It provides a seamless NumPy/SciPy-compatible API for performing operations on NVIDIA CUDA (or AMD ROCm, experimentally) platforms.


## Overview

CuPy is built on the philosophy of minimal code changes for maximum acceleration. By providing a drop-in replacement for NumPy arrays that transparently execute on the GPU, CuPy allows developers to leverage GPU computing power without learning CUDA programming or significantly refactoring existing code.

Key advantages:

- Nearly identical API to NumPy/SciPy - often just changing `numpy` to `cupy` is sufficient
- Automatic GPU execution of array operations
- Access to optimized CUDA libraries (cuBLAS, cuFFT, cuSPARSE, cuDNN)
- Support for custom kernels when needed
- Excellent interoperability with other GPU libraries


## History

- **2015**: Initial release as an open-source project
- **2017**: Added comprehensive SciPy compatibility
- **2018**: Became part of the Chainer deep learning framework ecosystem
- **2019**: Standalone development accelerated; expanded beyond deep learning use cases
- **2020-Present**: Continuous expansion of NumPy/SciPy coverage, improved performance, and broader hardware support

CuPy is maintained by Preferred Networks and the community, with widespread adoption in scientific computing, data science, and machine learning. It's now a NumFOCUS affiliated project.


## What CuPy Does

**Core Functionality:**

- Drop-in NumPy/SciPy replacement for GPU acceleration
- Minimal code changes required (often just `import cupy as cp`)
- Complete API coverage for most NumPy/SciPy operations
- GPU acceleration without deep GPU programming knowledge

**Array Management:**

- `cupy.ndarray` class replaces `numpy.ndarray`
- Arrays allocated directly on GPU device memory
- Automatic device management and synchronization
- Support for multi-GPU systems

**GPU Execution:**

- Automatic GPU execution of NumPy operations
- Element-wise operations as Universal Functions (ufuncs)
- Leverages optimized CUDA libraries:
  - cuBLAS: Linear algebra operations
  - cuFFT: Fast Fourier transforms
  - cuSPARSE: Sparse matrix operations
  - cuDNN: Deep learning primitives
  - Thrust, CUB, cuTENSOR: Additional optimized operations

**Custom Kernel Support:**

- ElementwiseKernel: Custom element-wise operations
- ReductionKernel: Custom reduction operations
- RawKernel: Import existing CUDA C/C++ code
- Kernel Fusion: Automatically fuse multiple operations into single kernel

**Interoperability:**

- CUDA Array Interface: Zero-copy exchange with CUDA libraries
- DLPack: Data exchange with PyTorch, TensorFlow, JAX
- NumPy compatibility: Seamless conversion between CPU and GPU
- Numba integration: Use CuPy arrays in Numba kernels




## Memory Management

CuPy provides sophisticated memory management features to optimize GPU memory usage and performance.

### Memory Pools

CuPy uses memory pools by default to reduce the overhead of GPU memory allocation and deallocation.

**Example: Understanding Memory Pools**

```python
import cupy as cp
import numpy as np

# CuPy uses a memory pool by default
mempool = cp.get_default_memory_pool()        # GPU memory

# Check memory usage before allocation
print(f"Used memory: {mempool.used_bytes() / 1e9:.2f} GB")
print(f"Total memory: {mempool.total_bytes() / 1e9:.2f} GB")

# Allocate arrays - memory comes from pool
a = cp.random.randn(10000, 10000)  # ~800 MB
b = cp.random.randn(10000, 10000)  # ~800 MB

print(f"After allocation - Used: {mempool.used_bytes() / 1e9:.2f} GB")

# Delete arrays - memory returned to pool, not freed
del a, b

print(f"After deletion - Used: {mempool.used_bytes() / 1e9:.2f} GB")
print(f"Pool still holds: {mempool.total_bytes() / 1e9:.2f} GB")

# Free all unused memory blocks
mempool.free_all_blocks()

print(f"After free_all_blocks - Total: {mempool.total_bytes() / 1e9:.2f} GB")
```

**Key Points:**

- Memory pool reduces allocation overhead
- Deleted arrays return memory to pool, not to GPU
- Use `free_all_blocks()` to actually free GPU memory
- Useful for managing memory in long-running processes

**Custom Memory Pool:**

```python
import cupy as cp

# Create custom memory pool using Unified Memory (managed memory)
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)

# Or disable memory pool entirely
cp.cuda.set_allocator(cp.cuda.malloc)
```


Default CuPy Memory Pool (`cp.cuda.malloc`):

- GPU-only device memory
- Best performance for pure GPU computation
- Requires explicit CPU↔GPU transfers
- Recommended for most use cases


Managed Memory Pool (`cp.cuda.malloc_managed`):

- Unified Memory accessible from CPU and GPU
- Automatic data migration
- Simpler programming model
- Potential performance overhead
- Use for CPU-GPU interop or prototyping


## Custom Kernels

While CuPy provides many built-in operations, you can write custom CUDA kernels for specialized computations.

### ElementwiseKernel

For element-wise operations not available in CuPy.

**Example: Custom Sigmoid Activation**

```python
import cupy as cp

# Define custom elementwise kernel
sigmoid_kernel = cp.ElementwiseKernel(
    'float32 x',           # Input type and name
    'float32 y',           # Output type and name
    'y = 1.0f / (1.0f + expf(-x))',  # Operation (CUDA C syntax)
    'sigmoid'              # Kernel name
)

# Use the kernel
x = cp.random.randn(1000000, dtype=cp.float32)
y = sigmoid_kernel(x)

# Compare with pure CuPy
y_cupy = 1.0 / (1.0 + cp.exp(-x))

print(f"Results match: {cp.allclose(y, y_cupy)}")
```

**Benefits:**

- Fuses operations into single kernel (faster)
- CUDA C syntax for complex operations
- Automatic memory management

### ReductionKernel

For custom reduction operations (sum, max, etc.).

**Example: Weighted Sum**

```python
import cupy as cp

# Custom weighted sum reduction
weighted_sum_kernel = cp.ReductionKernel(
    'float32 x, float32 w',    # Input arguments
    'float32 y',                # Output type
    'x * w',                    # Map operation (per element)
    'a + b',                    # Reduce operation (combine results)
    'y = a',                    # Post-reduction
    '0',                        # Identity value
    'weighted_sum'              # Kernel name
)

# Use the kernel
data = cp.random.randn(1000000, dtype=cp.float32)
weights = cp.random.rand(1000000, dtype=cp.float32)

result = weighted_sum_kernel(data, weights)

# Verify against pure CuPy
result_cupy = cp.sum(data * weights)
print(f"Weighted sum: {result}")
print(f"Results match: {cp.allclose(result, result_cupy)}")
```

### RawKernel

For importing existing CUDA C/C++ code.

**Example: Custom Matrix Add with RawKernel**

```python
import cupy as cp

# CUDA C code as string
cuda_code = '''
extern "C" __global__
void matrix_add(const float* a, const float* b, float* c, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}
'''

# Compile and load kernel
matrix_add_kernel = cp.RawKernel(cuda_code, 'matrix_add')

# Prepare data
n = 1000000
a = cp.random.randn(n, dtype=cp.float32)
b = cp.random.randn(n, dtype=cp.float32)
c = cp.empty_like(a)

# Launch kernel
threads_per_block = 256
blocks = (n + threads_per_block - 1) // threads_per_block
matrix_add_kernel((blocks,), (threads_per_block,), (a, b, c, n))

# Verify
print(f"Results match: {cp.allclose(c, a + b)}")
```

**Use cases:**

- Integrating existing CUDA code
- Complex algorithms needing full CUDA control
- Performance-critical custom kernels


## Kernel Fusion

CuPy automatically fuses multiple operations into a single kernel for better performance.

**Example: Automatic Kernel Fusion**

```python
import cupy as cp
import time

# Create large arrays
x = cp.random.randn(10000000, dtype=cp.float32)
y = cp.random.randn(10000000, dtype=cp.float32)

# Multiple operations - CuPy will fuse them
with cp.fuse():
    # All operations fused into single kernel
    result = (x * 2.0 + y * 3.0) / (x + y + 1.0)
    cp.cuda.Stream.null.synchronize()

# Without fusion (for comparison): launches multiple kernels
result_unfused = (x * 2.0 + y * 3.0) / (x + y + 1.0)

print(f"Results match: {cp.allclose(result, result_unfused)}")

# Demonstrate fusion benefit
def with_fusion():
    with cp.fuse():
        z = (x * 2.0 + y * 3.0) / (x + y + 1.0)
        return z

def without_fusion():
    z = (x * 2.0 + y * 3.0) / (x + y + 1.0)
    return z

# Benchmark (warm up first)
_ = with_fusion()
_ = without_fusion()

# Measure
start = time.time()
for _ in range(100):
    _ = with_fusion()
cp.cuda.Stream.null.synchronize()
fused_time = time.time() - start

start = time.time()
for _ in range(100):
    _ = without_fusion()
cp.cuda.Stream.null.synchronize()
unfused_time = time.time() - start

print(f"With fusion: {fused_time:.3f}s")
print(f"Without fusion: {unfused_time:.3f}s")
print(f"Speedup: {unfused_time/fused_time:.2f}x")
```

**Benefits:**

- Reduces kernel launch overhead
- Minimizes memory transfers
- Automatic optimization


## Multi-GPU Support

CuPy makes it easy to use multiple GPUs for parallel computation.

**Example: Computing on Multiple GPUs**

```python
import cupy as cp
import numpy as np

# Check available GPUs
n_gpus = cp.cuda.runtime.getDeviceCount()
print(f"Available GPUs: {n_gpus}")

def compute_on_device(device_id, data):
    """Compute on specific GPU"""
    with cp.cuda.Device(device_id):
        # This computation happens on the specified GPU
        gpu_data = cp.asarray(data)
        result = cp.sum(gpu_data ** 2)
        return result.get()  # Transfer back to CPU

# Split work across GPUs
data = np.random.randn(10000000)
chunk_size = len(data) // n_gpus

results = []
for i in range(n_gpus):
    start = i * chunk_size
    end = start + chunk_size if i < n_gpus - 1 else len(data)
    chunk = data[start:end]

    result = compute_on_device(i, chunk)
    results.append(result)

total = sum(results)
print(f"Total sum of squares: {total}")

# Verify against single GPU
with cp.cuda.Device(0):
    gpu_data = cp.asarray(data)
    expected = cp.sum(gpu_data ** 2).get()
    print(f"Results match: {np.isclose(total, expected)}")
```

**Key Points:**

- `cp.cuda.Device(id)` context manager selects GPU
- Each GPU has its own memory space
- Results need explicit transfer between GPUs or CPU


## No Multi-Core CPU Support

CuPy is fundamentally a GPU-accelerated library and does NOT have native multi-core CPU parallelization capabilities.

### CuPy's Execution Model

**GPU-Only Design:**

- CuPy operations execute on NVIDIA GPUs via CUDA
- No built-in support for multi-threaded CPU execution
- Arrays reside in GPU memory, operations run on GPU cores
- Designed for GPU parallelism, not CPU multi-threading

**CPU Fallback via NumPy:**

When GPU is unavailable or for specific workflows, you can use NumPy as a CPU fallback, but this is just standard NumPy behavior:

```python
import numpy as np
import cupy as cp

# Try GPU first, fall back to NumPy for CPU
try:
    xp = cp  # Use CuPy if available
    x = xp.random.randn(1000000)
    print("Using GPU acceleration")
except:
    xp = np  # Fall back to NumPy
    x = xp.random.randn(1000000)
    print("Using CPU (NumPy)")

# Code works with either
result = xp.sum(x ** 2)
```

**NumPy's CPU Parallelization:**

- NumPy itself is mostly single-threaded for element-wise operations
- Some operations (matrix multiplication via BLAS) can be multi-threaded
- This depends on the underlying BLAS library (OpenBLAS, MKL, etc.)
- Not under CuPy's control - it's just NumPy's native behavior



### When to Use What

**Use CuPy when:**

- You have access to NVIDIA GPU
- Your data is large enough to benefit from GPU parallelism
- You want to accelerate NumPy code with minimal changes
- GPU memory is sufficient for your dataset

**Use Numba for CPU multi-core when:**

- No GPU available but you have multi-core CPU
- Custom algorithms with loops that need CPU parallelization
- You want same code to run on both CPU (multi-threaded) and GPU
- Fine-grained control over parallelization strategy

**Use both together:**

Many workflows combine CuPy and Numba:

```python
import cupy as cp
from numba import cuda

# CuPy for high-level operations
data = cp.random.randn(1000000, dtype=cp.float32)
result = cp.fft.fft(data)  # CuPy FFT on GPU

# Numba CUDA kernel for custom operation on CuPy array
@cuda.jit
def custom_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] = arr[idx] * 2.0

threads = 256
blocks = (data.size + threads - 1) // threads
custom_kernel[blocks, threads](result)
```

### Key Takeaways

**CuPy's parallelization:**

- GPU-only: thousands of GPU threads, not CPU cores
- No native multi-core CPU support
- Falls back to NumPy (mostly single-threaded) on CPU
- Designed exclusively for GPU acceleration


**Bottom line:** CuPy is a GPU acceleration library, not a multi-core CPU parallelization library. If you need multi-core CPU parallelism for array operations, use Numba or other CPU-focused tools.


## Interoperability

CuPy seamlessly exchanges data with other GPU libraries through standard protocols.

### DLPack: Zero-Copy Exchange

**Example: CuPy-PyTorch**

```python
import cupy as cp
import torch

# CuPy to PyTorch (zero-copy)
cupy_array = cp.random.randn(1000, 1000, dtype=cp.float32)
print(f"CuPy array on GPU: {cupy_array.device}")

# Convert to PyTorch tensor (zero-copy via DLPack)
torch_tensor = torch.utils.dlpack.from_dlpack(cupy_array.toDlpack())
print(f"PyTorch tensor on: {torch_tensor.device}")

# Modify PyTorch tensor - affects CuPy array (same memory)
torch_tensor[0, 0] = 999.0
print(f"CuPy array modified: {cupy_array[0, 0]}")  # Shows 999.0

# PyTorch to CuPy (zero-copy)
torch_tensor = torch.randn(500, 500, device='cuda')
cupy_array = cp.from_dlpack(torch_tensor)

print(f"Zero-copy: same memory? {cupy_array.data.ptr == torch_tensor.data_ptr()}")
```

### CUDA Array Interface

**Example: CuPy-Numba**

```python
import cupy as cp
from numba import cuda

# CuPy array
cupy_array = cp.arange(1000000, dtype=cp.float32)

# Use in Numba kernel via CUDA Array Interface
@cuda.jit
def multiply_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] *= 2.0

# Launch Numba kernel on CuPy array (zero-copy)
threads = 256
blocks = (cupy_array.size + threads - 1) // threads
multiply_kernel[blocks, threads](cupy_array)

print(f"First elements: {cupy_array[:5]}")  # [0, 2, 4, 6, 8]
```

**Benefits:**

- Zero-copy data sharing
- No format conversion overhead
- Seamless integration across libraries



## Examples

### Example 1: Vector Addition (NumPy Drop-in Replacement)

Vector addition demonstrates CuPy's core philosophy: minimal code changes for maximum GPU acceleration.

**Key Concepts:**

- Drop-in replacement for NumPy
- Automatic GPU parallelization
- Memory transfer between CPU and GPU
- Performance comparison with NumPy

**Implementation:**

```python
import numpy as np
import cupy as cp
import time

# Vector size
n = 50_000_000

# NumPy (CPU) version
print("=" * 50)
print("NumPy (CPU) Version")
print("=" * 50)

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

# CuPy (GPU) version
print("\n" + "=" * 50)
print("CuPy (GPU) Version")
print("=" * 50)

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

# Speedup calculation
print("\n" + "=" * 50)
print("Performance Comparison")
print("=" * 50)
print(f"CPU time: {cpu_time:.4f}s")
print(f"GPU time: {gpu_time:.4f}s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x faster")

# Including memory transfer overhead
print("\n" + "=" * 50)
print("Including Memory Transfer Overhead")
print("=" * 50)

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

# Verify correctness
print("\n" + "=" * 50)
print("Correctness Check")
print("=" * 50)
# Compare small portion (avoid memory)
test_size = 1000
a_test = np.random.randn(test_size).astype(np.float32)
b_test = np.random.randn(test_size).astype(np.float32)

c_numpy = a_test + b_test
c_cupy = cp.asnumpy(cp.asarray(a_test) + cp.asarray(b_test))

print(f"Results match: {np.allclose(c_numpy, c_cupy)}")
print(f"Max difference: {np.max(np.abs(c_numpy - c_cupy))}")
```

**Key Points:**

- **Minimal code change**: Just replace `numpy` with `cupy`
- **Automatic GPU execution**: The `+` operator automatically triggers GPU kernel
- **Memory transfers**: Use `cp.asarray()` (CPU→GPU) and `cp.asnumpy()` (GPU→CPU)
- **Synchronization**: `synchronize()` ensures GPU finishes before timing
- **Transfer overhead**: For small operations, transfer time can dominate


**When CuPy Shines:**

- Large arrays (transfer cost amortized)
- Multiple operations on same GPU data
- Already have data on GPU
- Iterative algorithms keeping data on GPU

**Best Practice:**
```python
# Good: Keep data on GPU for multiple operations
a_gpu = cp.asarray(a_cpu)  # Transfer once
b_gpu = cp.asarray(b_cpu)

# Many operations without transfer
c_gpu = a_gpu + b_gpu
d_gpu = c_gpu * 2.0
e_gpu = cp.sin(d_gpu)

result = cp.asnumpy(e_gpu)  # Transfer once at end
```


### Example 2: Matrix Multiplication (cuBLAS Performance)


Large-scale matrix multiplication is fundamental to scientific computing, appearing in linear algebra, deep learning, physics simulations, and data analysis. CuPy makes it trivial to accelerate this operation.

- **One-line change**: `numpy` $\rightarrow$ `cupy`
- **cuBLAS library**: Uses NVIDIA's highly optimized BLAS implementation
- **Perfect for CuPy**: Standard operation where libraries beat custom kernels

**Implementation:**

```python
import numpy as np
import cupy as cp
import time

def benchmark_matmul(size, dtype=np.float32):
    """Benchmark matrix multiplication on CPU vs GPU"""

    print(f"\n{'='*60}")
    print(f"Matrix Multiplication: {size}x{size} ({dtype.__name__})")
    print(f"{'='*60}")

    # ------------ NumPy (CPU) version --------------
    print("\nNumPy (CPU):")
    print("-" * 40)

    A_cpu = np.random.randn(size, size).astype(dtype)
    B_cpu = np.random.randn(size, size).astype(dtype)

    # Warm-up
    _ = A_cpu @ B_cpu

    # Benchmark
    start = time.time()
    C_cpu = A_cpu @ B_cpu
    cpu_time = time.time() - start

    memory_mb = (size * size * 4) / 1e6  # 4 bytes per float32
    flops = 2 * size**3  # Matrix multiply operations
    cpu_gflops = flops / cpu_time / 1e9

    print(f"Time: {cpu_time:.4f} seconds")
    print(f"Memory per matrix: {memory_mb:.1f} MB")
    print(f"Performance: {cpu_gflops:.1f} GFLOPS")

    # ---------- CuPy (GPU) version ---------------  
    print("\nCuPy (GPU):")
    print("-" * 40)

    A_gpu = cp.random.randn(size, size).astype(dtype)
    B_gpu = cp.random.randn(size, size).astype(dtype)

    # Warm-up (includes cuBLAS initialization)
    _ = A_gpu @ B_gpu
    cp.cuda.Stream.null.synchronize()

    # Benchmark
    start = time.time()
    C_gpu = A_gpu @ B_gpu
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start

    gpu_gflops = flops / gpu_time / 1e9

    print(f"Time: {gpu_time:.4f} seconds")
    print(f"Memory per matrix: {memory_mb:.1f} MB")
    print(f"Performance: {gpu_gflops:.1f} GFLOPS")

    # Comparison
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"CPU: {cpu_time:.4f}s ({cpu_gflops:.1f} GFLOPS)")
    print(f"GPU: {gpu_time:.4f}s ({gpu_gflops:.1f} GFLOPS)")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    print(f"GPU is {gpu_gflops/cpu_gflops:.1f}x more efficient")

    # Verify correctness (small sample)
    if size <= 1000:
        A_test = A_cpu[:100, :100]
        B_test = B_cpu[:100, :100]
        C_numpy = A_test @ B_test
        C_cupy = cp.asnumpy(cp.asarray(A_test) @ cp.asarray(B_test))
        print(f"\nCorrectness: {np.allclose(C_numpy, C_cupy, rtol=1e-5)}")

# Run benchmarks for different sizes
print("Matrix Multiplication Benchmarks")
print("=" * 60)

# Small matrices
benchmark_matmul(1000, np.float32)

# Medium matrices
benchmark_matmul(5000, np.float32)

# Large matrices (may need significant GPU memory)
# benchmark_matmul(10000, np.float32)

# Compare float32 vs float64
print("\n\n" + "=" * 60)
print("Data Type Comparison (5000x5000)")
print("=" * 60)

benchmark_matmul(5000, np.float32)
benchmark_matmul(5000, np.float64)
```

**Additional Linear Algebra Operations:**

CuPy accelerates many linear algebra operations through cuBLAS and cuSOLVER:

```python
import numpy as np
import cupy as cp
import time

n = 5000

# Create symmetric positive definite matrix
A_cpu = np.random.randn(n, n).astype(np.float32)
A_cpu = A_cpu @ A_cpu.T + np.eye(n) * n  # Make positive definite
A_gpu = cp.asarray(A_cpu)

print("Linear Algebra Operations Comparison")
print("=" * 60)

# 1. Eigenvalue decomposition
print("\nEigenvalue Decomposition:")
start = time.time()
eigvals_cpu, eigvecs_cpu = np.linalg.eigh(A_cpu)
cpu_time = time.time() - start
print(f"  NumPy (CPU): {cpu_time:.3f}s")

start = time.time()
eigvals_gpu, eigvecs_gpu = cp.linalg.eigh(A_gpu)
cp.cuda.Stream.null.synchronize()
gpu_time = time.time() - start
print(f"  CuPy (GPU): {gpu_time:.3f}s")
print(f"  Speedup: {cpu_time/gpu_time:.1f}x")

# 2. SVD (Singular Value Decomposition)
print("\nSingular Value Decomposition:")
B_cpu = np.random.randn(n, n).astype(np.float32)
B_gpu = cp.asarray(B_cpu)

start = time.time()
U_cpu, s_cpu, Vt_cpu = np.linalg.svd(B_cpu)
cpu_time = time.time() - start
print(f"  NumPy (CPU): {cpu_time:.3f}s")

start = time.time()
U_gpu, s_gpu, Vt_gpu = cp.linalg.svd(B_gpu)
cp.cuda.Stream.null.synchronize()
gpu_time = time.time() - start
print(f"  CuPy (GPU): {gpu_time:.3f}s")
print(f"  Speedup: {cpu_time/gpu_time:.1f}x")

# 3. Solving linear systems
print("\nSolving Linear System (Ax = b):")
b_cpu = np.random.randn(n).astype(np.float32)
b_gpu = cp.asarray(b_cpu)

start = time.time()
x_cpu = np.linalg.solve(A_cpu, b_cpu)
cpu_time = time.time() - start
print(f"  NumPy (CPU): {cpu_time:.3f}s")

start = time.time()
x_gpu = cp.linalg.solve(A_gpu, b_gpu)
cp.cuda.Stream.null.synchronize()
gpu_time = time.time() - start
print(f"  CuPy (GPU): {gpu_time:.3f}s")
print(f"  Speedup: {cpu_time/gpu_time:.1f}x")

# 4. Matrix inversion
print("\nMatrix Inversion:")
start = time.time()
A_inv_cpu = np.linalg.inv(A_cpu)
cpu_time = time.time() - start
print(f"  NumPy (CPU): {cpu_time:.3f}s")

start = time.time()
A_inv_gpu = cp.linalg.inv(A_gpu)
cp.cuda.Stream.null.synchronize()
gpu_time = time.time() - start
print(f"  CuPy (GPU): {gpu_time:.3f}s")
print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
```


**Why CuPy Dominates Here:**

- **cuBLAS**: NVIDIA's highly tuned BLAS library
- **Hardware acceleration**: Tensor cores on modern GPUs
- **Memory bandwidth**: GPU memory is much faster
- **Parallelism**: Thousands of GPU cores vs handful of CPU cores

**Perfect Drop-in Replacement:**

```python
# Original NumPy code
import numpy as np
A = np.random.randn(5000, 5000)
B = np.random.randn(5000, 5000)
C = A @ B
eigenvalues, eigenvectors = np.linalg.eigh(C)

# GPU-accelerated version (just change import!)
import cupy as cp
A = cp.random.randn(5000, 5000)
B = cp.random.randn(5000, 5000)
C = A @ B
eigenvalues, eigenvectors = cp.linalg.eigh(C)
```

**Key Takeaways:**

- **Minimal effort**: Often just changing the import statement
- **Maximum performance**: Leverages highly optimized GPU libraries
- **No expertise needed**: Don't need to understand GPU programming
- **Ideal for standard operations**: Where libraries beat custom code


## Installation

**Basic Installation:**
```bash
python -m venv .venv
source .venv/bin/activate

# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

**Check Installation:**
```bash
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

**Verify cuBLAS/cuFFT:**
```bash
python -c "import cupy as cp; a = cp.random.randn(100, 100); print((a @ a).shape)"
```