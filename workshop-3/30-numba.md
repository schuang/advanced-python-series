# Numba

Numba is a Just-in-Time (JIT) compiler for Python designed to accelerate computationally intensive Python code, particularly on hardware accelerators like NVIDIA GPUs.


## Overview

Numba provides a powerful way to speed up Python code without requiring developers to leave the Python ecosystem or learn lower-level languages. It translates Python functions to optimized machine code at runtime using the industry-standard LLVM compiler library. The key advantage is that developers can write in familiar Python syntax while achieving performance comparable to compiled languages like C or Fortran.

Numba is particularly effective for:

- Array-oriented and math-heavy Python code
- Code with many loops that would typically be slow in pure Python
- NumPy operations that need additional acceleration
- Parallel algorithms that can leverage multiple CPU cores or GPU devices


## History

Numba was created by Travis Oliphant, founder of NumPy and Anaconda, and was first released in 2012. The project emerged from the need to accelerate scientific Python code without requiring developers to write C extensions or Cython.

Key milestones:

- **2012**: Initial release focused on CPU JIT compilation
- **2013**: Added CUDA GPU support, enabling Python programmers to write GPU kernels directly
- **2014**: Became part of the NumFOCUS sponsored projects
- **2015-Present**: Continuous improvements in performance, expanded hardware support, and broader Python language coverage

Numba is now maintained by Anaconda Inc. and has become a core component of the PyData ecosystem, with widespread adoption in scientific computing, data science, and machine learning communities.


## What Numba Does

**Core Functionality:**

- Compiler for Python array and numerical functions
- Optimized for numerically-oriented code with loops
- Works exceptionally well with NumPy arrays and functions

**JIT Compilation:**

- Decorators: `@jit`, `@cuda.jit` trigger compilation
- Reads Python bytecode + analyzes input argument types
- Compilation pipeline: **Python $\rightarrow$ LLVM IR $\rightarrow$ Machine code**
  - LLVM IR = Intermediate Representation (platform-independent assembly-like code)
  - Enables optimization and portability across different CPUs/GPUs
- Uses LLVM compiler library for optimization
- Generates native machine code for target CPU or GPU

**Performance Benefits:**

- Achieves C/C++/Fortran-level speeds
- Best performance in `nopython` mode (no Python interpreter overhead)
- Compiles to native machine instructions at runtime

**GPU Target Support:**

- Supports NVIDIA CUDA GPU programming
    - No AMD GPU support
- Write parallel code in pure Python syntax
- Compiles restricted Python subset to CUDA kernels
- Manages CUDA execution model (threads, blocks, grid)


## Use Cases

Numba has become essential in many scientific computing domains where Python's ease of use needs to be combined with high performance:

- Numerical Simulations: N-body simulations, finite element analysis, computational fluid dynamics

- Signal and Image Processing: Custom FFT implementations, wavelet transforms, medical image analysis

- Monte Carlo Methods: Financial risk analysis, quantum Monte Carlo, Bayesian inference

- Machine Learning and AI: Custom gradient computations, reinforcement learning environments, graph neural networks

- Computational Biology: BLAST-like algorithms, gene expression analysis, population genetics simulations

- Optimization Problems: Genetic algorithms, particle swarm optimization, linear programming solvers


## Compilation Modes

Numba offers different compilation modes that balance performance and compatibility.

### Nopython Mode

- The compiled code runs entirely **without the Python interpreter**
- Maximum performance - executes at native machine code speed
- No Python object overhead
- Enabled by default with `@jit` or explicitly with `@jit(nopython=True)`

**Benefits:**

- Fastest execution possible
- Predictable performance
- No Global Interpreter Lock (GIL) limitations
- Can release the GIL for parallel execution

**Limitations:**

- Only supports a subset of Python and NumPy features
- Cannot use arbitrary Python objects
- Limited to Numba-supported operations

**Example:**

```python
from numba import jit

@jit(nopython=True)
def compute_sum(arr):
    total = 0.0
    for i in range(arr.shape[0]):
        total += arr[i]
    return total
```

### Object Mode

- Falls back to Python interpreter for unsupported operations
- Allows use of arbitrary Python objects and libraries
- Enabled automatically when nopython mode fails, or explicitly with `@jit(forceobj=True)`

**When to use:**

- Code contains operations not supported in nopython mode
- Rapid prototyping before optimizing to nopython mode
- Wrapping functions that call other Python libraries

**Performance:**

- Slower than nopython mode
- Still provides some speedup through type specialization
- Python overhead remains present

### Loop-Lifting

**Hybrid approach:**

- Automatically identifies loops that can be compiled in nopython mode
- Compiles those loops to native code even when the function uses object mode
- Provides partial acceleration without full nopython compatibility

**Example:**

```python
from numba import jit

@jit  # Will use loop-lifting
def mixed_function(arr):
    result = []  # Python list (object mode)
    for i in range(len(arr)):  # This loop can be lifted to nopython
        result.append(arr[i] * 2)
    return result
```


## Type Specialization and Compilation Caching

### Type Specialization

- Numba compiles a separate version of the function for each unique combination of input types
- First call with specific types triggers compilation (at runtime)
- Subsequent calls with same types use cached compiled version
- Different type combinations trigger new compilations

**Example:**

```python
from numba import jit
import numpy as np

@jit
def add_arrays(a, b):
    return a + b

# First call: compiles for float64 arrays
x = np.array([1.0, 2.0, 3.0])
result1 = add_arrays(x, x)  # Compilation happens here

# Second call: uses cached version
y = np.array([4.0, 5.0, 6.0])
result2 = add_arrays(y, y)  # Fast, no compilation

# Third call: different types, triggers new compilation
z = np.array([1, 2, 3], dtype=np.int32)
result3 = add_arrays(z, z)  # New compilation for int32
```

**Benefits:**

- Optimal code generation for each type combination
- Amortizes compilation cost over multiple calls
- Automatic optimization without manual type declarations

**Considerations:**

- First-call overhead for each type combination
- Multiple type combinations increase memory usage
- Consider using explicit signatures for critical paths

### Compilation Caching to Disk

- Saves compiled machine code to disk
- Eliminates compilation overhead on subsequent program runs
- Enabled with `cache=True` parameter

**Example:**

```python
from numba import jit

@jit(nopython=True, cache=True)
def expensive_computation(n):
    result = 0
    for i in range(n):
        result += i * i
    return result

# First run: compiles and caches to disk
# Subsequent runs: loads from cache, no compilation
```

**Benefits:**

- Faster startup times for subsequent runs
- Especially valuable for large codebases
- Reduces first-call latency in production

**Cache location:**

- Stored in `__pycache__` directory by default
- Automatically invalidated when source code changes
- Can be cleared manually if needed

**Best practices:**

- Enable caching for production code
- Use for functions with expensive compilation
- Particularly valuable for large GPU kernels


## Vectorization

Numba provides high-level decorators that automatically parallelize functions across array elements without writing explicit loops or kernels.

### @vectorize Decorator

- Converts scalar functions into universal functions (ufuncs)
- Automatically applies the function to each element of arrays
- Supports CPU and GPU targets
- Similar to NumPy's vectorize but much faster

**Basic Usage:**

```python
from numba import vectorize
import numpy as np

@vectorize
def scalar_add(a, b):
    return a + b

# Automatically works on arrays
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 30, 40, 50])
result = scalar_add(x, y)  # [11, 22, 33, 44, 55]
```

**GPU Target:**

```python
from numba import vectorize
import numpy as np

@vectorize(['float64(float64, float64)'], target='cuda')
def gpu_add(a, b):
    return a + b

# Automatically executes on GPU
x = np.random.randn(1000000)
y = np.random.randn(1000000)
result = gpu_add(x, y)
```

**Supported targets:**

- `'cpu'`: Single-threaded CPU execution
- `'parallel'`: Multi-threaded CPU execution
- `'cuda'`: NVIDIA GPU execution

**Benefits:**

- Write scalar logic, get parallel execution
- No explicit loop management
- Clean, readable code
- Easy to switch between CPU and GPU

### @guvectorize Decorator

- Generalized universal functions (gufuncs)
- Operates on subarrays rather than scalars
- Supports reduction operations and complex array manipulations
- More flexible than `@vectorize`

**Example: Moving Average:**

```python
from numba import guvectorize
import numpy as np

@guvectorize(['void(float64[:], intp[:], float64[:])'],
             '(n),()->(n)', target='parallel')
def moving_average(data, window, result):
    for i in range(data.shape[0]):
        start = max(0, i - window[0] + 1)
        result[i] = 0.0
        for j in range(start, i + 1):
            result[i] += data[j]
        result[i] /= (i - start + 1)

# Usage
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
window = np.array([3])
result = np.empty_like(data)
moving_average(data, window, result)
```

**Signature format:**

- Function signature: type specification
- Layout signature: input/output dimensions
- `'(n),()->(n)'` means:
    - input: 1D array of length n, scalar (0-dimensional)
    - output: 1D array of length n

**Use cases:**

- Reduction operations (sum, mean, max over windows)
- Sliding window computations
- Custom array transformations
- Matrix operations on subarrays


## Multi-Core CPU Parallelization

Numba can automatically parallelize code across multiple CPU cores, similar to SMP (Symmetric Multi-Processing) multi-threaded execution. This provides significant speedups on multi-core systems without requiring explicit thread management.

### Automatic Parallelization with parallel=True

Numba can automatically identify parallel loops and distribute iterations across CPU cores.

```python
from numba import jit
import numpy as np
import time

# Sequential version
@jit(nopython=True)
def sum_squares_sequential(arr):
    """Sequential computation"""
    n = arr.shape[0]
    result = np.zeros(n)
    for i in range(n):
        result[i] = arr[i] ** 2
    return result

# Parallel version - Numba automatically parallelizes the loop
@jit(nopython=True, parallel=True)
def sum_squares_parallel(arr):
    """Parallel computation across CPU cores"""
    n = arr.shape[0]
    result = np.zeros(n)
    for i in range(n):
        result[i] = arr[i] ** 2
    return result

# Benchmark
n = 100_000_000
data = np.random.randn(n)

# Sequential
start = time.time()
result_seq = sum_squares_sequential(data)
seq_time = time.time() - start

# Parallel
start = time.time()
result_par = sum_squares_parallel(data)
par_time = time.time() - start

print(f"Sequential time: {seq_time:.3f}s")
print(f"Parallel time: {par_time:.3f}s")
print(f"Speedup: {seq_time/par_time:.2f}x")
print(f"Results match: {np.allclose(result_seq, result_par)}")
```

**Key Features:**

- `parallel=True`: Enables automatic parallelization
- Numba analyzes loops for data dependencies
- Independent iterations are distributed across CPU threads
- Releases the Python Global Interpreter Lock (GIL)
- Works with multiple CPU cores simultaneously

### Explicit Parallel Loops with prange

For more control, use `prange` (parallel range) to explicitly mark loops for parallelization.

```python
from numba import jit, prange
import numpy as np
import time

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

# Usage
n = 50_000_000
a = np.random.randn(n)
b = np.random.randn(n)

# Time the execution
start = time.time()
result = parallel_computation(a, b)
elapsed = time.time() - start

print(f"Time: {elapsed:.3f}s")
print(f"Utilized {np.ceil(n / 1e6 / elapsed):.0f}M operations/second")
```

**prange vs range:**

- `range`: Sequential execution
- `prange`: Parallel execution across CPU cores
    - similiar to OpenMP's "parallel for"
- Only use `prange` when iterations are independent
- Numba handles thread creation and synchronization

### Parallel Reductions

Numba can parallelize reduction operations (sum, mean, etc.) automatically.

```python
from numba import jit, prange
import numpy as np
import time

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

# Benchmark
n = 100_000_000
arr = np.random.randn(n)

# Parallel sum
start = time.time()
par_sum = parallel_sum(arr)
par_time = time.time() - start

# NumPy sum (single-threaded for large arrays)
start = time.time()
np_sum = np.sum(arr)
np_time = time.time() - start

print(f"Parallel sum: {par_sum:.6f} ({par_time:.3f}s)")
print(f"NumPy sum: {np_sum:.6f} ({np_time:.3f}s)")
print(f"Speedup: {np_time/par_time:.2f}x")
```

**How it works:**

- Numba splits the reduction across threads
- Each thread computes partial result
- Results are combined efficiently
- Automatic handling of race conditions

### Nested Parallel Loops

Numba can handle nested parallel loops for 2D operations.

**Example: Parallel Matrix Operations**

```python
from numba import jit, prange
import numpy as np
import time

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

# Benchmark
m, n = 10000, 10000
A = np.random.randn(m, n).astype(np.float32)
B = np.random.randn(m, n).astype(np.float32)

start = time.time()
C = parallel_matrix_op(A, B)
elapsed = time.time() - start

print(f"Matrix size: {m}×{n}")
print(f"Time: {elapsed:.3f}s")
print(f"Throughput: {m*n/elapsed/1e6:.1f} Million ops/sec")
```

**Best practices:**

- Parallelize outer loop, keep inner loop sequential
- Only parallelize one level to avoid overhead
- Ensure sufficient work per thread (avoid too fine-grained parallelism)

Or vectorize the inner loop:

```python
# Outer loop parallelized
    for i in prange(m):
        # Inner operation vectorized (NumPy)
        C[i, :] = np.sqrt(A[i, :]**2 + B[i, :]**2)
```

### When to Use Multi-Core Parallelization

**Good candidates for parallel=True:**

- Large arrays (>10^6 elements)
- Independent operations per element
- Compute-intensive operations (not just memory access)
- Nested loops with substantial work

**Not beneficial for:**

- Small arrays (overhead dominates)
- Sequential dependencies between iterations
- Memory-bound operations (already limited by bandwidth)
- Very simple operations (addition, copying)

**Sequential vs Parallel Performance**

```python
from numba import jit, prange
import numpy as np
import time

def benchmark_sizes():
    """Show when parallel execution helps"""
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

    print(f"{'Size':<12} {'Sequential':<12} {'Parallel':<12} {'Speedup':<12}")
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

benchmark_sizes()
```

### Thread Control

You can control the number of threads Numba uses for parallel execution.

```python
from numba import jit, prange, set_num_threads, get_num_threads
import numpy as np
import time
import os

# Check default thread count
print(f"Default threads: {get_num_threads()}")
print(f"CPU cores: {os.cpu_count()}")

@jit(nopython=True, parallel=True)
def parallel_work(arr):
    result = np.zeros_like(arr)
    for i in prange(arr.shape[0]):
        result[i] = np.sqrt(arr[i]**2)
    return result

arr = np.random.randn(50_000_000)

# Test with different thread counts
for nthreads in [1, 2, 4, 8]:
    set_num_threads(nthreads)

    start = time.time()
    result = parallel_work(arr)
    elapsed = time.time() - start

    print(f"Threads: {nthreads}, Time: {elapsed:.3f}s")

# Environment variable alternative
# export NUMBA_NUM_THREADS=4
```

**Thread control methods:**

- `set_num_threads(n)`: Set at runtime
- `NUMBA_NUM_THREADS` environment variable
- Default: all available CPU cores
- Optimal: usually equals physical CPU cores

### Parallel NumPy Operations

Numba automatically parallelizes many NumPy operations when `parallel=True`.

```python
from numba import jit
import numpy as np
import time

@jit(nopython=True, parallel=True)
def parallel_numpy_ops(a, b, c):
    """NumPy operations automatically parallelized"""
    # These operations are automatically parallel
    result1 = a + b + c
    result2 = np.sqrt(result1)
    result3 = np.sin(result2) + np.cos(result2)
    return result3

n = 50_000_000
a = np.random.randn(n)
b = np.random.randn(n)
c = np.random.randn(n)

# Warmup
_ = parallel_numpy_ops(a, b, c)

start = time.time()
result = parallel_numpy_ops(a, b, c)
elapsed = time.time() - start

print(f"Time with automatic parallelization: {elapsed:.3f}s")
```

**Automatically parallelized NumPy operations:**

- Element-wise arithmetic (`+`, `-`, `*`, `/`)
- Math functions (`np.sin`, `np.cos`, `np.sqrt`, etc.)
- Array creation (`np.zeros`, `np.ones`)
- Reductions with explicit axis
- Boolean operations and comparisons

### Performance Considerations

**Factors affecting speedup:**

- Number of CPU cores
- Operation complexity (more complex = better speedup)
- Array size (larger = better speedup)
- Memory bandwidth (can become bottleneck)
- Thread synchronization overhead

### Key Takeaways

**Advantages of Numba's Multi-Core Parallelization:**

- Simple: Just add `parallel=True` or use `prange`
- Automatic: Numba identifies parallelizable loops
- Efficient: Releases GIL, uses native threads
- Portable: Same code works on different CPU counts
- No manual thread management required

**Best Practices:**

- Use `parallel=True` for large datasets
- Use `prange` when you need explicit control
- Ensure loop iterations are independent
- Profile to verify speedup (overhead for small arrays)
- Consider memory bandwidth limitations
- Test with different thread counts to find optimum


## CPU and GPU Code Adaptation

Numba enables Python developers to write GPU code without learning CUDA C/C++. While CPU and GPU versions require different code (due to different execution models), the core algorithm logic remains in Python, making the adaptation straightforward.

### Adapting Algorithms Between CPU and GPU

**What changes:**
- CPU uses loop-based execution (`@jit`)
- GPU uses thread-based execution (`@cuda.jit` with thread indexing)

**What stays the same:**
- Python syntax
- Core mathematical operations
- Algorithm logic

**CPU Version with @jit:**
```python
from numba import jit
import numpy as np

@jit(nopython=True)
def matrix_multiply_cpu(A, B, C):
    # Loop-based: iterate sequentially or in parallel
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
```

**GPU Version with @cuda.jit:**
```python
from numba import cuda

@cuda.jit
def matrix_multiply_gpu(A, B, C):
    # Thread-based: each thread handles one output element
    i, j = cuda.grid(2)  # Get thread position
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]  # Same computation!
        C[i, j] = tmp
```

**Key difference:** Loop iteration (`for i in range(...)`) becomes thread indexing (`i, j = cuda.grid(2)`), but the mathematical operations `A[i, k] * B[k, j]` remain identical.

### True Portability with @vectorize

For **genuine write-once, run-anywhere code**, use `@vectorize` where you can switch hardware targets by simply changing one parameter:

```python
from numba import vectorize
import numpy as np

# Define the function once
@vectorize(['float64(float64, float64)'], target='cpu')  # Single-threaded CPU
# @vectorize(['float64(float64, float64)'], target='parallel')  # Multi-core CPU
# @vectorize(['float64(float64, float64)'], target='cuda')  # GPU
def compute(a, b):
    return a * a + b * b  # Same code for all targets!

# Usage is identical regardless of target
data = np.random.randn(1_000_000)
result = compute(data, data)
```

**This is truly portable** - the function body never changes, only the `target` parameter determines where it runs.

### Benefits of This Approach

**Compared to writing CUDA C/C++:**

- Stay in Python ecosystem - no separate CUDA C compilation
- Use familiar NumPy array operations
- Easier debugging with Python tools
- Faster prototyping and iteration

**Development workflow:**

1. **Develop on CPU** with `@jit` for rapid prototyping and debugging
2. **Test and optimize** with small datasets using Python tools
3. **Adapt to GPU** by converting to `@cuda.jit` (requires thread indexing changes)
4. **Or use `@vectorize`** for element-wise operations (truly portable, just change `target`)
5. **Benchmark and profile** on target hardware

**Key advantages:**

- **Lower barrier to entry**: Python instead of CUDA C/C++
- **Systematic translation**: CPU loops → GPU thread indexing follows predictable patterns
- **True portability with `@vectorize`**: Same function code across CPU/GPU targets
- **Incremental adoption**: Start with CPU, move to GPU when needed




## NVIDIA GPU Support with Numba CUDA

Numba's CUDA support is one of its most powerful features, allowing Python developers to harness GPU computing power without learning CUDA C/C++. This democratizes GPU programming and significantly reduces the barrier to entry for parallel computing.

### Key Features

**Direct CUDA Kernel Programming:**

- Use `@cuda.jit` decorator to write CUDA kernels in Python
- Compiles to actual GPU code
- Access to CUDA-specific features:
  - Thread indexing
  - Shared memory
  - Synchronization primitives

**Memory Management:**

- `cuda.to_device()`: Transfer host data to GPU
- `cuda.device_array()`: Allocate uninitialized GPU memory
- `.copy_to_host()`: Transfer GPU data back to host

**Automatic Parallelization:**

- `@vectorize` and `@guvectorize` decorators
- Automatically parallelize array operations across GPU threads
- No explicit kernel code required

**Multi-GPU Support:**

- Compatible with multi-GPU systems
- Select specific devices
- Coordinate computation across multiple GPUs

### Performance Characteristics

GPU acceleration with Numba is most effective when:

- Operations are data-parallel (same operation on many data elements)
- Datasets are large enough to amortize transfer costs (typically >10^6 elements)
- Computation is arithmetic-intensive relative to memory access
- Algorithm can leverage GPU shared memory and coalesced memory access

Typical speedups range from 10-100x for well-suited problems, though results vary based on the algorithm and GPU hardware.

### Hardware Requirements

- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- CUDA Toolkit installed (drivers and runtime libraries)
- Compatible with most modern NVIDIA GPUs (GeForce, Quadro, Tesla, A100, H100)


## Examples

### Vector Addition (GPU Kernel Basics)

Vector addition demonstrates the fundamental concepts of GPU kernel programming with Numba.

**Key Concepts:**

- Defining CUDA kernels with `@cuda.jit`
- GPU memory management
- Thread indexing with `cuda.grid()`
- Kernel launch configuration (blocks and threads)

**Implementation:**


```python
from numba import cuda
import numpy as np
import math

@cuda.jit
def vector_add_kernel(a, b, c):
    """
    GPU kernel for element-wise vector addition
    Each thread computes one element of the output
    """
    # Calculate global thread index
    idx = cuda.grid(1)

    # Boundary check
    if idx < c.size:
        c[idx] = a[idx] + b[idx]

def vector_add_gpu(a, b):
    """
    Wrapper function to launch GPU kernel
    """
    # Allocate device memory
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(a)

    # Configure kernel launch
    threads_per_block = 256
    blocks_per_grid = math.ceil(a.size / threads_per_block)

    # Launch kernel
    vector_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

    # Copy result back to host
    c = d_c.copy_to_host()
    return c

# Usage
n = 1_000_000
a = np.random.randn(n)
b = np.random.randn(n)
c = vector_add_gpu(a, b)
```

**Key Points:**

- `cuda.grid(1)`: Computes the global thread index in a 1D grid
- `cuda.to_device()`: Transfers host memory to GPU
- `cuda.device_array_like()`: Allocates GPU memory with same shape/dtype
- `[blocks, threads]`: Kernel launch configuration
- Boundary checking prevents out-of-bounds access

**Performance Considerations:**

- Thread block size (256) balances occupancy and resource usage
- Grid size ensures enough threads to cover all array elements
- Memory transfers are often the bottleneck for simple operations
- Best for large arrays where computation dominates transfer time

See `examples/vecadd-numba.py` for complete implementation.


### Monte Carlo Pi Estimation (CPU/GPU Portability)


**Problem:**

Estimate π by randomly sampling points in a unit square and counting how many fall inside a quarter circle.

**Algorithm:**

1. Generate random points $(x, y)$ in $[0, 1] × [0, 1]$
2. Check if $x^2 + y^2 \leq 1$ (inside quarter circle)
3. $\pi \approx 4$ × (points inside circle) / (total points)

#### CPU Version with @jit

```python
from numba import jit
import numpy as np
import time

@jit(nopython=True)
def monte_carlo_pi_cpu(n_samples):
    """
    Estimate pi using Monte Carlo method on CPU
    """
    count_inside = 0

    for i in range(n_samples):
        x = np.random.random()
        y = np.random.random()

        # Check if point is inside quarter circle
        if x*x + y*y <= 1.0:
            count_inside += 1

    return 4.0 * count_inside / n_samples

# Usage
n = 10_000_000
start = time.time()
pi_estimate = monte_carlo_pi_cpu(n)
cpu_time = time.time() - start

print(f"CPU Estimate: π ≈ {pi_estimate:.6f}")
print(f"CPU Time: {cpu_time:.3f} seconds")
```

#### GPU Version with @cuda.jit

```python
from numba import cuda
import numpy as np
import math
import time

@cuda.jit
def monte_carlo_pi_kernel(n_samples, counts):
    """
    GPU kernel for Monte Carlo pi estimation
    Each thread generates samples and counts hits
    """
    idx = cuda.grid(1)

    # Create thread-local RNG state
    rng_states = cuda.random.create_xoroshiro128p_states(
        cuda.gridsize(1), seed=idx
    )

    # Each thread processes multiple samples
    samples_per_thread = (n_samples + cuda.gridsize(1) - 1) // cuda.gridsize(1)
    thread_count = 0

    for i in range(samples_per_thread):
        # Generate random point
        x = cuda.random.xoroshiro128p_uniform_float32(rng_states, idx)
        y = cuda.random.xoroshiro128p_uniform_float32(rng_states, idx)

        # Check if inside circle
        if x*x + y*y <= 1.0:
            thread_count += 1

    # Store thread result
    counts[idx] = thread_count

def monte_carlo_pi_gpu(n_samples):
    """
    Wrapper for GPU Monte Carlo pi estimation
    """
    # Launch configuration
    threads_per_block = 256
    blocks_per_grid = 512
    total_threads = threads_per_block * blocks_per_grid

    # Allocate device memory for counts
    d_counts = cuda.device_array(total_threads, dtype=np.int32)

    # Launch kernel
    monte_carlo_pi_kernel[blocks_per_grid, threads_per_block](
        n_samples, d_counts
    )

    # Copy results and sum
    counts = d_counts.copy_to_host()
    total_inside = counts.sum()

    return 4.0 * total_inside / n_samples

# Usage
n = 100_000_000  # More samples for GPU
start = time.time()
pi_estimate = monte_carlo_pi_gpu(n)
gpu_time = time.time() - start

print(f"GPU Estimate: π ≈ {pi_estimate:.6f}")
print(f"GPU Time: {gpu_time:.3f} seconds")
```

#### Simplified GPU Version with @vectorize

For simpler Monte Carlo simulations, `@vectorize` provides an easier approach:

```python
from numba import vectorize, cuda
import numpy as np

@vectorize(['int32(float32, float32)'], target='cuda')
def check_inside_circle(x, y):
    """
    Check if point is inside unit circle
    Automatically parallelized across GPU threads
    """
    return 1 if (x*x + y*y <= 1.0) else 0

def monte_carlo_pi_vectorized(n_samples):
    """
    Vectorized GPU implementation
    """
    # Generate random points on CPU (or use CuPy for GPU generation)
    x = np.random.random(n_samples).astype(np.float32)
    y = np.random.random(n_samples).astype(np.float32)

    # GPU automatically processes all points in parallel
    inside = check_inside_circle(x, y)

    return 4.0 * inside.sum() / n_samples

# Usage
n = 50_000_000
pi_estimate = monte_carlo_pi_vectorized(n)
print(f"Vectorized GPU Estimate: π ≈ {pi_estimate:.6f}")
```

**Observations**

1. **Custom Algorithm Implementation:**
   - Not available as a library function
   - Requires explicit loops and logic
   - Perfect for Numba's JIT compilation

2. **CPU/GPU Portability:**
   - Same basic algorithm structure for both targets
   - Easy to develop on CPU, deploy on GPU
   - Can run on systems without GPU

3. **True Parallel Computing:**
   - Each sample is independent
   - Naturally data-parallel
   - Scales well to many GPU threads


**Expected Performance Characteristics:**

- **Pure Python**: Very slow for large sample counts (baseline)
- **CPU @jit**: Typically 50-100x faster than pure Python
- **GPU @cuda.jit**: Can handle 100M+ samples efficiently, often 1000x+ faster than pure Python
- **GPU @vectorize**: Simpler to implement, performance between CPU and explicit GPU kernels

**Key Takeaways:**

- Same algorithm logic across CPU and GPU
- GPU shines for embarrassingly parallel problems
- `@vectorize` provides easier GPU programming for simple patterns
- `@cuda.jit` gives more control for complex kernels


## Installation

**Basic Installation:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install numba
```

**For GPU Support:**

```bash
# Ensure CUDA Toolkit is installed first
pip install numba

# Verify CUDA support
python -c "from numba import cuda; print(cuda.is_available())"
```
