# GPU Computing Ecosystem in Python

This document provides a quick overview of Python frameworks for GPU computing. Use this guide to understand your options and choose the right tool for your needs. Detailed tutorials are in subsequent documents.


## Overview

Python offers multiple frameworks for GPU acceleration, each designed for different use cases and programming styles.


**Key Principle:**

- Start high-level (CuPy, JAX) for productivity
- Drop to mid-level (Numba) for custom algorithms
- Use low-level (CUDA Python) only when necessary
- Write CUDA C directly

Most scientific computing stays at high/mid level


## Framework Summaries

### CuPy: NumPy Drop-in Replacement

- GPU-accelerated array library with NumPy/SciPy-compatible API
- Often just requires changing `import numpy as np` to `import cupy as cp`
- Leverages optimized CUDA libraries (cuBLAS, cuFFT, cuDNN)

**Syntax Style:**

```python
import cupy as cp
c = a @ b  # Matrix multiply on GPU, just like NumPy
```

**Best for:**

- Existing NumPy/SciPy code that needs GPU acceleration
- Standard operations: linear algebra, FFT, statistics
- Quick prototyping with minimal code changes
- When you think in terms of arrays and operations

**Not ideal for:**

- Custom algorithms not expressible as array operations
- Code requiring explicit control over GPU threads

**Installation:**

```bash
pip install cupy-cuda12x  # For CUDA 12.x
```


### Numba: JIT Compiler for Custom Kernels

- Just-in-Time compiler for Python and NumPy code
- Compiles Python functions to native machine code (CPU or GPU)
- Write GPU kernels in Python syntax, not CUDA C

**Syntax Style:**

```python
from numba import cuda

@cuda.jit
def my_kernel(data):
    idx = cuda.grid(1)
    data[idx] *= 2  # Each GPU thread processes one element
```

**Best for:**

- Custom algorithms with explicit loops
- Code that doesn't fit NumPy's array operations model
- When you need control over GPU thread organization
- CPU/GPU portable code (same function runs on both)

**Not ideal for:**

- Simple NumPy operations (use CuPy instead)
- When pre-optimized libraries exist

Example:

```python
# CuPy: One line, uses optimized cuBLAS library
import cupy as cp
result = cp.matmul(a, b)  # Or just: a @ b

# Numba: Would require writing a full matrix multiply kernel
# (dozens of lines, harder to optimize than vendor libraries)
@cuda.jit
def matmul_kernel(A, B, C):
    # Complex tiling logic needed for performance
    # Must handle shared memory manually
    # Easy to write slow code
    ...  # 50+ lines of kernel code
```

**Installation:**

```bash
pip install numba
```


### JAX: Functional Programming with Auto-Differentiation

- Functional array computing library with NumPy-like API
- Automatic differentiation (gradients) built-in
- JIT compilation via XLA compiler
- Composable function transformations

**Syntax Style:**

```python
import jax.numpy as jnp
from jax import jit, grad

@jit  # Compile for performance
def loss(params):
    return jnp.sum(params ** 2)

gradient = grad(loss)  # Automatic differentiation
```

**Best for:**

- Machine learning research and optimization
- Problems requiring gradients (inverse problems, parameter estimation)
- Functional programming style
- Multi-GPU/TPU computing with `pmap`
- Same code runs on CPU, GPU, or TPU (hardware-agnostic)

Example:

```python
import jax.numpy as jnp
from jax import jit

@jit
def compute(x):
    return jnp.sum(x ** 2)

# Same code, different backends:
# jax.default_device('cpu')  → runs on CPU
# jax.default_device('gpu')  → runs on GPU  
# jax.default_device('tpu')  → runs on TPU
```

**Not ideal for:**

- Simple array operations without gradients (use CuPy)
- Imperative programming style
- Code with side effects or global state

**Installation:**

```bash
pip install jax[cuda12]  # For CUDA 12.x
```


### CUDA Python: Low-Level GPU Control

- Official Python bindings to CUDA runtime and driver APIs
- Direct access to all CUDA features
- Thin wrapper around CUDA C APIs

**Syntax Style:**

```python
from cuda import cuda
err, device = cuda.cuDeviceGet(0)
err, context = cuda.cuCtxCreate(0, device)
# Manual memory allocation, kernel loading, etc.
```

**Best for:**

- Integrating existing CUDA C/C++ code
- Building higher-level frameworks
- Performance optimization requiring low-level control
- Custom memory management strategies

**Not ideal for:**

- Application development (too low-level)
- Most scientific computing tasks

**Installation:**

```bash
pip install cuda-python
```


## Comparison

| Framework | Abstraction | Ease of Use | Flexibility | Auto-Diff | CPU/GPU Support | Best For |
|-----------|-------------|-------------|-------------|-----------|-----------------|----------|
| **CuPy** | High | Easiest | Moderate | No | GPU only | NumPy code acceleration |
| **JAX** | High | Easy | High | Yes | CPU/GPU/TPU | Numerical computing with gradients, ML, optimization |
| **Numba** | Mid | Moderate | High | No | CPU/GPU | Custom algorithms, loops |
| **CUDA Python** | Low | Hard | Maximum | No | GPU only | Framework building, CUDA integration |

**Performance:**

- All can achieve excellent performance when used correctly
- CuPy: Leverages highly optimized vendor libraries
- JAX: XLA compiler produces efficient code
- Numba: Performance matches hand-written CUDA C
- CUDA Python: Maximum control, performance depends on your code



## Interoperability

These frameworks work together through standard protocols:

**CUDA Array Interface:**

- Zero-copy data exchange between CuPy and Numba
- Arrays share GPU memory, no copying needed

**DLPack:**

- Zero-copy exchange between CuPy, JAX, PyTorch, TensorFlow
- Industry standard for GPU array interchange

**Common Pattern:**

```python
# Use CuPy for high-level operations
import cupy as cp
data = cp.random.randn(1000000)
result = cp.fft.fft(data)  # CuPy's optimized FFT

# Use Numba for custom kernel on same data
from numba import cuda
@cuda.jit
def custom_process(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] *= 2

threads = 256
blocks = (data.size + threads - 1) // threads
custom_process[blocks, threads](result)  # Works directly with CuPy array!
```

**Details:** Each framework document (30/40/50/60) covers interoperability

