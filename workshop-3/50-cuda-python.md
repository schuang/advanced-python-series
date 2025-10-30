# CUDA Python

CUDA Python is NVIDIA's official Python interface to CUDA, providing low-level access to the CUDA runtime and driver APIs directly from Python. It enables developers to leverage the full power of CUDA while staying within the Python ecosystem.


## Overview

CUDA Python provides Pythonic bindings to CUDA's C APIs, allowing developers to access low-level GPU programming features without writing C/C++ code. Unlike higher-level libraries that abstract away CUDA details, CUDA Python exposes the full CUDA programming model, giving you fine-grained control over GPU resources.

Key advantages:

- Direct access to CUDA runtime and driver APIs from Python
- Full control over GPU resources (streams, events, memory, contexts)
- Thin Python wrapper around CUDA C APIs - minimal overhead
- Seamless interoperability with existing CUDA ecosystem
- Official NVIDIA support and maintenance


## Brief History

CUDA Python represents NVIDIA's evolution toward better Python support for GPU computing. While CUDA has always been accessible from Python through various third-party tools, NVIDIA developed official Python bindings to provide a standardized, well-supported interface.

Key milestones:

- **2020**: NVIDIA announced plans for official Python bindings to CUDA
- **2021**: Initial release of cuda-python package with core runtime APIs
- **2022**: Expanded coverage of CUDA APIs, added Driver API support
- **2023**: Introduction of `cuda.core` - a higher-level Pythonic interface
- **2024-Present**: Continuous expansion of API coverage and improved Python ergonomics

CUDA Python is maintained by NVIDIA and represents the company's commitment to making CUDA accessible to the broader Python scientific computing community while maintaining low-level control.


## What CUDA Python Does

**Core Functionality:**

- Official Python bindings to CUDA C APIs
- Low-level access to GPU programming features
- Thin wrapper maintaining CUDA's performance characteristics
- Bridges gap between high-level Python and low-level CUDA

**API Coverage:**

**Runtime API Bindings:**

- Memory management (malloc, memcpy, memset)
- Stream and event management
- Device management and queries
- Kernel launch configuration
- Texture and surface memory

**Driver API Bindings:**

- Module and function management
- Context management
- Advanced memory operations
- Unified memory control
- Peer-to-peer device access

**cuda.core Module:**

- Higher-level Pythonic interface built on top of runtime/driver APIs
- Object-oriented abstractions for CUDA concepts
- Context managers for resource management
- More intuitive API while maintaining low-level control

**Memory Management:**

- Direct control over device memory allocation
- Support for pinned (page-locked) host memory
- Managed (unified) memory support
- Zero-copy memory for integrated GPUs
- Memory pools and asynchronous allocation

**Stream and Synchronization:**

- Create and manage CUDA streams for concurrent execution
- Event-based synchronization and timing
- Stream priorities and callbacks
- Graph execution support

**Interoperability:**

- Works with arrays from CuPy, PyTorch, TensorFlow via pointer interfaces
- Compatible with Numba CUDA kernels
- Integrates with C/C++ CUDA code
- Supports external memory and semaphore sharing


## Key Use Cases in Scientific Computing

### 1. **Custom GPU Resource Management**
When you need precise control over GPU memory, streams, and execution flow beyond what high-level libraries provide. Ideal for implementing custom memory pools, scheduling strategies, or multi-GPU coordination.

Example: Building custom task schedulers, implementing memory-efficient algorithms, multi-GPU workload distribution

### 2. **Integrating Existing CUDA C/C++ Code**
Loading and executing pre-compiled CUDA kernels (PTX or cubin) from Python without rewriting in Numba or CuPy. Useful for leveraging existing CUDA codebases or using highly optimized vendor-provided kernels.

Example: Wrapping proprietary CUDA libraries, using vendor-optimized kernels, migrating legacy CUDA applications

### 3. **Advanced Performance Optimization**
Fine-tuning performance through explicit stream management, memory transfers, and kernel launch configurations. When library defaults aren't sufficient and you need manual optimization.

Example: Overlapping computation and memory transfers, multi-stream execution, custom profiling and timing

### 4. **Low-Level Algorithm Implementation**
Implementing algorithms that require direct manipulation of GPU features not exposed by higher-level libraries, such as texture memory, surface memory, or complex synchronization patterns.

Example: Custom ray tracing kernels, advanced image processing with texture memory, lock-free data structures

### 5. **Framework Development**
Building higher-level GPU computing frameworks or libraries that need to manage GPU resources programmatically. CUDA Python provides the foundation for creating custom abstractions.

Example: Developing domain-specific GPU libraries, building workflow engines, creating custom array libraries

### 6. **Multi-GPU and Distributed Computing**
Implementing sophisticated multi-GPU algorithms with explicit peer-to-peer transfers, unified virtual addressing, and cross-device synchronization that automated tools don't provide.

Example: Multi-GPU molecular dynamics, distributed deep learning, multi-GPU linear solvers


## CUDA Python vs CuPy: Key Differences

Understanding the fundamental differences helps you choose the right tool for your needs.

### Abstraction Level

**CUDA Python:**

- Low-level access to CUDA APIs
- You manage memory, streams, kernels explicitly
- Mirrors CUDA C programming model
- Requires understanding of CUDA concepts

**CuPy:**

- High-level NumPy-compatible arrays
- Automatic memory and execution management
- Hides CUDA details behind NumPy interface
- Minimal CUDA knowledge required

### Programming Model

**CUDA Python:**
```python
# Explicit memory allocation and management
import cuda.core as cuda

device = cuda.Device()
mem_handle = device.allocate(nbytes)
# Manual memory copies, kernel launches, synchronization
```

**CuPy:**
```python
# NumPy-style array operations
import cupy as cp

a = cp.array([1, 2, 3])  # Memory managed automatically
b = a + 5  # Operations execute automatically on GPU
```

### Use Case Comparison

**CUDA Python - Choose When:**

- You need low-level control over GPU resources
- Integrating existing CUDA C/C++ code
- Implementing custom memory management strategies
- Building frameworks or libraries
- Performance tuning requires explicit stream/memory control
- You're a CUDA expert wanting Python bindings

**CuPy - Choose When:**

- You have NumPy code to accelerate
- You want array-level operations
- You prefer automatic resource management
- You want pre-optimized library functions
- Development speed matters more than fine-grained control
- You're a Python/NumPy expert learning GPU computing

### Complementary Usage

**They work well together:**

- Use CuPy for high-level array operations
- Use CUDA Python to access CuPy array pointers for custom operations
- Leverage CuPy's memory allocator with CUDA Python's stream management
- Combine CuPy's convenience with CUDA Python's control

**Example workflow:**
```python
import cupy as cp
import cuda.core as cuda

# CuPy creates and manages arrays
data = cp.random.randn(1000000)

# Access underlying pointer for CUDA Python operations
ptr = data.data.ptr
stream = cuda.Stream()

# Use CUDA Python for custom low-level operations
# while CuPy handles memory lifecycle
```

### Comparison Table

| Criterion | CUDA Python | CuPy |
|-----------|-------------|------|
| **Abstraction level** | Low (direct CUDA APIs) | High (NumPy arrays) |
| **Learning curve** | Steep (requires CUDA knowledge) | Gentle (NumPy knowledge) |
| **Control** | Complete control over GPU | Automatic management |
| **Code verbosity** | More verbose | Concise |
| **Best for** | Custom kernels, framework building | Array operations, algorithm implementation |
| **Memory management** | Manual (explicit allocation) | Automatic (array lifecycle) |
| **CUDA expertise** | Required | Not required |
| **Development speed** | Slower (more code) | Faster (less code) |
| **Use case** | Low-level optimization, integration | Accelerating NumPy workflows |


## Getting Started with CUDA Python

**Installation:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install cuda-python
```

**Basic Example - Device Query:**
```python
from cuda import cuda

# Initialize CUDA
err, = cuda.cuInit(0)
assert err == cuda.CUresult.CUDA_SUCCESS

# Get device count
err, device_count = cuda.cuDeviceGetCount()
print(f"Found {device_count} CUDA devices")

# Query device properties
err, device = cuda.cuDeviceGet(0)
err, name = cuda.cuDeviceGetName(128, device)
print(f"Device 0: {name.decode()}")
```

**Using cuda.core (Higher-level interface):**
```python
import cuda.core as cuda

# More Pythonic interface
device = cuda.Device()
print(f"Device: {device.name}")
print(f"Compute capability: {device.compute_capability}")
print(f"Total memory: {device.total_memory / 1e9:.2f} GB")

# Context managers for resource safety
with cuda.Stream() as stream:
    # Operations in this stream
    pass
```

**Hardware Requirements:**
- NVIDIA GPU with CUDA support (Compute Capability 3.0+)
- CUDA Toolkit installed (version 11.2 or later recommended)
- Compatible with all CUDA-capable NVIDIA GPUs


## Vector Addition Example

**Overview:**

- Demonstrates loading and executing a pre-compiled CUDA kernel
- Shows explicit memory management and kernel launch
- Illustrates the low-level nature of CUDA Python

**Key Concepts Covered:**

- Module loading (PTX or cubin)
- Device memory allocation
- Host-to-device and device-to-host memory transfers
- Explicit kernel launch with grid/block configuration
- Stream management and synchronization

**Key Differences from CuPy:**

- Requires pre-compiled kernel code (PTX/cubin) or runtime compilation
- Explicit memory allocation with `cuMemAlloc()`
- Manual memory copies with `cuMemcpyHtoD()` and `cuMemcpyDtoH()`
- Explicit kernel launch with `cuLaunchKernel()`
- More code, but complete control over execution

**Key Differences from Numba:**

- Uses pre-compiled kernels rather than JIT compilation
- More verbose setup and memory management
- Direct access to driver/runtime APIs
- Suitable for integrating existing CUDA code

See `examples/vecadd-cuda-python.py` for complete implementation.

**Note:** CUDA Python is typically used when you need the lowest-level access or are integrating existing CUDA C/C++ code. For most scientific computing tasks, CuPy or Numba provide better productivity while CUDA Python excels at framework development and custom optimization.
