# CUDA C

This document provides a foundation for understanding CUDA (Compute Unified Device Architecture) and writing simple CUDA C kernels. These concepts are essential for using CuPy's RawKernel feature and understanding GPU-accelerated Python frameworks.


## Overview

CUDA is NVIDIA's parallel computing platform and programming model for GPU computing. It allows developers to harness the massive parallelism of NVIDIA GPUs for general-purpose computing, not just graphics.

**Key Concepts:**

- **Parallel Computing Platform**: Enables GPU acceleration for scientific computing, data analysis, machine learning, and more
- **Extension of C/C++**: CUDA C/C++ adds GPU programming keywords to standard C/C++
- **Heterogeneous Computing**: Programs execute on both CPU (host) and GPU (device)
- **Massive Parallelism**: Modern GPUs have thousands of cores executing threads concurrently


## Brief History

CUDA was introduced by NVIDIA in 2006 as a way to make GPU computing accessible beyond graphics programming.

**Key Milestones:**

- **2006**: CUDA 1.0 released, making GPU computing accessible to C programmers
- **2008**: CUDA 2.0 added shared memory and synchronization primitives
- **2010**: Fermi architecture introduced unified address space
- **2012**: Kepler architecture brought dynamic parallelism
- **2014**: Maxwell architecture improved power efficiency
- **2016**: Pascal architecture added unified memory improvements
- **2017**: Volta architecture introduced Tensor Cores for AI workloads
- **2018**: Turing architecture enhanced ray tracing capabilities
- **2020**: Ampere architecture (A100) for data centers
- **2022**: Hopper architecture (H100) for extreme-scale AI
- **Present**: CUDA is the dominant GPU computing platform for scientific computing and AI

CUDA has become the de facto standard for GPU computing, with extensive library support (cuBLAS, cuFFT, cuDNN) and a large developer community.


## CUDA Programming Model

CUDA uses a heterogeneous computing model where code runs on both CPU and GPU.

### Host and Device

**Host (CPU):**

- Manages overall program flow
- Allocates GPU memory
- Copies data between CPU and GPU
- Launches kernels (GPU functions)
- Retrieves results from GPU

**Device (GPU):**

- Executes massively parallel computations
- Operates on data in GPU memory
- Thousands of threads run concurrently
- Optimized for throughput, not latency

**Typical Workflow:**

1. Allocate memory on GPU
2. Copy input data from CPU to GPU
3. Launch kernel to process data on GPU
4. Copy results from GPU back to CPU
5. Free GPU memory

### Kernels

**What are Kernels:**

- Functions that execute on the GPU
- Launched by CPU, executed by thousands of GPU threads
- Each thread runs the same kernel code but operates on different data
- Written in CUDA C/C++ with special syntax

**Kernel Launch:**

```cuda
// Kernel definition
__global__ void my_kernel(float* data, int n) {
    // Kernel code here
}

// Kernel launch from host
my_kernel<<<blocks, threads>>>(data, n);
```

**Key Points:**

- `__global__` keyword marks a function as a kernel
- Triple angle brackets `<<<...>>>` specify execution configuration
- First parameter: number of thread blocks
- Second parameter: number of threads per block


## GPU Architecture Essentials

Understanding GPU hardware organization helps you write efficient CUDA code.

### Streaming Multiprocessors (SMs)

**Architecture Overview:**

- GPU consists of multiple Streaming Multiprocessors (SMs)
- Each SM contains many CUDA cores (Streaming Processors)
- Modern GPUs have 10-100+ SMs, each with 64-128 cores
- Example: A100 GPU has 108 SMs with 64 FP32 cores each = 6,912 CUDA cores

**Execution Model:**

- Each SM executes one or more thread blocks
- Threads within a block can cooperate via shared memory
- Threads in different blocks cannot directly communicate
- SM schedules threads in groups of 32 called warps

### Thread Hierarchy: Grids, Blocks, and Threads

CUDA organizes threads in a hierarchical structure to map computation onto GPU hardware.

**Thread:**

- Smallest unit of execution
- Executes kernel code on one data element
- Has unique thread ID within its block
- Runs on a CUDA core

**Warp:**

- Group of **32 threads** that execute together in lockstep
- The actual scheduling unit on an SM
- All threads in a warp execute the same instruction simultaneously (SIMT)
- If threads diverge (different branches), execution is serialized until reconvergence
- Performance consideration: Keep threads in a warp following the same path

**Block (Thread Block):**

- Group of **threads** (typically 128-1024 threads)
- Divided into warps of 32 threads each
- Threads in a block execute on the same SM
- Can cooperate via shared memory
- Can synchronize with barriers
- Has unique block ID within the grid

**Grid:**

- Collection of **blocks** executing the same kernel
- Can contain thousands of blocks
- Blocks execute independently (any order)
- Grid spans entire problem domain

**Visual Hierarchy:**

```
Grid (entire problem)
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   ├── ...
│   └── Thread N-1
├── Block 1
│   ├── Thread 0
│   ├── Thread 1
│   ├── ...
│   └── Thread N-1
└── Block M-1
    ├── Thread 0
    ├── Thread 1
    ├── ...
    └── Thread N-1
```


## Thread Coordinates and Indexing

Every CUDA thread needs to know its unique position to determine which data element to process.

### Built-in Variables

CUDA provides built-in variables accessible within kernel code:

**Thread Dimensions and Indices:**

| Variable | Type | Description |
|----------|------|-------------|
| `threadIdx.x` | uint3 | Thread index within its block (0 to blockDim.x-1) |
| `blockIdx.x` | uint3 | Block index within the grid (0 to gridDim.x-1) |
| `blockDim.x` | dim3 | Number of threads per block (set at launch) |
| `gridDim.x` | dim3 | Number of blocks in the grid (set at launch) |

**Note:** Variables have `.x`, `.y`, `.z` components for 1D, 2D, or 3D indexing

### Calculating Global Thread Index

The most common task is calculating a unique global index for each thread to access array elements.

**1D Grid and 1D Blocks (Most Common):**

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

**Breakdown:**

- `blockIdx.x * blockDim.x`: Starting position of current block in global array
- `+ threadIdx.x`: Offset within the block
- Result: Unique index for each thread across entire grid

**Example:**

```
Grid: 3 blocks, each with 4 threads

Block 0:                Block 1:                Block 2:
Thread 0 → idx = 0     Thread 0 → idx = 4     Thread 0 → idx = 8
Thread 1 → idx = 1     Thread 1 → idx = 5     Thread 1 → idx = 9
Thread 2 → idx = 2     Thread 2 → idx = 6     Thread 2 → idx = 10
Thread 3 → idx = 3     Thread 3 → idx = 7     Thread 3 → idx = 11
```

**Formula Visualization:**

```
Block 0: (0 * 4) + 0,1,2,3  →  0,1,2,3
Block 1: (1 * 4) + 0,1,2,3  →  4,5,6,7
Block 2: (2 * 4) + 0,1,2,3  →  8,9,10,11
```

### 2D Indexing

For 2D problems (images, matrices):

```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * width + x;  // Convert 2D to 1D for array access
```

**Use case:** Image processing, matrix operations


## Writing Simple CUDA C Kernels


**Function Qualifiers:**

- `__global__`: Marks function as a kernel (callable from host, runs on device)
- `__device__`: Function runs on device, callable only from device
- `__host__`: Function runs on host (normal C function)


**`extern "C"`:**

- Prevents C++ name mangling
- Allows Python to find the kernel by name
- Required when using CUDA C++ code with Python

Note: C++ name mangling is a technique where the C++ compiler encodes additional information into function names to support features like function overloading and namespaces.

**Pointer Types:**

- `const float* input`: Read-only input array
- `float* output`: Output array (writable)
- Pointers reference GPU global memory

### Example: Vector Addition Kernel

**Complete CUDA C Kernel:**

```cuda
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int n) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: ensure thread doesn't access beyond array
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

Notes:

1. `extern "C" __global__`: Kernel declaration for Python interop
2. `void vector_add(...)`: Kernel name and parameters
   - `const float* a, b`: Input arrays (read-only)
   - `float* c`: Output array (writable)
   - `int n`: Array size
3. `int idx = ...`: Calculate this thread's unique global index
4. `if (idx < n)`: Boundary check (explained below)
5. `c[idx] = a[idx] + b[idx]`: Actual computation

### Boundary Checking

- Grid size may not perfectly divide array size
- You might launch 256 threads but only have 250 elements
- Extra threads must not access invalid memory
- Accessing out-of-bounds causes crashes or corruption

**Example Scenario:**

- Array size: 1000 elements
- Block size: 256 threads
- Blocks needed: ceil(1000/256) = 4 blocks
- Total threads launched: 4 * 256 = 1024 threads
    - Threads 0-999: Process valid data
    - Threads 1000-1023: Must be filtered out with boundary check


## Memory Hierarchy

Understanding GPU memory is important for performance optimization.

**Memory Types:**

| Memory Type | Location | Access Speed | Scope | Lifetime |
|-------------|----------|--------------|-------|----------|
| **Global Memory** | GPU DRAM | Slow (~400 cycles) | All threads | Program duration |
| **Shared Memory** | On-chip (SM) | Fast (~1 cycle) | Block threads | Block lifetime |
| **Registers** | On-chip (SM) | Fastest | Single thread | Thread lifetime |
| **Local Memory** | GPU DRAM | Slow | Single thread | Thread lifetime |
| **Constant Memory** | GPU DRAM | Slow (cached) | All threads (read-only) | Program duration |

**For Simple Kernels:**

- Use global memory for input/output arrays
- Let compiler manage registers for local variables
- Shared memory for advanced optimization (not covered here)

**Accessing Global Memory:**

```cuda
__global__ void kernel(float* global_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Read from global memory
        float value = global_array[idx];

        // Compute (uses registers)
        float result = value * 2.0f;

        // Write to global memory
        global_array[idx] = result;
    }
}
```

**Accessing Shared Memory:**

```cuda
__global__ void kernel_with_shared(float* global_array, int n) {
    // Declare shared memory (shared by all threads in the block)
    __shared__ float shared_data[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;

    if (idx < n) {
        // Read from global memory into shared memory
        shared_data[local_idx] = global_array[idx];
        
        // Synchronize to ensure all threads have loaded data
        __syncthreads();
        
        // Now all threads can access shared memory (fast)
        float value = shared_data[local_idx];
        float result = value * 2.0f;
        
        // Synchronize before writing back
        __syncthreads();
        
        // Write result to global memory
        global_array[idx] = result;
    }
}
```

**Note: Global vs Local Memory**

Despite the name, **local memory is NOT faster than global memory** - both reside in slow GPU DRAM (~400 cycles). The "local" refers to visibility (single thread only), not speed.

Local memory is automatically used when:
- Too many variables to fit in registers (register spilling)
- Large thread-local arrays
  
Example:
```cuda
__global__ void kernel() {
    float temp[100];  // Too large for registers → spills to local memory (slow!)
}
```

**Note:** Minimize local memory by keeping thread-local variables small so they stay in registers (fastest).

**Constant Memory**

Constant memory is for **read-only data that all threads access**. Although stored in GPU DRAM, it's cached and broadcast-efficient.

Use for:
- Mathematical constants (π, gravity, etc.)
- Lookup tables and coefficients
- Configuration parameters

Example:
```cuda
__constant__ float PI = 3.14159f;
__constant__ float coefficients[256];

__global__ void kernel() {
    float result = data * PI;  // All threads read same value (cached & broadcast)
}
```

**Key benefit:** When all threads in a warp read the same location, one fetch is broadcast to all threads (much faster than global memory). Limited to 64KB per kernel.




## Summary: CUDA C

**Key Concepts to Remember:**

1. **Thread Indexing**: Always calculate global index with `blockIdx.x * blockDim.x + threadIdx.x`

2. **Boundary Checking**: Always check `if (idx < n)` before array access

3. **Kernel Syntax**: Use `__global__` to mark GPU kernel functions

4. **Launch Configuration**: Determine blocks and threads:
   ```cuda
   threads_per_block = 256
   blocks = (array_size + threads_per_block - 1) / threads_per_block
   kernel<<<blocks, threads_per_block>>>(args);
   ```

5. **Memory**: Pointers in kernels reference GPU global memory


**Resources for CUDA **

- [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Programming Massively Parallel Processors: A Hands-on Approach](https://a.co/d/aaI9Wtb)
