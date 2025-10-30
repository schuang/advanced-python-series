# CPU vs. GPU Computing

This document provides essential background on parallel computing, covering both CPU vectorization (SIMD) and GPU computing. Understanding these concepts is crucial for effective Python GPU programming with CUDA, CuPy, Numba, CUDA Python, and JAX.


## Overview

Modern computing relies on parallelism to achieve high performance. Two dominant paradigms exist:

**CPU Parallelism (SIMD):**

- Single Instruction, Multiple Data
- Vector processing within CPU cores
- 4-16 way parallelism per core
- Latency-oriented design

**GPU Parallelism (SIMT):**

- Single Instruction, Multiple Threads
- Massive thread-level parallelism
- Thousands of threads executing concurrently
- Throughput-oriented design

**Why This Matters:**

- Python frameworks leverage both CPU and GPU parallelism
- NumPy uses CPU SIMD (AVX-512) for vectorized operations
- CuPy, Numba, JAX use GPU SIMT for massive parallelism
- Understanding both helps you choose the right tool
- Critical for scientific computing, data analysis, and machine learning


## CPU Vector Processing (SIMD)

CPU vector processing uses SIMD (Single Instruction, Multiple Data) to perform the same operation on multiple data elements simultaneously.

### What is SIMD

**Core Concept:**

- Single instruction operates on multiple data elements in parallel
- Uses special wide vector registers to hold multiple values
- One instruction can add/multiply 4-16 numbers simultaneously
- Available on all modern CPUs (Intel, AMD, ARM)

**Example:**

```
Traditional (Scalar):
for i in range(8):
    c[i] = a[i] + b[i]  # 8 separate additions

SIMD (Vector):
c[0:8] = a[0:8] + b[0:8]  # Single instruction, 8 additions in parallel
```

### CPU SIMD Instruction Sets

Modern CPUs have evolved through multiple generations of SIMD instructions:

**Intel/AMD Evolution:**

| Instruction Set | Year | Register Width | Float32 Parallel Ops |
|----------------|------|----------------|---------------------|
| MMX | 1997 | 64-bit | N/A (integer only) |
| SSE | 1999 | 128-bit (XMM) | 4 |
| SSE2 | 2001 | 128-bit (XMM) | 4 |
| AVX | 2011 | 256-bit (YMM) | 8 |
| AVX2 | 2013 | 256-bit (YMM) | 8 (improved) |
| AVX-512 | 2016 | 512-bit (ZMM) | 16 |

**Example SIMD Instructions:**

- `vmulps`: Vector multiply packed single-precision (8 multiplications at once with AVX)
- `vaddpd`: Vector add packed double-precision (4 additions at once with AVX)
- `vfmadd`: Vector fused multiply-add (8 operations at once with AVX)

### How CPUs Use SIMD

**Vector Registers:**

- Special wide registers hold multiple values
- AVX: 256-bit YMM registers (8 × float32 or 4 × float64)
- AVX-512: 512-bit ZMM registers (16 × float32 or 8 × float64)
- Data must be aligned in memory for best performance

**Programming SIMD:**

1. **Automatic (Compiler)**: Compiler auto-vectorizes loops
   ```c
   for (int i = 0; i < n; i++) {
       c[i] = a[i] + b[i];  // Compiler may use SIMD
   }
   ```

2. **Explicit (Intrinsics)**: Direct SIMD programming
   ```c
   __m256 a_vec = _mm256_load_ps(&a[i]);
   __m256 b_vec = _mm256_load_ps(&b[i]);
   __m256 c_vec = _mm256_add_ps(a_vec, b_vec);
   _mm256_store_ps(&c[i], c_vec);
   ```

3. **Library (NumPy)**: Libraries use optimized SIMD
   ```python
   import numpy as np
   c = a + b  # NumPy uses AVX/AVX-512 internally
   ```

### CPU SIMD Performance

**Typical Speedups:**

- SSE (4-way): ~3-4x over scalar code
- AVX (8-way): ~6-8x over scalar code
- AVX-512 (16-way): ~10-15x over scalar code
- Actual speedup depends on memory bandwidth and algorithm

**Example CPU Peak Performance:**

- Intel Xeon Platinum 8380 (40 cores, AVX-512):
  - Peak FP32: ~5 TFLOPS
  - Peak FP64: ~2.5 TFLOPS
- AMD EPYC 9654 (96 cores, AVX-512):
  - Peak FP32: ~11 TFLOPS
  - Peak FP64: ~5.5 TFLOPS

### CPU Design Philosophy: Latency-Oriented

**Optimization Goals:**

- Minimize latency for single-threaded execution
- Fast execution of individual instruction sequences
- Complex control logic for branch prediction
- Large caches to reduce memory latency

**Silicon Budget:**

- ~50-60% for cache memory (L1/L2/L3)
- ~20-30% for control logic (branch prediction, out-of-order execution)
- ~10-20% for ALUs (arithmetic logic units)
- Small fraction dedicated to vector units (SIMD)

**Strengths:**

- Excellent for sequential algorithms
- Good for complex control flow
- Low latency for individual operations
- Versatile general-purpose computing

**Limitations:**

- Limited parallelism (4-16 way SIMD per core)
- Even with 64 cores, peak throughput is modest
- SIMD complexity increases with wider vectors
- Diminishing returns beyond AVX-512


## GPU Computing: Massive Parallelism

GPUs evolved from graphics accelerators to general-purpose parallel processors capable of executing thousands of threads concurrently.

### Brief History

**Evolution of GPU Computing:**

- **1999**: NVIDIA GeForce 256 - first consumer GPU
- **2006**: NVIDIA CUDA - GPU computing platform launched
- **2007**: AMD Stream SDK - AMD's GPGPU initiative
- **2010**: NVIDIA Fermi - unified compute/graphics architecture
- **2017**: NVIDIA Volta - Tensor Cores for AI
- **2020**: NVIDIA Ampere A100 - dominant data center GPU
- **2020**: AMD MI100 CDNA - dedicated compute architecture
- **2022**: NVIDIA Hopper H100 - transformer engine for LLMs
- **2023**: NVIDIA Grace Hopper (CPU+GPU) — tightly integrated CPU and GPU architecture enabling unified memory and large-scale AI workloads
- **Present**: GPUs essential for AI, scientific computing, data analytics

**Industry Landscape:**

- NVIDIA: ~80-90% of GPU computing market (CUDA platform)
- AMD: Growing with ROCm open-source platform
- Intel: Entering with Data Center GPU Max series
- Apple: Unified memory architecture with Metal

### GPU Architecture Overview

**Key Characteristics:**

- **Massive Parallelism**: Thousands of cores executing concurrently
- **Throughput-Oriented**: Optimized for processing large amounts of data
- **SIMT Execution**: Single Instruction, Multiple Threads
- **Specialized Memory**: Multiple memory types for different access patterns
- **Heterogeneous**: Works with CPU for maximum performance

**Why GPUs Excel:**

- 10-1000x speedup over CPU for parallel workloads
- Essential for deep learning and AI
- Ideal for array operations and linear algebra
- High memory bandwidth (1-3 TB/s vs 50-100 GB/s for CPU)
- Cost-effective performance per watt

### GPU vs CPU: Architectural Comparison

**Design Philosophy:**

| Aspect | CPU (Latency-Oriented) | GPU (Throughput-Oriented) |
|--------|------------------------|---------------------------|
| **Optimization Goal** | Minimize latency for single thread | Maximize throughput for many threads |
| **Core Count** | 4-64 powerful cores | 2,000-10,000+ simple cores |
| **Core Complexity** | Complex (branch prediction, OoO) | Simple (in-order execution) |
| **Cache Size** | Large (megabytes L1/L2/L3) | Small (kilobytes per SM) |
| **Control Logic** | Extensive (~30% of silicon) | Minimal (~5% of silicon) |
| **ALU Density** | ~10-20% of silicon | ~70-80% of silicon |
| **Memory Strategy** | Large caches hide latency | Parallelism hides latency |

**Visual Comparison:**

```
CPU Die:                          GPU Die:
┌──────────────────┐             ┌──────────────────┐
│  Cache  │ Cache  │             │ Compute│Compute  │
│─────────┼────────│             │────────┼─────────│
│ Control │ Cache  │             │ Compute│Compute  │
│─────────┼────────│             │────────┼─────────│
│ ALU ALU │ ALU ALU│             │ Compute│Compute  │
└──────────────────┘             │────────┼─────────│
  Few powerful cores              │ Compute│Compute  │
                                  └──────────────────┘
                                   Many simple cores
```

**Performance Comparison:**

| Metric | Typical CPU | Typical GPU |
|--------|-------------|-------------|
| Cores | 16-64 | 2,000-10,000+ |
| Peak FP32 | 1-10 TFLOPS | 20-80 TFLOPS |
| Peak FP64 | 0.5-5 TFLOPS | 10-20 TFLOPS |
| Memory Bandwidth | 50-100 GB/s | 500-3000 GB/s |
| Power | 100-300W | 250-700W |
| TFLOPS/Watt | 0.01-0.05 | 0.05-0.3 |

### When to Use CPU vs GPU

**Use CPU for:**

- Sequential algorithms with complex control flow
- Small datasets (< 10,000 elements)
- Tasks requiring low latency (single operation)
- I/O-bound operations
- Irregular memory access patterns
- Code with many branches and conditionals

**Use GPU for:**

- Data-parallel algorithms (same operation on many elements)
- Large datasets (millions of elements)
- Regular, predictable memory access patterns
- Compute-intensive operations (matrix multiplication, FFT)
- Can tolerate higher latency if throughput is high
- Operations that can be expressed as kernels on arrays


## SIMD vs SIMT: Detailed Comparison

Understanding the differences between CPU SIMD and GPU SIMT helps you leverage both effectively.

### Execution Models

**CPU SIMD (Single Instruction, Multiple Data):**

- Single instruction operates on fixed-size vector register
- 4-16 data elements processed in parallel
- Programmer manages vector operations explicitly (or compiler auto-vectorizes)
- Limited by vector register width

**GPU SIMT (Single Instruction, Multiple Threads):**

- Single instruction issued to group of threads (warp)
- 32 threads (NVIDIA) or 64 threads (AMD) execute together
- Each thread has its own registers and can follow independent control flow
- Massive scalability: thousands of warps

### Comparison Table

| Feature | CPU SIMD | GPU SIMT |
|---------|----------|----------|
| **Parallelism Scale** | 4-16 way per core | 1000s of threads per GPU |
| **Execution Unit** | Vector register (256-512 bit) | Warp (32 threads on NVIDIA) |
| **Control Flow** | Difficult with divergence | Handles divergence naturally |
| **Memory Model** | Shared cache hierarchy | Separate memory spaces (global, shared, registers) |
| **Latency Hiding** | Cache reduces memory latency | Thread switching hides latency |
| **Programming Model** | Vectorized loops or intrinsics | Kernel functions with thread indexing |
| **Best For** | Moderate parallelism within CPU | Massive data parallelism |
| **Peak Performance** | ~1-10 TFLOPS | ~20-80 TFLOPS |

### Control Flow Divergence

**CPU SIMD Divergence:**

```c
// SIMD code with divergence
for (int i = 0; i < 8; i++) {
    if (data[i] > 0) {
        result[i] = compute_A(data[i]);  // Complex with SIMD
    } else {
        result[i] = compute_B(data[i]);
    }
}
```

- SIMD struggles with divergent paths
- Often requires predication or masking
- Compiler may serialize the operations
- Performance degrades significantly

**GPU SIMT Divergence:**

```cuda
__global__ void kernel(float* data, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (data[idx] > 0) {
            result[idx] = compute_A(data[idx]);  // Natural with SIMT
        } else {
            result[idx] = compute_B(data[idx]);
        }
    }
}
```

- SIMT handles divergence naturally
- Warp executes both paths with different threads active
- Uses per-thread active masks
- Performance impact: 2x slowdown (not serialization)
- Modern GPUs (Volta+) have Independent Thread Scheduling

**Warp Divergence Example:**

```
Warp with 32 threads:
Threads 0-15: data[i] > 0 is true → execute compute_A()
Threads 16-31: data[i] > 0 is false → execute compute_B()

Execution:
Step 1: Run compute_A() with threads 0-15 active (threads 16-31 masked)
Step 2: Run compute_B() with threads 16-31 active (threads 0-15 masked)
Step 3: Reconverge - all threads continue together
```


## NVIDIA GPU Architecture

NVIDIA GPUs dominate scientific computing. Understanding their architecture is essential for Python GPU programming.

### Streaming Multiprocessors (SMs)

**Core Building Block:**

- GPU consists of multiple Streaming Multiprocessors (SMs)
- Each SM contains many CUDA cores (Streaming Processors)
- SMs execute thread blocks independently
- Number of SMs varies by GPU model

**Example GPU Configurations:**

| GPU Model | Architecture | SMs | CUDA Cores | FP32 TFLOPS | Memory |
|-----------|-------------|-----|------------|-------------|--------|
| RTX 3090 | Ampere | 82 | 10,496 | 35.6 | 24 GB |
| A100 (40GB) | Ampere | 108 | 6,912 | 19.5 | 40 GB |
| H100 | Hopper | 132 | 16,896 | 51.0 | 80 GB |
| RTX 4090 | Ada Lovelace | 128 | 16,384 | 82.6 | 24 GB |

**SM Components:**

- CUDA Cores: Execute arithmetic operations
- Special Function Units (SFUs): Fast transcendentals (sin, cos, exp)
- Tensor Cores: Matrix multiply-accumulate (AI/ML acceleration)
- Shared Memory: Fast on-chip memory (48-164 KB per SM)
- Registers: Per-thread storage (~64K registers per SM)
- Warp Scheduler: Selects warps for execution

### Thread Hierarchy: Grids, Blocks, Warps

**Three-Level Organization:**

**1. Thread (Finest Granularity):**

- Single execution context running kernel code
- Has unique thread ID within its block
- Operates on one data element
- Has private registers and local memory

**2. Block (Thread Block):**

- Group of threads (typically 128-1024 threads)
- All threads in a block execute on the same SM
- Threads can cooperate via shared memory
- Can synchronize using barriers (`__syncthreads()`)
- Block has unique block ID within the grid

**3. Grid:**

- Collection of blocks executing the same kernel
- Can contain thousands of blocks
- Blocks execute independently (no synchronization across blocks)
- Grid spans entire problem domain

**4. Warp (Execution Unit):**

- Group of 32 threads that execute together
- Fundamental unit of execution on NVIDIA GPUs
- All threads in a warp execute same instruction simultaneously
- Hardware schedules entire warps, not individual threads

**Visual Hierarchy:**

```
Grid (launched by CPU)
  ├── Block 0 (executes on SM)
  │     ├── Warp 0 (threads 0-31)
  │     ├── Warp 1 (threads 32-63)
  │     ├── Warp 2 (threads 64-95)
  │     └── Warp 3 (threads 96-127)
  ├── Block 1 (executes on SM)
  │     ├── Warp 0
  │     └── ...
  └── Block N-1
```

### Latency Hiding Through Parallelism

**The Problem:**

- Global memory access takes 400-800 cycles
- Arithmetic operations are much faster (1-10 cycles)
- If GPU waited for memory, cores would be idle

**The Solution:**

- SM has many more warps resident than it can execute simultaneously
- When a warp stalls (waiting for memory), SM instantly switches to another ready warp
- Zero-overhead scheduling: no context switch cost
- Keeps compute units busy while memory operations complete

**Example:**

```
SM with 64 CUDA cores, can execute 2 warps simultaneously:
- 32 resident warps total (1024 threads)
- Currently executing: Warp 0 and Warp 1
- Warp 0 stalls on memory load
- SM immediately switches to Warp 2 (no overhead)
- Warp 0's memory load completes in background
- Warp 0 becomes ready again, will execute later
```

**Occupancy:**

- Percentage of maximum resident warps actually achieved
- Higher occupancy = better latency hiding
- Limited by register usage and shared memory per thread/block
- Target: 50-100% occupancy for good performance


## GPU Memory Hierarchy

GPUs have multiple memory types optimized for different access patterns.

### Memory Types

**1. Global Memory (Device Memory):**

- Largest capacity (8-80 GB)
- Accessible by all threads and host CPU
- Slowest (~400-800 cycles latency)
- Highest bandwidth (500-3000 GB/s)
- Off-chip GDDR6/HBM2/HBM3
- This is the "24 GB" or "80 GB" in GPU specs

**2. Shared Memory:**

- Small capacity (~48-164 KB per SM)
- Accessible by threads in same block only
- Fast (~1-2 cycles latency)
- On-chip memory (part of SM)
- Programmer-managed
- Used for thread cooperation and data reuse

**3. Registers:**

- Tiny capacity (~64K 32-bit registers per SM)
- Private to each thread
- Fastest (<1 cycle)
- On-chip, part of SM
- Compiler-managed
- Limited supply affects occupancy

**4. L1/L2 Cache:**

- L1: 128 KB per SM (Ampere/Hopper)
- L2: 6-50 MB shared across GPU
- Automatic caching of global memory
- Hardware-managed
- Reduces global memory latency

**5. Constant Memory:**

- Small capacity (64 KB)
- Read-only from kernels
- Cached and broadcast efficiently
- Good for parameters used by all threads

### Memory Performance

**Bandwidth Comparison:**

| Memory Type | Latency | Bandwidth | Capacity |
|-------------|---------|-----------|----------|
| Registers | <1 cycle | ~20 TB/s | ~256 KB/SM |
| Shared Memory | 1-2 cycles | ~15 TB/s | 48-164 KB/SM |
| L1 Cache | ~30 cycles | ~10 TB/s | 128 KB/SM |
| L2 Cache | ~200 cycles | ~3 TB/s | 6-50 MB |
| Global Memory | 400-800 cycles | 1-3 TB/s | 8-80 GB |

### Memory Access Patterns

**Coalesced Access (Efficient):**

- Consecutive threads access consecutive memory addresses
- GPU combines into single memory transaction
- Example: `data[threadIdx.x]` with 32 threads → one transaction
- Maximum bandwidth utilization

**Strided Access (Less Efficient):**

- Threads access memory with constant stride
- May require multiple memory transactions
- Example: `data[threadIdx.x * 2]` → two transactions for warp
- Performance: ~50% of coalesced

**Random Access (Inefficient):**

- Threads access unpredictable memory locations
- Each access may be separate transaction
- Poor cache utilization
- Performance: <10% of coalesced


## Other GPU Architectures

While NVIDIA dominates, other vendors offer alternatives.

### AMD ROCm

**Platform:**

- ROCm: Radeon Open Compute (open-source)
- HIP: Portable GPU programming (CUDA-compatible syntax)
- Growing ecosystem, smaller than CUDA

**Python Support:**

- CuPy: Experimental ROCm backend
- PyTorch: ROCm support
- TensorFlow: ROCm builds available

**Terminology Differences:**

| Concept | NVIDIA CUDA | AMD ROCm |
|---------|-------------|----------|
| Processing Unit | Streaming Multiprocessor (SM) | Compute Unit (CU) |
| Execution Unit | CUDA Core | Stream Processor |
| Thread Group | Warp (32 threads) | Wavefront (64 threads) |
| Fast Memory | Shared Memory | Local Data Share (LDS) |

**Hardware:**

- Radeon Instinct MI100/MI200/MI300: Data center compute
- RX 6000/7000: Consumer GPUs with limited compute support

### Intel oneAPI

**Platform:**

- oneAPI: Cross-architecture programming
- SYCL: Open standard based on C++
- Supports Intel GPUs, CPUs, FPGAs

**Hardware:**

- Intel Data Center GPU Max series
- Intel Arc: Consumer GPUs
- Integrated GPUs in Intel CPUs

**Status:** Emerging platform, smaller ecosystem

### Apple Silicon

**Platform:**

- Metal: GPU programming framework
- Unified memory (CPU and GPU share RAM)
- Metal Performance Shaders (MPS)

**Python Support:**

- PyTorch: MPS backend
- TensorFlow: Metal plugin

**Characteristics:**

- No discrete GPU memory copies needed
- Good power efficiency
- Limited to Apple hardware

### Platform Comparison

| Platform | Vendor | Maturity | Ecosystem | Python Support |
|----------|--------|----------|-----------|----------------|
| NVIDIA CUDA | NVIDIA only | Mature (18+ years) | Extensive | Excellent |
| AMD ROCm | AMD only | Growing (8+ years) | Moderate | Good |
| Intel oneAPI | Intel only | New (3+ years) | Emerging | Limited |
| Apple Metal | Apple only | Mature (graphics) | Moderate | Limited |
| OpenCL | Multi-vendor | Mature but stagnant | Moderate | Moderate |


## Summary: Key Concepts for Python Programming

### Essential Concepts

**1. CPU SIMD:**

- 4-16 way parallelism using vector registers
- NumPy leverages AVX/AVX-512 automatically
- Good for moderate parallelism on CPU
- Limited scalability compared to GPU

**2. GPU SIMT:**

- Thousands of threads executing concurrently
- Warp-based execution (32 threads per warp on NVIDIA)
- Massive parallelism for data-parallel workloads
- Higher throughput than CPU SIMD

**3. Latency vs Throughput:**

- CPU optimized for latency (single operation fast)
- GPU optimized for throughput (many operations total)
- Choose based on problem characteristics

**4. Memory Hierarchy:**

- GPU has complex memory hierarchy
- Coalesced access patterns critical for performance
- Global memory is large but slow
- Shared memory is small but fast

**5. Control Flow:**

- CPU SIMD struggles with divergence
- GPU SIMT handles divergence naturally (with performance cost)
- Minimize divergence within warps for best performance

### For Python GPU Frameworks

**NumPy (CPU SIMD):**

```python
import numpy as np
c = a + b  # Uses AVX-512 on supported CPUs
```

**CuPy (GPU, High-Level):**

```python
import cupy as cp
c = a + b  # Same API, runs on GPU with massive parallelism
```

**Numba (GPU, Medium-Level):**

```python
from numba import cuda
@cuda.jit
def kernel(a, b, c):
    i = cuda.grid(1)
    if i < c.size:
        c[i] = a[i] + b[i]  # Explicit thread programming
```

**JAX (GPU, Functional):**

```python
import jax.numpy as jnp
c = a + b  # Auto-parallelized, auto-differentiated
```

### When GPU Accelerates Your Code

**Good for GPU:**

- Large arrays (millions of elements)
- Data-parallel operations (element-wise, matrix ops)
- Regular memory access patterns
- Many operations per data element
- Data stays on GPU across multiple operations

**Not Good for GPU:**

- Small datasets (< 10,000 elements)
- Sequential algorithms
- Complex control flow with divergence
- Irregular memory access
- One-off operations (transfer overhead dominates)

### Performance Expectations

**Typical Speedups:**

- CPU SIMD over scalar: 3-15x
- GPU over CPU (well-suited problems): 10-1000x
- GPU over CPU (poorly-suited problems): <2x or slower

**Example:**

- Matrix multiplication (5000×5000):
  - Pure Python: ~300 seconds
  - NumPy (CPU SIMD): ~3 seconds (100x faster)
  - CuPy (GPU): ~0.03 seconds (10,000x faster than Python, 100x faster than NumPy)

This foundation prepares you for GPU programming in Python using CUDA, CuPy, Numba, CUDA Python, and JAX covered in the following sections.
