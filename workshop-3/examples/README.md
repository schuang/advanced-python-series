# Numba Examples

This directory contains standalone runnable examples demonstrating Numba's capabilities for CPU and GPU acceleration.

## Requirements

**All Examples:**
```bash
pip install numba numpy
```

**GPU Examples (requires NVIDIA GPU with CUDA):**
- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- CUDA Toolkit installed
- Verify GPU support:
  ```bash
  python -c "from numba import cuda; print(cuda.is_available())"
  ```

## Examples Overview

### GPU Examples

#### 1. `vecadd-numba.py` - Vector Addition (GPU Basics)

**What it demonstrates:**
- Defining CUDA kernels with `@cuda.jit`
- GPU memory management (`cuda.to_device`, `cuda.device_array_like`)
- Thread indexing with `cuda.grid()`
- Kernel launch configuration (blocks and threads)
- Boundary checking

**Key concepts:**
- GPU kernel programming fundamentals
- Memory transfers between host and device
- Optimal thread block size selection

**Run:**
```bash
python vecadd-numba.py
```

**Expected output:**
- Test results for various array sizes (1K to 10M elements)
- GPU execution times
- Verification against CPU computation

---

#### 2. `monte-carlo-pi-gpu.py` - Monte Carlo Pi Estimation (GPU)

**What it demonstrates:**
- GPU kernel for parallel random number generation
- Reduction operations across GPU threads
- Optimal kernel launch configuration
- Performance scaling with sample size

**Key concepts:**
- Parallel Monte Carlo methods
- Thread-local random number generation
- GPU speedup for embarrassingly parallel problems

**Run:**
```bash
python monte-carlo-pi-gpu.py
```

**Expected output:**
- Pi estimates for 1M to 500M samples
- GPU execution times and throughput
- Accuracy comparison with actual π value

---

#### 3. `monte-carlo-pi-vectorized.py` - Monte Carlo (Vectorized GPU)

**What it demonstrates:**
- `@vectorize` decorator for automatic GPU parallelization
- Simpler code than explicit CUDA kernels
- Automatic thread management
- Easy CPU/GPU portability

**Key concepts:**
- High-level GPU programming with `@vectorize`
- Element-wise operations on GPU
- Trade-off: simplicity vs performance

**Run:**
```bash
python monte-carlo-pi-vectorized.py
```

**Expected output:**
- Pi estimates using vectorized GPU implementation
- Comparison of simplicity vs explicit kernel approach

---

### CPU Examples

#### 4. `monte-carlo-pi-cpu.py` - Monte Carlo Pi Estimation (CPU)

**What it demonstrates:**
- CPU acceleration with `@jit` decorator
- JIT compilation overhead and warmup
- Speedup comparison with pure Python
- Numba's effectiveness for loop-heavy code

**Key concepts:**
- Nopython mode for maximum performance
- JIT compilation workflow
- Typical CPU speedups (50-100x)

**Run:**
```bash
python monte-carlo-pi-cpu.py
```

**Expected output:**
- Pi estimates for various sample sizes
- Comparison: Pure Python vs Numba JIT
- Speedup measurements

---

#### 5. `numba-cpu-parallel.py` - Multi-Core CPU Parallelization

**What it demonstrates:**
- Automatic parallelization with `parallel=True`
- Explicit parallel loops with `prange`
- Parallel reductions (sum, dot product)
- Nested parallel loops for 2D operations
- Thread control and tuning
- Performance scaling analysis

**Key concepts:**
- Multi-core CPU parallelization
- SMP (Symmetric Multi-Processing) execution
- GIL release for true parallel execution
- When parallelization helps vs overhead

**Run:**
```bash
python numba-cpu-parallel.py
```

**Expected output:**
- 6 examples demonstrating different parallelization techniques
- Sequential vs parallel timing comparisons
- Speedup analysis across different array sizes
- Thread scaling behavior

---

## Example Progression

**Recommended order for learning:**

1. **Start with CPU**: `monte-carlo-pi-cpu.py`
   - Understand JIT compilation basics
   - See speedup without GPU complexity

2. **CPU Parallelization**: `numba-cpu-parallel.py`
   - Learn multi-core parallelization
   - Understand parallel=True and prange
   - See when parallelization helps

3. **GPU Basics**: `vecadd-numba.py`
   - Learn GPU kernel fundamentals
   - Understand memory management
   - Master thread indexing

4. **GPU Application**: `monte-carlo-pi-gpu.py`
   - Apply GPU knowledge to real problem
   - See massive parallelism in action
   - Compare CPU vs GPU performance

5. **Simplified GPU**: `monte-carlo-pi-vectorized.py`
   - Learn high-level GPU programming
   - Understand @vectorize decorator
   - Appreciate simplicity vs control trade-off

## Performance Expectations

**CPU JIT (@jit):**
- Speedup: 50-100x over pure Python
- Best for: Loop-heavy numerical code
- Overhead: First call compilation (~1s for complex functions)

**CPU Parallel (parallel=True, prange):**
- Speedup: 2-8x over sequential Numba (depends on cores)
- Best for: Large arrays (>1M elements), independent operations
- Overhead: Thread creation and synchronization

**GPU (@cuda.jit):**
- Speedup: 10-1000x over pure Python (problem dependent)
- Best for: Very large arrays (>10M elements), data-parallel operations
- Overhead: Memory transfers, kernel launch

**GPU Vectorized (@vectorize):**
- Speedup: Similar to @cuda.jit for simple operations
- Best for: Element-wise operations, rapid prototyping
- Limitation: Less control than explicit kernels

## Common Issues and Solutions

### GPU Not Detected

**Problem:**
```
CUDA is not available. This example requires an NVIDIA GPU.
```

**Solutions:**
1. Verify CUDA Toolkit is installed:
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. Check Numba CUDA support:
   ```bash
   python -c "from numba import cuda; print(cuda.is_available())"
   ```

3. Reinstall numba if needed:
   ```bash
   pip uninstall numba
   pip install numba
   ```

### Slow First Run

**Problem:** First execution is very slow

**Explanation:** This is normal! Numba compiles functions on first use (JIT = Just-In-Time). Subsequent runs use cached compiled code.

**Solution:** Examples include warmup calls before timing measurements.

### Memory Errors on GPU

**Problem:**
```
cuda.cudadrv.driver.CudaAPIError: [700] CUDA_ERROR_ILLEGAL_ADDRESS
```

**Common causes:**
- Array index out of bounds (check boundary conditions)
- Race conditions in kernel
- Incorrect memory allocation

**Solution:** Add boundary checks in kernels, verify array sizes match.

## Modifying Examples

All examples are well-commented and designed to be educational. Feel free to:

- Change array sizes to see performance scaling
- Modify thread configurations (threads_per_block, blocks_per_grid)
- Add your own kernels and computations
- Compare different approaches

## Additional Resources

- **Numba Documentation**: https://numba.pydata.org/
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/
- **Workshop Materials**: See `../30-numba.md` for comprehensive guide

## Questions or Issues?

These examples are part of the Advanced Python GPU Computing workshop. For questions:
1. Review the workshop materials in `../30-numba.md`
2. Check Numba documentation
3. Verify your CUDA installation and GPU compatibility

---

**Note:** Performance numbers vary based on:
- CPU: cores, clock speed, cache size
- GPU: model, CUDA cores, memory bandwidth
- Problem size and complexity
- System load and thermal throttling
