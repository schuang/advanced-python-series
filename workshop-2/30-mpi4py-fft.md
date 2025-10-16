# Parallel FFT

- FFT is widely used in scientific computing (spectral solvers, turbulence analysis, seismic imaging). 
- Large-scale FFT computations are done in parallel with MPI. 
- We use `mpi4py` to orchestrate data decomposition and local FFT kernels, connecting the workshop examples with production-ready Python toolchains.

## Fast Fourier Transform Primer

The Fast Fourier Transform (FFT) is the workhorse algorithm that converts regularly sampled data from the time or spatial domain into its frequency components, doing the same job as a discrete Fourier transform but about $O(N \log N)$ fast instead of $O(N^2)$. That speedup turned spectral methods from a theoretical curiosity into a daily tool for scientists: we can isolate dominant wavelengths in turbulence snapshots, track seismic reflections buried in noisy sensor arrays, or reconstruct medical images from raw k-space samples within seconds.

For $N$ complex sequence $x_n$, the discrete Fourier transform is

$$ 
X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i k n / N},    k = 0, …, N-1.
$$

The FFT is any algorithm that evaluates this sum for all $k$ in $O(N \log N)$ operations (Cooley & Tukey, 1965).

### FFT Applications

- **Turbulence & climate modeling (large data):** spectral solvers advance Navier–Stokes equations by hopping between velocity space and frequency space each timestep, and 3D grids at 2048³ resolution only fit when the FFT is parallelized; MPI lets us decompose the grid and keep turnaround times compatible with daily forecast cycles.

- **Seismic & subsurface imaging (large data):** batched FFTs across thousands of geophones highlight stratigraphic boundaries that hint at oil, gas, or geothermal reservoirs; exploration crews expect overnight processing, so distributed FFT pipelines convert petabytes of field shots into actionable maps on schedule.

- **Biomedical imaging (MRI, CT, PET, ultrasound Doppler) (large data + latency pressure):** scanners record k-space or projection data in the frequency domain; fast inverse FFTs turn it into anatomy, while Doppler FFTs extract blood-flow velocities in real time—parallel transforms shave minutes off scans, enabling shorter breath-holds, reduced anesthesia time, and near-real-time interventional guidance.

- **Biomedical signals:** spectral decomposition exposes arrhythmias, seizure signatures, and sleep-stage bands, enabling clinicians to work directly in the frequency domain.

- **Spectroscopy & metabolomics:** FFTs transform time-domain free-induction decays and transient ion signals into molecular fingerprints.

- **Accelerator physics & plasma codes (large data):** FFT-based Poisson solvers update electric potentials across colossal meshes that span entire accelerator rings; high-throughput transforms keep pace with beam dynamics simulations tied to accelerator operations.

- **Signal processing & radio astronomy (large data):** correlating antenna feeds or filtering instrument noise uses FFTs to peel out astrophysical signals from static; instruments like the Square Kilometre Array ingest terabits per second, so parallel FFT stages are the only way to keep up with the data firehose.

Each example mirrors the cases we will explore: big multidimensional arrays, repeated forward/backward transforms, and the need to keep communications efficient as we scale to more ranks or GPUs.

### References

- J. W. Cooley and J. W. Tukey, “An algorithm for the machine calculation of complex Fourier series,” *Mathematics of Computation*, 1965.
- M. Frigo and S. G. Johnson, “The Design and Implementation of FFTW3,” *Proceedings of the IEEE*, 2005.
- FFTW website: https://www.fftw.org/

## Parallel strategy

- Data layout: slice a multidimensional array along one axis so each rank owns a contiguous block that fits in memory. 

- Local work: execute `numpy.fft` (or `pyfftw.interfaces.numpy_fft`) on the assigned block; these kernels are already threaded and vectorized. 

- Communication: exchange or gather transformed slabs so that boundary conditions or spectral post-processing can continue. 

- Scaling step: for serious 3D FFTs, use pencil or slab decompositions plus all-to-all exchanges; libraries like `mpi4py-fft` and PETSc automate those patterns.

## Sequential vs Parallel FFT

### Sequential version (`examples/05_fft_sequential.py`)

The sequential example establishes a baseline by computing a 2D FFT on a single process with plain NumPy:

```python
rows = 12
cols = 32

# set up data
x = np.linspace(0, 2 * np.pi, cols, endpoint=False)
base_signal = np.sin(3 * x) + 0.3 * np.sin(9 * x)
data = np.vstack([np.roll(base_signal, shift) for shift in range(rows)])

# compute FFT
fft_result = np.fft.fft(data, axis=1)
```

**How it works:**

1. Creates a 12×32 array where each row is a phase-shifted version of a composite sine wave (3x and 9x harmonics).

2. Applies `np.fft.fft` along axis=1 (columns), transforming each row independently into the frequency domain.

3. The result is a 12×32 complex array containing frequency coefficients.

This sequential approach is simple and efficient for small arrays but doesn't scale to large datasets that exceed single-node memory or require faster turnaround.

**run:** `python examples/05_fft_sequential.py`

### Parallel Row-wise FFT (`examples/05_mpi_parallel_fft.py`)

The parallel example distributes the same workload across multiple MPI ranks using a **scatter-compute-gather** pattern:

```python
# Rank 0 creates the data
if rank == root:
    x = np.linspace(0, 2 * np.pi, cols, endpoint=False)
    base_signal = np.sin(3 * x) + 0.3 * np.sin(9 * x)
    data = np.vstack([np.roll(base_signal, shift) for shift in range(rows)])
    chunks = np.array_split(data, size, axis=0)
else:
    chunks = None

# Scatter row blocks to all ranks
local_block = comm.scatter(chunks, root=root)

# Each rank computes FFT on its local rows
local_fft = np.fft.fft(local_block, axis=1)

# Gather results back to rank 0
gathered = comm.gather(local_fft, root=root)

# Rank 0 validates against sequential reference
if rank == root:
    fft_result = np.vstack(gathered)
    reference = np.fft.fft(data, axis=1)
    max_error = np.max(np.abs(fft_result - reference))
```

**How it works:**
1. **Data generation (rank 0):** Creates the same 12×32 signal array as the sequential version and splits it into `size` chunks along axis=0 (rows). With 4 ranks, each chunk contains 3 rows.

2. **Scatter:** `comm.scatter()` distributes one chunk to each rank. After scattering, rank 0 holds rows 0-2, rank 1 holds rows 3-5, rank 2 holds rows 6-8, and rank 3 holds rows 9-11.

3. **Local FFT:** Each rank independently computes `np.fft.fft(local_block, axis=1)`, transforming its assigned rows. Since FFT along axis=1 treats each row independently, no inter-rank communication is needed during this step.

4. **Gather:** `comm.gather()` collects all transformed blocks back to rank 0, which stacks them with `np.vstack()` to reconstruct the complete 12×32 result.

5. **Validation:** Rank 0 compares the parallel result against the sequential baseline to verify correctness (error should be near machine epsilon).

**Key insight:** Row-wise FFT is embarrassingly parallel when transforming along axis=1 because each row's transform is independent. Communication overhead is limited to the initial scatter and final gather.

**Run:** `mpiexec -n 4 python examples/05_mpi_parallel_fft.py`

NOTE: The scatter-compute-gather pattern is similar to [Hadoop](https://hadoop.apache.org/)/[Spark](https://spark.apache.org/)'s map-reduce pattern. But they differ in implementation details and use cases. 

MPI (scatter-compute-gather):

- Numerical simulations requiring low latency
- Problems where all ranks must synchronize frequently
- HPC clusters with fast interconnects
- Code needs fine-grained control over communication

MapReduce/Spark:

- Data analytics on petabyte datasets
- Embarrassingly parallel tasks with minimal inter-task communication
- Need fault tolerance for long-running jobs
- Prefer declarative APIs over manual message passing



## Using mpi4py-fft

- The `mpi4py-fft` library provides high-level abstractions for parallel FFTs, handling domain decomposition, transposes, and communication automatically. 

- This example demonstrates a true 2D FFT (transforming along both axes) rather than just row-wise transforms.

### 2D FFT (`examples/06_mpi4py_fft.py`)


```python
from mpi4py_fft import PFFT, newDistArray

global_shape = (12, 32)

# Plan a 2D complex FFT across all available ranks
fft = PFFT(comm, global_shape, axes=(0, 1), dtype=np.complex128)

# Create distributed arrays - each rank owns a portion of data
u = newDistArray(fft, forward_output=False)   # spatial domain
uh = newDistArray(fft, forward_output=True)   # frequency domain

# Fill local portion with reproducible random data
rng = np.random.default_rng(seed=13 + rank)
u[:] = rng.standard_normal(u.shape)
u_original = u.copy()

# Forward and backward transforms
fft.forward(u, uh)
fft.backward(uh, u)

# Validate reconstruction (accounting for unnormalized backward FFT)
total_points = np.prod(global_shape)
u /= total_points
local_error = np.max(np.abs(u - u_original))
global_error = comm.allreduce(local_error, op=MPI.MAX)
```

**How it works:**

1. **Planning:** `PFFT(comm, global_shape, axes=(0, 1))` creates an FFT plan for a 12×32 global array, transforming along both axes. The library automatically:
   - Decomposes the domain across available ranks (typically splitting along axis=0).
   - Plans the necessary local FFTs and all-to-all transposes.
   - Optimizes memory layouts and communication patterns.

2. **Distributed arrays:** `newDistArray()` allocates arrays that are partitioned across ranks. Each rank owns a contiguous slab:
   - With 4 ranks, rank 0 might own rows 0-2, rank 1 owns rows 3-5, etc.
   - The library tracks global vs. local indexing and manages ghost/halo regions automatically.

3. **Data initialization:** Each rank fills its local portion with random data seeded by `13 + rank`, ensuring reproducibility while keeping data decorrelated across ranks.

4. **Forward transform:** `fft.forward(u, uh)` performs a full 2D FFT:
   - First, local 1D FFTs along axis=1 (columns within each rank's slab).
   - Then, an MPI transpose redistributes data so each rank owns a different slab for axis=0 transforms.
   - Finally, local 1D FFTs along axis=0 complete the 2D transform.
   - The result `uh` is stored in a distributed frequency-domain array.

5. **Backward transform:** `fft.backward(uh, u)` reverses the process with inverse FFTs and transposes. Note that the backward FFT is unnormalized (standard FFTW convention), so we must divide by `total_points` to recover the original data.

6. **Validation:** Each rank computes its local max error, then `MPI.MAX` allreduce finds the global maximum. The error should be near machine epsilon ($\approx 10^{-15}$) for float64).

**Key differences from manual parallelization:**

| Aspect | Manual (`05_mpi_parallel_fft.py`) | Library (`06_mpi4py_fft.py`) |
|--------|-----------------------------------|------------------------------|
| **Transform dimensionality** | 1D (row-wise only) | True 2D (both axes) |
| **Decomposition** | Explicit scatter/gather | Automatic domain partitioning |
| **Communication** | User manages | Hidden behind library API |
| **Transpose handling** | Not needed (1D transforms) | Automatic all-to-all exchanges |
| **Scalability** | Limited to embarrassingly parallel cases | Scales to 3D, pencil decompositions |
| **Code complexity** | ~40 lines | ~20 lines |

**When to use mpi4py-fft:**
- Multi-dimensional FFTs where transforms span multiple axes.
- Large-scale 3D problems (turbulence, seismic imaging, cosmology) that require pencil decompositions.
- Production workflows where you need battle-tested transpose algorithms and FFTW integration.
- When you want to minimize communication code and focus on physics/algorithms.

**Requirements:** `python -m pip install mpi4py-fft` (FFTW libraries strongly recommended for performance).

**Launch:** `mpiexec -n 4 python examples/06_mpi4py_fft.py`



## Notes

Use `mpi4py-fft` for turnkey pencil decompositions, transpose-heavy 3D plans, and collective-aware FFTW plans. Pair `mpi4py` with `pyFFTW` or MKL-accelerated NumPy when node-level performance is critical. PETSc (`petsc4py`) provides FFT-friendly distributed arrays (`DMDA`) that integrate with solvers and time integrators. Profile communication vs compute: large FFTs often become network-bound without topology-aware layouts.

MRI scaling takeaway: 128×128 phantoms fit on a laptop, but clinical slices exceed 1024×1024 and multi-coil stacks multiply the data volume. Parallel FFT shortens reconstruction latency. Streaming benefit: with MPI you can reconstruct subsets of k-space as they arrive, overlapping acquisition and reconstruction for near real-time imaging.

## Discussion Prompts

Where do FFTs appear in your pipelines (spectral PDEs, filtering, convolution)? Which axis should you decompose first when memory grows faster than cores? How would you extend the row-wise pattern to a full 3D FFT with multiple transpose stages?
