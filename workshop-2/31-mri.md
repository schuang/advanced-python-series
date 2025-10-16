# MRI Reconstruction with Parallel FFT

## MRI overview

- Magnetic Resonance Imaging (MRI) reconstruction presents a canonical application of parallel FFT algorithms. 

- Unlike conventional imaging where sensors directly capture spatial intensity, MRI scanners measure **k-space**—the 2D or 3D Fourier transform of the spin density. 

- Reconstruction requires applying an inverse FFT to convert frequency-domain measurements into interpretable anatomical images.

### K-space and the Fourier Encoding

- In MRI physics, spatial information is encoded through magnetic field gradients that create position-dependent resonance frequencies. 
- The raw signal received by RF coils populates k-space, where:

- **k-space coordinates** ($k_x$, $k_y$, $k_z$) represent spatial frequencies, not physical positions.
- The **center of k-space** captures low spatial frequencies (coarse anatomy, contrast).
- The **periphery of k-space** captures high spatial frequencies (edges, fine detail).

Each k-space trajectory (typically line-by-line in Cartesian acquisition) is related to the image I(x,y) by the 2D Fourier transform:

$$
S(k_x, k_y) = ∫∫ I(x,y) exp(-2πi(k_x·x + k_y·y)) dx dy
$$

Reconstruction inverts this relationship:

```
I(x,y) = IFFT2D[S(k_x, k_y)]
```

### Computational Challenge

The computational burden scales with resolution and hardware complexity:

- **Clinical scans:** 256×256 complex-valued arrays (131K points per slice, multiple slices per volume).
- **Research protocols:** 512×512 or 1024×1024 grids (262K to 1M points).
- **Phased array coils:** 8–32 independent receiver channels, each requiring separate reconstruction and coil combination.
- **Real-time constraints:** Interventional imaging (MR-guided surgery, cardiac cine) demands reconstruction latencies under 50 ms.

Parallel FFT infrastructure directly addresses these demands: distributing k-space rows or employing slab/pencil decompositions reduces wall-clock time, enables higher throughput, and makes real-time applications feasible.

## Test Data: Shepp-Logan Phantom

To demonstrate reconstruction algorithms without requiring real scanner data, we use the **Shepp-Logan phantom**—a standard test image in medical imaging since 1974. This synthetic phantom consists of overlapping ellipses with intensities representing different tissue types (gray matter, white matter, CSF, bone).

The phantom provides several advantages for algorithm development:
- **Ground truth:** Known analytical form allows precise error quantification.
- **Reproducibility:** Deterministic, no patient privacy concerns, consistent across studies.
- **Controlled complexity:** Captures essential features (piecewise constant regions, smooth boundaries) without MRI physics complications.

Our examples use `_phantoms.shepp_logan()` to generate a 128×128 phantom, then simulate k-space acquisition via forward FFT. This lets us validate reconstruction correctness by comparing the reconstructed image against the original phantom.

## Implementation Examples

### Sequential Baseline (`examples/07_mri_fft_sequential.py`)

The sequential implementation establishes a reference reconstruction using NumPy's optimized FFT routines:

```python
from _phantoms import shepp_logan

shape = (128, 128)
phantom = shepp_logan(shape)          # Generate test image (128×128 real-valued)
kspace = np.fft.fft2(phantom)         # Simulate k-space: forward FFT
recon = np.fft.ifft2(kspace)          # Reconstruct: inverse FFT

max_err = np.max(np.abs(recon.real - phantom))
```

**Pipeline:**

1. **Phantom generation:** `shepp_logan(shape)` returns a 128×128 array with intensity values $\in$ [0, 1] representing tissue contrast (CSF $\approx$ 0, bone $\approx$ 1, soft tissue intermediate).

2. **K-space simulation:** `np.fft.fft2(phantom)` computes the 2D DFT, yielding a 128×128 complex array. In real acquisitions, this k-space data comes directly from the scanner; here we generate it synthetically to isolate reconstruction logic from acquisition physics.

3. **Reconstruction:** `np.fft.ifft2(kspace)` inverts the transform. Since the forward FFT was applied to real data, the result is Hermitian-symmetric; we extract `recon.real` for comparison.

4. **Validation:** The max absolute error should be O($\varepsilon$_machine) $\approx$  $10^{-16}$  for float64, confirming that FFT $\rightarrow$ IFFT is numerically stable.

**Observed output:**
```
Phantom shape: (128, 128)
Sequential recon max error: 3.331e-16
```

This establishes the baseline for parallel correctness checks.

**Launch:** `python examples/07_mri_fft_sequential.py`

### Parallel Row-Decomposition (`examples/07_mpi_fft_parallel.py`)

The parallel implementation uses **row-wise domain decomposition** to distribute the 2D IFFT across ranks. Since the 2D transform factors as a sequence of 1D transforms along each axis, we can parallelize stages that operate independently within row blocks:

```python
# Rank 0: Generate data and partition
if rank == root:
    phantom = shepp_logan(shape)
    kspace = np.fft.fft2(phantom)
    reference = np.fft.ifft2(kspace)  # Sequential baseline
    chunks = np.array_split(kspace, size, axis=0)
else:
    chunks = None

# Distribute k-space rows
local_kspace = comm.scatter(chunks, root=root)

# Stage 1: Independent column transforms (axis=1)
stage_one = np.fft.ifft(local_kspace, axis=1)

# Gather partial results
partials = comm.gather(stage_one, root=root)

# Stage 2: Complete row transforms (axis=0) on rank 0
if rank == root:
    assembled = np.vstack(partials)
    recon = np.fft.ifft(assembled, axis=0)
    max_err = np.max(np.abs(recon - reference))
```

**Algorithm breakdown:**

1. **Data partitioning (rank 0):** The 128×128 k-space is split along axis=0 into `size` contiguous row blocks. With 4 ranks, each receives a 32×128 chunk.

2. **Scatter phase:** MPI scatter distributes one chunk to each rank. Post-scatter, the k-space is decomposed as:
   - Rank 0: rows 0–31
   - Rank 1: rows 32–63
   - Rank 2: rows 64–95
   - Rank 3: rows 96–127

3. **Stage 1—Parallel column transforms:** Each rank applies 1D IFFT along axis=1 (across columns, within its local rows). Since each row's column transform is independent of other rows, this stage exhibits perfect parallelism with **zero inter-rank communication**.

4. **Gather phase:** Partial results are collected back on rank 0 and reassembled with `vstack`.

5. **Stage 2—Sequential row transform:** Rank 0 completes the 2D IFFT by applying 1D IFFT along axis=0 (across rows, down columns). This stage is sequential because the row transform couples all rows—data from all ranks is required.

6. **Validation:** Two error checks confirm correctness:
   - `|recon - reference|` (parallel vs. sequential) $\approx$  $10^{-16}$ 
   - `|recon.real - phantom|` (reconstructed vs. ground truth) $\approx$  $10^{-16}$ 

**Performance characteristics:**

The 2D IFFT factors as:
```
IFFT2D(S) = IFFT_axis0(IFFT_axis1(S))
```

For an N×N grid:
- Stage 1 (axis=1): N × O(N log N) = O(N² log N) distributed across P ranks → O(N²/P log N) per rank
- Stage 2 (axis=0): N × O(N log N) = O(N² log N) sequential on rank 0

Work distribution:
- Each rank handles 1/P of stage 1 (ideal speedup).
- Stage 2 remains sequential (Amdahl bottleneck).
- Communication: 2 collective operations (scatter + gather), O(N²/P) data per rank.

For the 128×128 case with 4 ranks, approximately 75% of the FFT work is parallelized.

**Observed output:**
```
[Rank 1] Processed rows: 32
[Rank 2] Processed rows: 32
[Rank 3] Processed rows: 32
[Rank 0] Parallel MRI recon max error vs sequential: 5.204e-16
[Rank 0] Parallel MRI recon max error vs phantom: 5.204e-16
```

**Limitations:**
- Stage 2 is sequential (limits scalability for large P).
- All-to-all transpose would enable full parallelization but increases code complexity.
- For production 3D FFTs, libraries like `mpi4py-fft` employ pencil decompositions that parallelize all stages.

**Launch:** `mpiexec -n 4 python examples/07_mri_fft_parallel.py`

## Reconstruction Pipeline Summary

The complete MRI acquisition and reconstruction workflow:

**Common initial steps:**
- Scanner acquires raw signal from tissue
- K-space data populated (frequency domain): S(k_x, k_y) as complex 2D grid

**Sequential reconstruction path:**
- Apply IFFT2D on single node
- Output: I(x,y) reconstructed image

**Parallel reconstruction path:**
- Partition k-space by rows across MPI ranks
- Distribution: Rank 0, 1, 2, 3, etc. each holds a row subset
- Stage 1 (parallel):
  - Each rank applies 1D IFFT along axis=1 (column transforms)
  - No inter-rank communication required
- Communication phase:
  - Gather partial results to rank 0
- Stage 2 (sequential):
  - Rank 0 applies 1D IFFT along axis=0 (row transforms)
- Output: I(x,y) reconstructed image

## Clinical Impact and Performance

### Throughput and Latency

Parallel reconstruction directly improves clinical workflow metrics:

**Resolution scaling:**
- 256×256 clinical protocols: Reconstruction latency reduced from ~1 second (sequential) to ~200 ms (4-rank parallel).
- 512×512 research protocols: Wall-clock time drops from ~5 seconds to ~1 second with 4-8 ranks.
- 1024×1024 high-field imaging: Minutes → seconds, making high-resolution viable for routine use.

**Multi-coil parallelism:**
Modern phased-array systems employ 8–32 receiver coils. Each coil requires independent reconstruction followed by optimal combination (e.g., sum-of-squares or SENSE). Parallel FFT enables:
- Coil-level work distribution: Assign coils to rank groups.
- Concurrent reconstruction: All coils processed simultaneously.
- Aggregate speedup: 10–30× for 32-coil systems compared to sequential coil-by-coil processing.

### Real-Time Applications

**Interventional MRI (MR-guided surgery):**
- **Requirement:** Sub-200 ms latency for surgical navigation feedback.
- **Challenge:** Continuous acquisition + reconstruction + display pipeline must operate at 5–10 fps.
- **Solution:** Parallel FFT infrastructure + pipelined stages (while slice N reconstructs, slice N+1 acquires).

**Cardiac cine imaging:**
- **Requirement:** Reconstruct cardiac phases within one R-R interval (~50 ms @ 60 bpm).
- **Challenge:** Real-time imaging to visualize valve motion, wall dynamics during stress tests.
- **Solution:** Distribute temporal frames across ranks; streaming reconstruction keeps pace with acquisition.

### Streaming and Pipelining

Parallel infrastructure enables sophisticated pipelining strategies:
- **Slice-level concurrency:** While ranks 0–3 reconstruct slice A, ranks 4–7 handle slice B.
- **Temporal streaming:** Dynamic studies (perfusion, functional MRI) pipeline frames through reconstruction stages.
- **Overlapped I/O:** Reconstruction begins as soon as k-space lines arrive from the scanner, rather than waiting for complete acquisition.

## Scaling Beyond Row Decomposition

The two-stage scatter-gather approach demonstrated here is pedagogically clear but has limitations for large-scale production use:

**Bottlenecks:**
- Stage 2 (row-wise IFFT) remains sequential, limiting strong scalability.
- Rank 0 memory becomes a bottleneck when reassembling large datasets.
- Gather/scatter communication patterns are less efficient than all-to-all transposes for balanced workloads.

**Production alternatives:**
For 3D FFTs on 512³ grids or higher-dimensional problems (4D time series, 5D diffusion), production libraries employ:
- **Pencil decompositions:** Data partitioned along multiple axes, enabling parallel transforms in all dimensions.
- **All-to-all transposes:** MPI `Alltoall` operations redistribute data between transform stages with optimal communication patterns.
- **Hybrid parallelism:** Combine MPI (inter-node) with OpenMP/CUDA (intra-node) for hierarchical machines.

Libraries like `mpi4py-fft` automate these strategies. 
