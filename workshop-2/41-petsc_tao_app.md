# Parameter Estimation using PETSc/Tao

## Problem Overview

This case study demonstrates how PETSc's **Toolkit for Advanced Optimization (TAO)** solves a canonical parameter estimation problem: **fitting exponential decay curves to noisy measurements**. While our motivating application is MRI tissue characterization ($T_2$ mapping), the mathematical framework applies broadly to:

- **Pharmacokinetics:** Drug concentration decay in blood plasma.
- **Nuclear physics:** Radioactive decay rate determination.
- **Chemical kinetics:** Reaction rate constant estimation.
- **Geophysics:** Electrical conductivity relaxation in subsurface materials.
- **Battery science:** Charge-discharge capacity fade modeling.

The computational challenge: **millions of independent small-scale nonlinear least-squares problems** that must be solved efficiently. TAO's parallel optimization infrastructure addresses this need.

## The Physical Process: MRI $T_2$ Relaxation

In magnetic resonance imaging, hydrogen nuclei (protons) in tissue are first **excited** by a radiofrequency pulse, causing their magnetic moments to precess coherently. After excitation stops, this coherence gradually **decays** due to local field inhomogeneities—a process called **transverse relaxation**.

### What is $T_2$?

**$T_2$** (pronounced "T-two") is a **tissue-specific time constant** that characterizes how quickly the MRI signal decays after excitation. Physically, it measures how long proton spins remain synchronized:

- **Large $T_2$ values** (100–300 ms): Spins stay coherent longer, producing sustained signal. This occurs in tissues with free water molecules (edema, CSF, cysts) where protons experience relatively uniform local magnetic environments.

- **Small $T_2$ values** (10–50 ms): Spins rapidly lose synchronization, causing fast signal decay. This occurs in structured tissues (bone, tendons, fibrotic regions) where protons are tightly bound and experience heterogeneous magnetic fields.

Mathematically, $T_2$ is the time constant in the exponential decay function: after time $T_2$, the signal intensity falls to approximately 37% (1/e) of its initial value. After 3×$T_2$, only ~5% of the original signal remains.

**Why $T_2$ varies across tissues:** The local magnetic environment surrounding each proton depends on molecular mobility, chemical composition, and tissue microstructure. Different tissues have characteristic $T_2$ values:

| Tissue Type | Typical $T_2$ (ms) | Interpretation |
|-------------|-----------------|----------------|
| Bone, calcification | 1–10 | Very fast decay (tightly bound protons) |
| Muscle | 40–60 | Moderate decay |
| Gray matter (brain) | 80–100 | Slower decay |
| Edema, inflammation | 150–300 | Very slow decay (excess free water) |

**Clinical value:** $T_2$ mapping converts qualitative MRI "brightness" into quantitative tissue properties, enabling objective diagnosis of cartilage degeneration, tumor characterization, and treatment response monitoring.

**Acquisition:** The scanner measures signal intensity at multiple **echo times** (time delays after excitation), producing a decay curve at each spatial location (voxel). Our task: **invert** these measurements to recover the underlying decay parameters.

## Mathematical Model

The observed MRI signal at a single voxel follows an exponential decay:

$$ 
S(t) = A \exp\left(-\frac{t}{T_2}\right) + \varepsilon(t)
$$

where:
- **S(t)**: Measured signal intensity at echo time t (observable).
- **A**: Initial signal amplitude, proportional to proton density (unknown parameter).
- **$T_2$**: Transverse relaxation time constant in milliseconds (unknown parameter, our primary target).
- **$\varepsilon$(t)**: Measurement noise from thermal fluctuations, system imperfections, patient motion.

Given a set of measurements $\{(t_i, S_i)\}_{i=1}^{N}$ acquired at echo times $t_1, t_2, \ldots, t_N$ (typically 8–15 samples), we seek the parameters $\theta = (A, T_2)$ that best explain the data.

## The Inverse Problem: Least-Squares Estimation

### Formulation

We cast parameter estimation as a **nonlinear least-squares problem**: find $\theta = (A, T_2)$ that minimizes the sum of squared residuals:

$$
f(A, T_2) = \frac{1}{2} \sum_{i=1}^{N} \left[ A \exp\left(-\frac{t_i}{T_2}\right) - S_i \right]^2
$$

**Intuition:** For any candidate parameter pair $(A, T_2)$, we:
1. **Predict** signal values at each echo time: $\hat{S}_i = A \exp(-t_i / T_2)$.
2. **Compute residuals**: $r_i = \hat{S}_i - S_i$ (prediction error).
3. **Penalize** via squared norm: $\|r\|^2 = \sum r_i^2$.

The optimizer iteratively adjusts $(A, T_2)$ to minimize this penalty, converging to the **maximum likelihood estimate** under Gaussian noise assumptions.

### Why This is Challenging

**Nonlinearity:** The exponential term couples A and $T_2$ in a nonlinear fashion. Unlike linear least squares (where parameters appear linearly), we cannot solve this in closed form via matrix inversion. The problem requires **iterative** optimization with gradient-based methods.

**Parameter scaling:** A and $T_2$ live on different scales (A  $\approx$  0.1–2.0, $T_2$  $\approx$  10–300 ms). Naive gradient descent often struggles without proper preconditioning or quasi-Newton methods.

**Scale:** A 256×256×128 MRI volume contains ~8 million voxels. Even with fast convergence (5–10 iterations per voxel), sequential processing is prohibitively slow. **Parallel execution** across voxels is essential.

### Gradient Computation

Gradient-based optimizers (L-BFGS, TRON, etc.) require the derivative of f with respect to each parameter:

$$
\frac{\partial f}{\partial A} = \sum_{i=1}^{N} r_i \exp\left(-\frac{t_i}{T_2}\right)
$$

$$
\frac{\partial f}{\partial T_2} = \sum_{i=1}^{N} r_i \cdot A \exp\left(-\frac{t_i}{T_2}\right) \cdot \frac{t_i}{T_2^2}
$$

where $r_i = A \exp(-t_i / T_2) - S_i$ is the residual at echo time $t_i$.

**Why provide gradients?** While TAO can approximate gradients via finite differences, **analytical gradients** are:
- **More accurate:** No truncation error from $\varepsilon$-perturbations.
- **Faster:** One function evaluation instead of 2N (for N parameters).
- **Numerically stable:** Avoids catastrophic cancellation in difference quotients.

## PETSc TAO Implementation

### Algorithm Selection

PETSc TAO provides multiple optimization algorithms. For our bounded, smooth, nonlinear least-squares problem, we use **LMVM** (Limited-Memory Variable Metric), a quasi-Newton method that:

- **Approximates the Hessian** using gradient history (L-BFGS approach).
- **Requires only first derivatives** (no need to code second derivatives).
- **Handles thousands of parameters efficiently** (though we have only 2 per voxel).
- **Supports bound constraints** (e.g., \(T_2 > 0\), \(A > 0\)).

Alternative TAO solvers include:
- `blmvm`: Bound-constrained L-BFGS (enforces \(T_2 \geq \epsilon\)).
- `tron`: Trust-region Newton (faster for well-conditioned problems).
- `nls`: Nonlinear least-squares specialized (explicitly exploits residual structure).

### Code Structure

The implementation (`examples/08_petsc_tao_t2_fit.py`) consists of three components:

**1. Objective and gradient callback (T2Objective class):**

```python
class T2Objective:
    """Least-squares objective for T2 decay fitting."""

    def __init__(self, times: np.ndarray, signal: np.ndarray):
        self.times = times      # Echo times [t1, t2, ..., tN]
        self.signal = signal    # Measured intensities [S1, S2, ..., SN]

    def __call__(self, tao: PETSc.TAO, x: PETSc.Vec, g: PETSc.Vec) -> float:
        # Extract parameters from distributed Vec
        vs, seq = PETSc.Scatter.toAll(x)
        vs.scatter(x, seq, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        params = seq.getArray(readonly=True).copy()

        amp, t2 = params
        t2 = max(t2, 1e-6)  # Numerical safeguard

        # Forward model: predict signal at each echo time
        exp_term = np.exp(-self.times / t2)
        pred = amp * exp_term
        residual = pred - self.signal

        # Objective: f = 0.5 * ||residual||^2
        f = 0.5 * np.dot(residual, residual)

        # Gradient: df/dA, df/dT2
        full_grad = np.empty_like(params)
        full_grad[0] = np.dot(residual, exp_term)
        full_grad[1] = np.dot(residual, amp * exp_term * self.times / (t2 * t2))

        # Write gradient into PETSc Vec
        start, end = x.getOwnershipRange()
        g.getArray()[:] = full_grad[start:end]

        return f
```

**Key design choice:** 

The callback computes **both** objective and gradient in one call, avoiding redundant exponential evaluations.

**2. Synthetic data generation:**

```python
# Echo times (ms) - typical clinical sequence
times_ms = np.array([10, 20, 40, 60, 80, 120, 160, 200, 240, 280])

# Ground truth parameters
true_amp = 1.0
true_t2 = 85.0  # Typical gray matter value

# Generate noisy measurements
signal = true_amp * np.exp(-times_ms / true_t2) + noise
```

**3. TAO solver setup and execution:**

```python
# Create distributed parameter vector (2 parameters)
x = PETSc.Vec().createMPI(2, comm=comm)
x.setValues(range(2), [0.8, 50.0])  # Initial guess
x.assemble()

# Configure TAO optimizer
tao = PETSc.TAO().create(comm=comm)
tao.setType("lmvm")                         # Limited-memory quasi-Newton
tao.setObjectiveGradient(objective, None)   # Register callback
tao.setFromOptions()                        # Allow runtime override
tao.solve(x)                                # Run optimization

# Extract solution
solution = retrieve_global_solution(x)
amp_est, t2_est = solution
```

### Runtime and Results

**Launch the example:**
```bash
mpiexec -n 2 python examples/08_petsc_tao_t2_fit.py
```

**Expected output:**

```
--- PETSc TAO MRI T2 Fit ---
True parameters: amplitude = 1.000, T2 = 85.0 ms
Estimated parameters: amplitude = 0.998, T2 = 84.6 ms
Final objective value: 1.234e-03
```

**Convergence analysis:**

- Typical iteration count: 8–12 iterations for convergence.
- Final objective ~10⁻³: consistent with noise variance (σ²  $\approx$  0.02² × 10 points).
- Parameter errors < 1%: demonstrates robust estimation despite 2% noise.

### Why PETSc + Python for This Problem?

**Performance:** PETSc's C/MPI core handles:

- Line search algorithms (backtracking, quadratic interpolation).
- Convergence monitoring (gradient norm, function decrease).
- Parallel vector operations (dot products, norms) via MPI collectives.

**Productivity:** Python enables:

- Concise problem specification (decay model in ~10 lines).
- NumPy integration for vectorized exponential/residual computation.
- Rapid prototyping (test with synthetic data before real MRI I/O).

**Scalability:** The same code scales from:

- **Development (1 rank):** Debug on synthetic 1-voxel problems.
- **Validation (4 ranks):** Test on small 64×64 regions.
- **Production (1000+ ranks):** Process 512×512×300 whole-brain volumes in parallel, distributing voxels across nodes.

## Workflow Summary

**Data acquisition and preprocessing:**

- MRI scanner acquires raw k-space data
- Reconstruct images using inverse FFT (see 36-mri.md)
- Extract signal decay curves at each voxel location

**Parallel parameter estimation (distributed across MPI ranks):**

- For each voxel independently:
  - Input data:
    - Echo times: t1, t2, ..., tN (typically 10 samples)
    - Measured signals: S1, S2, ..., SN
  - Initial parameter guess: A0 = 0.8, T2_0 = 50 ms
  - Optimization setup:
    - Define objective function: f(A, T2) = sum of squared residuals
    - Compute analytical gradient: grad_f(A, T2)
  - TAO LMVM solver execution:
    - Iteratively refine parameters (8-12 iterations)
    - Terminate when gradient norm < tolerance
  - Output: Converged parameters A*, T2* for this voxel

**Post-processing:**

- Gather results from all ranks
- Assemble complete T2 map (e.g., 256x256x128 volume)
- Clinical interpretation:
  - Identify tissue abnormalities (edema, tumors, degeneration)
  - Quantify treatment response over time
  - Generate diagnostic reports for radiologists

## Extending the Example

### 1. Add Bound Constraints

Real parameters have physical limits: A > 0, $T_2$ > 0. Enforce these with TAO's built-in bound constraints:

```python
# Set lower bounds (no upper bounds)
lower = PETSc.Vec().createMPI(2, comm=comm)
lower.setValues([0, 1], [1e-6, 1e-3])  # A <=10⁻⁶, T_2 <= 1 ms
lower.assemble()

tao.setVariableBounds(lower, None)
tao.setType("blmvm")  # Bounded L-BFGS
```

### 2. Multi-Exponential Models

Tissues with multiple compartments (intracellular/extracellular water) exhibit multi-exponential decay:

$$
S(t) = A_1 \exp(-t / T_{2,1}) + A_2 \exp(-t / T_{2,2}) + \varepsilon(t)
$$

Extend the parameter vector to 4D: \((A_1, T_{2,1}, A_2, T_{2,2})\), update the objective/gradient, and TAO handles the increased dimensionality automatically.

### 3. Spatial Regularization

Anatomical structures have smooth $T_2$ variations. Add a spatial penalty to couple neighboring voxels:

\[
f_{\text{reg}}(\theta) = \sum_{\text{voxels}} f_{\text{data}}(\theta_v) + \lambda \sum_{\text{edges}} \|\theta_u - \theta_v\|^2
\]

This requires switching to PETSc **SNES** (nonlinear solvers with PDE coupling) or TAO's composite formulations with distributed Hessian approximations.

### 4. Real MRI Data Integration

Replace synthetic data with actual scanner outputs:

```python
import nibabel as nib  # NIfTI reader

# Load 4D time-series (X × Y × Z × Nechoes)
img = nib.load('multi_echo_scan.nii.gz')
data = img.get_fdata()

# Distribute voxels across ranks
local_voxels = partition_voxels(data.shape[:3], comm)

# Fit each voxel's decay curve in parallel
for vox in local_voxels:
    signal = data[vox[0], vox[1], vox[2], :]
    objective = T2Objective(echo_times, signal)
    # ... TAO solve ...
```

With MPI-parallel I/O (HDF5, MPI-IO), ranks read/write their assigned voxels directly, eliminating root-node bottlenecks.

## Discussion Questions

- **Algorithmic choice:** When would you prefer `tron` (trust region) over `lmvm` (quasi-Newton)?
- **Jacobian sparsity:** For multi-voxel coupled problems, how would you exploit sparsity in the Hessian?
- **Uncertainty quantification:** Can TAO provide parameter confidence intervals (bootstrap, Hessian-based covariance)?
- **GPU acceleration:** PETSc supports CUDA/HIP backends—how would you adapt the callback for GPU execution?
