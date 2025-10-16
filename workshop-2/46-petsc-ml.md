# Training a neural network with PETSc

## Background

- Hospitals, finance teams, and emergency services often need instant predictions (e.g., sepsis risk, fraud alerts) from large data feeds. Training models on millions of records requires splitting data across nodes while keeping model parameters synchronized, which is classic **data parallel** deep learning. 
- Frameworks such as PyTorch hide the MPI details. PETSc lets you build the same scalable pattern from first principles, useful for custom workflows or HPC integration.
- Training a neural network is a numerical optimization problem, so we will use TAO (for optimization) with PETSc.

**NOTE:** This example demonstrates how PETSc can be used for distributed machine learning workflows, illustrating the underlying MPI communication patterns that power modern frameworks. In production, you would typically use PyTorch, TensorFlow, or JAX, which provide optimized implementations and GPU support. This workshop example serves as a pedagogical tool to understand the data-parallel training pattern and how MPI enables it.

## Model and Objective

We train a logistic regression (single-layer neural net) with parameters $w \in \mathbb{R}^d$:
$$
\hat{y} = \sigma(w^\top x), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}.
$$
We minimize the average binary cross-entropy over samples $\{(x_i, y_i)\}_{i=1}^N$:
$$
L(w) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right].
$$
The gradient for each sample is $\nabla_w L_i = ( \hat{y}_i - y_i ) x_i$. The goal: each MPI rank owns a shard of the dataset, computes its local gradient, and all ranks **average gradients** to update the shared weight vector.



## TAO + Deeper Network (examples/14_petsc_deep.py)

### Neural Network as an Optimization Problem

Training a neural network is fundamentally an **unconstrained optimization problem**: given a loss function $L(\theta)$ parameterized by weights $\theta$, find $\theta^* = \arg\min_\theta L(\theta)$. This is exactly what PETSc's TAO (Toolkit for Advanced Optimization) is designed to solve. By framing deep learning as optimization, we can leverage sophisticated numerical methods (L-BFGS, Newton, trust-region) instead of just stochastic gradient descent.

### Architecture and Forward Pass

The example trains a three-layer binary classifier:

first hidden layer, 16 → 32
$$
h_1 = \tanh(x W_1 + b_1) 
$$

second hidden layer, 32 → 16)
$$
h_2 = \tanh(h_1 W_2 + b_2)
$$

output layer, 16 → 1
$$
\hat{y} = \sigma(h_2 W_3 + b_3)
$$

where $\sigma(z) = 1/(1 + e^{-z})$ is the sigmoid function. The loss is binary cross-entropy:
$$
L(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right].
$$

### Key Implementation Details

**1. Parameter Packing**

TAO requires a single flat vector for optimization. We pack all weights and biases into one NumPy array:
```python
theta = [W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel(), W3.ravel(), b3.ravel()]
```
The `unpack_params` function reverses this, reshaping the flat vector back into separate weight matrices for the forward pass. This is analogous to how PyTorch's optimizers work internally.

**2. Backpropagation (lines 89-126)**

The `forward_backward` function computes both the loss and its gradient using the chain rule:
- **Forward pass:** compute activations $a_1, a_2$ and predictions $\hat{y}$
- **Backward pass:** compute gradients using $\delta$ (error terms) propagated from output to input:
  - Output layer: $\delta_3 = \hat{y} - y$
  - Hidden layer 2: $\delta_2 = (\delta_3 W_3^T) \odot (1 - a_2^2)$ (tanh derivative)
  - Hidden layer 1: $\delta_1 = (\delta_2 W_2^T) \odot (1 - a_1^2)$

This is standard backpropagation, implemented in NumPy without automatic differentiation frameworks.

**3. Data-Parallel Training via MPI**

Each MPI rank owns a shard of the dataset (10,000 samples split across ranks). The `objective_gradient` callback:
1. **Local computation:** Each rank computes loss and gradients on its local data shard using mini-batches
2. **Global reduction:** `comm.allreduce` sums losses and `comm.Allreduce` sums gradients across all ranks
3. **Averaging:** Divide by total sample count to get the global gradient

This is the **data-parallel pattern**: replicate the model on all ranks, partition the data, and aggregate gradients. This is how PyTorch's `DistributedDataParallel` works under the hood.

**4. TAO Solver Setup**

```python
tao = PETSc.TAO().create(comm=PETSc.COMM_SELF)  # Sequential TAO on each rank
tao.setType(args.tao_type)  # L-BFGS-VM by default
tao.setObjectiveGradient(objective_gradient, grad)
tao.solve(w)
```

Key points:
- TAO is created with `COMM_SELF` (no TAO-level parallelism), but MPI parallelism happens inside `objective_gradient`
- Each rank maintains a full copy of weights (redundant storage), but data is partitioned
- L-BFGS-VM (`lmvm`) is a quasi-Newton method that approximates the Hessian, typically converging faster than vanilla gradient descent
- Alternative solvers: `tron` (trust-region Newton), `nls` (Newton line search), `cg` (conjugate gradient)

**5. Monitoring Convergence (lines 204-208)**

The monitor callback displays iteration number, loss, and gradient norm:
```python
its, f, res, cnorm, step, reason = tao.getSolutionStatus()
```
The gradient norm (`res`) should decrease as the optimizer converges. A norm below tolerance (e.g., 1e-5) indicates a local minimum.

### Why This Matters for HPC

This example demonstrates:
1. **MPI communication patterns:** `Allreduce` for gradient aggregation is the bottleneck in data-parallel training
2. **Optimization perspective:** Neural networks aren't magic—they're just high-dimensional nonlinear least squares
3. **Solver flexibility:** By using TAO, you can experiment with advanced optimizers (trust-region, line search) that deep learning frameworks rarely expose
4. **Scalability:** The same code scales to hundreds of ranks by just changing `mpirun -n <ranks>`

### Running at Scale

```bash
# Single node (debugging)
python examples/14_petsc_deep.py --samples 10000 --epochs 50

# Multi-node (production)
mpirun -n 16 python examples/14_petsc_deep.py --samples 200000 --features 64 --hidden1 256 --hidden2 128 --epochs 100

# Try different optimizers
python examples/14_petsc_deep.py --tao_type tron  # Trust-region Newton
python examples/14_petsc_deep.py --tao_type nls   # Newton line search
```

**Expected output:** Loss should decrease from ~0.69 (random initialization) to ~0.40-0.45, with training accuracy ~75-80% on the synthetic data. Convergence reason `-2` means maximum iterations reached; `-3` to `-8` indicate convergence to tolerance.

### Comparison to Production Frameworks

| Feature | This Example | PyTorch DDP |
|---------|--------------|-------------|
| Gradient computation | Manual backprop | Autograd |
| Parallelism | Explicit MPI `Allreduce` | Hidden in `DistributedDataParallel` |
| Optimizer | TAO (L-BFGS, trust-region) | SGD, Adam, etc. |
| Performance | Educational (CPU-only) | Production (GPU-optimized) |

This example strips away the abstractions to show *how* data-parallel training works at the MPI level—knowledge directly applicable to designing custom HPC workflows.
