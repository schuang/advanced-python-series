# JAX

JAX is a high-performance numerical computing library that combines NumPy's familiar API with automatic differentiation and just-in-time compilation. It's designed for machine learning research and high-performance scientific computing.


## Overview

JAX provides a powerful framework for numerical computing with automatic differentiation, making it ideal for optimization problems, machine learning, and scientific computing requiring gradients. Unlike traditional array libraries, JAX is built around composable function transformations that enable elegant solutions to complex problems.

**Key Characteristics:**

- **NumPy-Compatible API**: Familiar `jax.numpy` interface for array operations
- **Automatic Differentiation**: Built-in gradient computation with `grad`, `vjp`, `jvp`
- **JIT Compilation**: XLA compiler produces optimized code for CPU, GPU, TPU
- **Functional Programming**: Pure functions enable powerful transformations
- **Composable Transformations**: Combine `jit`, `grad`, `vmap`, `pmap` freely

**Why JAX for Scientific Computing:**

- Essential for gradient-based optimization and inverse problems
- Clean, composable code through functional programming
- High performance through XLA compilation
- Multi-device parallelization with minimal code
- Growing ecosystem for scientific computing and ML


## Brief History

JAX was developed by Google Research and emerged from the machine learning research community's need for flexible automatic differentiation.

**Key Milestones:**

- **2018**: JAX announced by Google Research, building on Autograd
- **2019**: Open-sourced, gained traction in ML research community
- **2020**: Added TPU support, improved XLA compilation
- **2021**: `pmap` for multi-device parallelization stabilized
- **2022**: Growing adoption in scientific computing beyond ML
- **2023**: Array API compatibility, improved documentation
- **Present**: Core library for ML research, expanding into scientific computing

JAX is maintained by Google Research and has become essential infrastructure for machine learning research, with growing adoption in physics, chemistry, and computational science.


## What JAX Does

**Core Functionality:**

- **Automatic Differentiation**: Compute gradients of arbitrary Python/NumPy functions
- **JIT Compilation**: XLA compiler optimizes and compiles functions to machine code
- **Vectorization**: Automatically map functions over batch dimensions
- **Parallelization**: Distribute computation across multiple GPUs/TPUs
- **Functional Transformations**: Composable tools that transform functions

**Function Transformations:**

- `grad()`: Compute gradients automatically
- `jit()`: Just-in-time compile for performance
- `vmap()`: Vectorize over batch dimensions
- `pmap()`: Parallelize across devices
- `vjp()` / `jvp()`: Vector-Jacobian and Jacobian-vector products

**XLA Compilation:**

- Uses Google's Accelerated Linear Algebra (XLA) compiler
- Fuses operations for efficiency
- Generates optimized code for CPU, GPU, TPU
- Significant performance improvements over NumPy

**Hardware Support:**

- CPU: Optimized execution via XLA
- NVIDIA GPUs: Full CUDA support
- Google TPUs: Native support
- AMD GPUs: Experimental ROCm support


## Key Use Cases in Scientific Computing

JAX has become essential in domains requiring gradients, optimization, and high-performance computing.

### 1. **Machine Learning Research**

Custom models, loss functions, and training loops benefit from automatic differentiation and JIT compilation.

**Example Use Cases:**

- Novel neural network architectures
- Custom optimization algorithms
- Reinforcement learning environments
- Probabilistic programming

### 2. **Optimization and Inverse Problems**

Gradient-based optimization is natural with JAX's automatic differentiation.

**Example Use Cases:**

- Parameter estimation in differential equations
- Inverse problems in physics
- Optimal control problems
- Variational methods

### 3. **Physics-Informed Neural Networks**

Combining physics equations with neural networks requires automatic differentiation.

**Example Use Cases:**

- Solving PDEs with neural networks
- Enforcing conservation laws in models
- Hybrid physics-ML simulations
- Scientific machine learning

### 4. **Computational Physics**

Simulations requiring gradients benefit from JAX's autodiff capabilities.

**Example Use Cases:**

- Molecular dynamics with learned potentials
- Quantum many-body systems
- Variational Monte Carlo
- Hamiltonian mechanics

### 5. **Bayesian Inference**

Probabilistic modeling and sampling algorithms leverage JAX's transformations.

**Example Use Cases:**

- Hamiltonian Monte Carlo (HMC)
- Variational inference
- Normalizing flows
- Uncertainty quantification

### 6. **Multi-Device Scientific Computing**

Large-scale simulations benefit from JAX's easy multi-GPU/TPU parallelization.

**Example Use Cases:**

- Distributed array computations
- Large-scale optimization
- Data-parallel training
- Multi-GPU simulations


## Functional Programming Model

JAX requires pure functions for its transformations to work correctly. This is a fundamental difference from NumPy.

### Pure Functions

**What is a Pure Function:**

- Output depends only on inputs (no hidden state)
- No side effects (no print, file I/O, global variables)
- Same inputs always produce same outputs
- Can be safely transformed and optimized

**Good (Pure Function):**

```python
def compute(x, y):
    return x ** 2 + y ** 2  # Pure: no side effects
```

**Bad (Impure Function):**

```python
counter = 0  # Global state

def compute(x, y):
    global counter
    counter += 1  # Side effect!
    print(f"Call {counter}")  # Side effect!
    return x ** 2 + y ** 2
```

**Why Purity Matters:**

- JAX traces functions once, replays later
- Side effects during tracing won't repeat during execution
- Transformations assume functions are pure
- Optimization requires predictable behavior

### Immutable Arrays

**JAX Arrays are Immutable:**

- Cannot modify arrays in-place
- Operations return new arrays
- Functional programming style

**NumPy (Mutable):**

```python
import numpy as np
x = np.array([1, 2, 3])
x[0] = 10  # In-place modification OK
```

**JAX (Immutable):**

```python
import jax.numpy as jnp
x = jnp.array([1, 2, 3])
# x[0] = 10  # Error! Cannot modify
x = x.at[0].set(10)  # Create new array
```

**Benefits:**

- Safe parallelization
- No race conditions
- Easier to reason about code
- Enables optimizations


## Core Transformations

JAX's power comes from composable function transformations.

### jit: Just-in-Time Compilation

**What it does:**

- Compiles function to optimized machine code
- Uses XLA compiler
- Significant speedup for numerical code
- Caches compiled functions

**Example:**

```python
import jax
import jax.numpy as jnp

def slow_function(x):
    return jnp.sum(x ** 2) + jnp.mean(x ** 3)

# Compiled version
fast_function = jax.jit(slow_function)

# Or use decorator
@jax.jit
def fast_function(x):
    return jnp.sum(x ** 2) + jnp.mean(x ** 3)

# First call: compilation overhead
x = jnp.arange(1000000)
result = fast_function(x)  # Compiles here

# Subsequent calls: fast
result = fast_function(x)  # Uses cached version
```

**Performance:**

- First call: compilation overhead (slower)
- Subsequent calls: 5-100x faster than uncompiled
- Best for functions called many times
- Most effective with complex computations

**When to Use:**

- Functions called repeatedly
- Numerical computations
- Inner loops in algorithms
- After verifying correctness (debug without jit first)

### grad: Automatic Differentiation

**What it does:**

- Computes gradients automatically
- Reverse-mode autodiff (backpropagation)
- Works with arbitrary Python control flow
- Essential for optimization

**Example:**

```python
import jax
import jax.numpy as jnp

# Define function
def loss(params):
    x, y = params
    return x ** 2 + 3 * x * y + 5 * y ** 2

# Compute gradient
grad_loss = jax.grad(loss)

# Evaluate gradient at point
params = jnp.array([1.0, 2.0])
gradient = grad_loss(params)
print(f"Gradient: {gradient}")  # [8.0, 23.0]
```

**With Multiple Arguments:**

```python
def loss(x, y, z):
    return x ** 2 + y * z

# Gradient w.r.t. first argument (default)
grad_x = jax.grad(loss, argnums=0)

# Gradient w.r.t. all arguments
grad_all = jax.grad(loss, argnums=(0, 1, 2))
```

**Combining with jit:**

```python
@jax.jit
@jax.grad
def fast_gradient(params):
    return loss(params)
```

**Use Cases:**

- Training neural networks
- Gradient-based optimization
- Inverse problems
- Sensitivity analysis

### vmap: Automatic Vectorization

**What it does:**

- Vectorizes function over batch dimension
- No explicit loops needed
- Efficient parallel execution
- Clean, readable code

**Example:**

```python
import jax
import jax.numpy as jnp

# Function for single input
def f(x):
    return x ** 2 + jnp.sin(x)

# Vectorize over batch
batched_f = jax.vmap(f)

# Apply to batch
batch = jnp.arange(10)
results = batched_f(batch)  # Processes all elements in parallel
```

**With Multiple Arguments:**

```python
def compute(x, y):
    return x * y + x ** 2

# Vectorize over first dimension of both arguments
batched_compute = jax.vmap(compute)

x_batch = jnp.array([1, 2, 3])
y_batch = jnp.array([4, 5, 6])
result = batched_compute(x_batch, y_batch)
```

**Specify Vectorization Axis:**

```python
# Vectorize over axis 1
batched_f = jax.vmap(f, in_axes=1, out_axes=1)
```

**Use Cases:**

- Processing batches of data
- Parallelizing over samples
- Monte Carlo simulations
- Parameter scans

### pmap: Multi-Device Parallelization

**What it does:**

- Distributes computation across multiple GPUs/TPUs
- Data parallelism with minimal code
- Automatic communication
- Scales to many devices

**Example:**

```python
import jax
import jax.numpy as jnp

# Function to run on each device
def compute(x):
    return jnp.sum(x ** 2)

# Parallelize across devices
parallel_compute = jax.pmap(compute)

# Data for each device (first dimension = number of devices)
n_devices = jax.device_count()
data = jnp.ones((n_devices, 1000))

# Compute on all devices in parallel
results = parallel_compute(data)
print(f"Results shape: {results.shape}")  # (n_devices,)
```

**Collective Operations:**

```python
@jax.pmap
def parallel_sum(x):
    # Sum across all devices
    return jax.lax.psum(x, axis_name='devices')
```

**Use Cases:**

- Multi-GPU training
- Distributed computing
- Large-scale simulations
- Data parallelism

### Composing Transformations

**Power of Composition:**

JAX transformations compose naturally:

```python
# Gradient of vectorized function
grad_batched = jax.grad(jax.vmap(f))

# JIT-compiled gradient
fast_grad = jax.jit(jax.grad(loss))

# Parallel gradient computation
parallel_grad = jax.pmap(jax.grad(loss))

# Gradient of JIT-compiled vectorized function
complex = jax.grad(jax.jit(jax.vmap(f)))
```


## Random Numbers

JAX uses explicit random state for reproducibility and parallelization.

**Why Explicit State:**

- Functional purity requires explicit state
- Enables parallelization without race conditions
- Reproducible across devices
- Follows functional programming principles

**Usage:**

```python
import jax
import jax.numpy as jnp

# Create random key (seed)
key = jax.random.PRNGKey(0)

# Split key for independent randomness
key, subkey = jax.random.split(key)

# Generate random numbers
random_array = jax.random.normal(subkey, shape=(1000,))

# Split again for next random operation
key, subkey = jax.random.split(key)
another_array = jax.random.uniform(subkey, shape=(100,))
```

**Multiple Splits:**

```python
# Split into many keys at once
key, *subkeys = jax.random.split(key, num=10)

# Use each subkey for independent randomness
arrays = [jax.random.normal(sk, (100,)) for sk in subkeys]
```

**Why Not Global State:**

- Global RNG state breaks functional purity
- Cannot safely parallelize
- Different devices would interfere
- Explicit keys ensure reproducibility


## Examples

### Example 1: Gradient-Based Optimization

Demonstrates JAX's core strength: automatic differentiation for optimization.

**Problem:**

Find parameters that minimize a loss function using gradient descent.

**Key Concepts:**

- Automatic differentiation with `grad`
- JIT compilation for speed
- Optimization loop
- Parameter updates

**Implementation:**

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Loss function: Rosenbrock function (banana function)
def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

# Gradient function
grad_rosenbrock = jax.jit(jax.grad(rosenbrock))

# Gradient descent optimization
def optimize(initial_params, learning_rate=0.001, n_steps=1000):
    params = initial_params
    history = [params]

    for step in range(n_steps):
        gradient = grad_rosenbrock(params)
        params = params - learning_rate * gradient
        history.append(params)

        if step % 100 == 0:
            loss = rosenbrock(params)
            print(f"Step {step}, Loss: {loss:.6f}")

    return params, jnp.array(history)

# Run optimization
initial = jnp.array([-1.0, 1.0])
final_params, history = optimize(initial)

print(f"Final parameters: {final_params}")
print(f"Final loss: {rosenbrock(final_params):.6f}")
print(f"Optimum at [1, 1]: {rosenbrock(jnp.array([1.0, 1.0])):.6f}")
```

**Key Points:**

- `jax.grad` computes gradient automatically
- No manual derivative calculation needed
- `jax.jit` makes gradient computation fast
- Works with arbitrary Python functions

**Performance:**

- Gradient computation: automatic and fast
- JIT compilation: 10-100x speedup
- Scales to high-dimensional problems

### Example 2: Physics-Informed Neural Network

Demonstrates JAX's power for scientific computing with automatic differentiation.

**Problem:**

Solve differential equation: dy/dx = -y, y(0) = 1 (solution: y = exp(-x))

**Key Concepts:**

- Neural network as function approximator
- Automatic differentiation for physics constraints
- Training to satisfy differential equation
- Combining ML and physics

**Implementation:**

```python
import jax
import jax.numpy as jnp

# Simple neural network
def neural_net(params, x):
    """Two-layer neural network"""
    w1, b1, w2, b2 = params
    hidden = jnp.tanh(w1 * x + b1)
    output = w2 * hidden + b2
    return output

# Physics loss: enforce dy/dx = -y
def physics_loss(params, x):
    """Loss based on differential equation"""
    # Function value
    y = neural_net(params, x)

    # Derivative dy/dx using autodiff
    dy_dx = jax.grad(lambda params: neural_net(params, x).sum())(params)

    # Physics constraint: dy/dx + y = 0
    residual = dy_dx[0] * x + y + y  # Simplified for this network

    return residual ** 2

# Initial condition loss: y(0) = 1
def initial_condition_loss(params):
    x0 = jnp.array(0.0)
    y0_pred = neural_net(params, x0)
    y0_true = 1.0
    return (y0_pred - y0_true) ** 2

# Total loss
def total_loss(params, x_points):
    phys_loss = jnp.mean(jax.vmap(lambda x: physics_loss(params, x))(x_points))
    ic_loss = initial_condition_loss(params)
    return phys_loss + ic_loss

# Training
@jax.jit
def update(params, x_points, learning_rate):
    loss = total_loss(params, x_points)
    grads = jax.grad(total_loss)(params, x_points)
    # Update parameters
    params = jax.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss

# Initialize network
key = jax.random.PRNGKey(0)
params = [
    jax.random.normal(key, ()) * 0.1,  # w1
    jax.random.normal(key, ()) * 0.1,  # b1
    jax.random.normal(key, ()) * 0.1,  # w2
    jax.random.normal(key, ()) * 0.1,  # b2
]

# Training points
x_points = jnp.linspace(0, 2, 20)

# Train
for step in range(1000):
    params, loss = update(params, x_points, learning_rate=0.01)
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss:.6f}")

# Evaluate
x_test = jnp.linspace(0, 2, 100)
y_pred = jax.vmap(lambda x: neural_net(params, x))(x_test)
y_true = jnp.exp(-x_test)

print(f"Mean error: {jnp.mean(jnp.abs(y_pred - y_true)):.6f}")
```

**Key Points:**

- `jax.grad` computes derivatives for physics constraints
- Neural network learned to satisfy differential equation
- No numerical solver needed
- Combines deep learning with physics

**Why JAX Excels Here:**

- Automatic differentiation through neural network
- Can compute higher-order derivatives
- Fast gradient computation with JIT
- Essential for physics-informed ML


## JAX vs NumPy: Key Differences

**Similarities:**

- Very similar API (`jax.numpy` mirrors `numpy`)
- Same array operations and functions
- Easy to port NumPy code to JAX

**Key Differences:**

| Feature | NumPy | JAX |
|---------|-------|-----|
| **Arrays** | Mutable | Immutable |
| **Random Numbers** | Global state | Explicit keys |
| **Performance** | CPU optimized | CPU/GPU/TPU via XLA |
| **Differentiation** | No | Automatic with `grad` |
| **Compilation** | Interpreted | JIT via XLA |
| **Vectorization** | Manual loops | `vmap` transformation |
| **Parallelization** | No | `pmap` for multi-device |

**Porting NumPy to JAX:**

```python
# NumPy code
import numpy as np
x = np.random.randn(1000)
y = np.sum(x ** 2)

# JAX equivalent
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1000,))
y = jnp.sum(x ** 2)
```


## Best Practices

### When to Use JAX

**Excellent for:**

- Gradient-based optimization
- Machine learning research
- Physics-informed neural networks
- Automatic differentiation needs
- Multi-GPU/TPU computing

**Not ideal for:**

- Simple NumPy operations (use CuPy)
- Code with many side effects
- Heavy I/O operations
- When you don't need gradients

### Development Workflow

**1. Start Without Transformations:**

```python
# First: write and test without jit/grad
def my_function(x):
    return x ** 2

# Test it works
result = my_function(jnp.array([1, 2, 3]))
```

**2. Add Transformations Gradually:**

```python
# Then: add jit for performance
@jax.jit
def my_function(x):
    return x ** 2
```

**3. Debug Without JIT:**

```python
# Remove @jax.jit for debugging
def my_function(x):
    print(f"x = {x}")  # Debug print
    return x ** 2
```

### Performance Tips

**Do:**

- Use JIT for functions called repeatedly
- Keep data on GPU across operations
- Use `vmap` instead of explicit loops
- Combine transformations for efficiency

**Don't:**

- JIT functions with side effects
- Transfer data between CPU/GPU unnecessarily
- Use Python loops when `vmap` works
- Compile functions called only once

### Common Pitfalls

**Pitfall 1: Mutable Operations**

```python
# Wrong: trying to mutate
x[0] = 5  # Error!

# Right: functional update
x = x.at[0].set(5)
```

**Pitfall 2: Side Effects in JIT**

```python
# Wrong: side effects in jitted function
@jax.jit
def bad(x):
    print(f"x = {x}")  # Only prints during tracing!
    return x ** 2

# Right: remove jit for debugging
def good(x):
    print(f"x = {x}")
    return x ** 2
```

**Pitfall 3: Global Random State**

```python
# Wrong: reusing same key
key = jax.random.PRNGKey(0)
x1 = jax.random.normal(key, (100,))
x2 = jax.random.normal(key, (100,))  # Same random numbers!

# Right: split keys
key = jax.random.PRNGKey(0)
key, subkey1 = jax.random.split(key)
key, subkey2 = jax.random.split(key)
x1 = jax.random.normal(subkey1, (100,))
x2 = jax.random.normal(subkey2, (100,))  # Different random numbers
```


## Installation

**Basic Installation:**

```bash
python -m venv .venv
source .venv/bin/activate

# CPU only
pip install jax

# CUDA 12.x (NVIDIA GPUs)
pip install jax[cuda12]

# CUDA 11.x (NVIDIA GPUs)
pip install jax[cuda11]
```

**Verify Installation:**

```python
import jax
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Check GPU support
try:
    import jax.numpy as jnp
    x = jnp.array([1, 2, 3])
    print(f"Array device: {x.device()}")
except:
    print("GPU not available")
```

**Hardware Requirements:**

- **CPU**: Any modern processor
- **NVIDIA GPU**: CUDA-capable (Compute Capability 3.5+)
- **TPU**: Google Cloud TPU (TPU v2, v3, v4)
- **AMD GPU**: Experimental ROCm support


## Summary

**JAX's Core Strengths:**

1. **Automatic Differentiation**: Essential for optimization and ML
2. **JIT Compilation**: XLA produces fast code for CPU/GPU/TPU
3. **Composable Transformations**: `grad`, `jit`, `vmap`, `pmap` work together
4. **Functional Programming**: Pure functions enable safe transformations
5. **NumPy-Compatible**: Easy to learn if you know NumPy

**When to Use JAX:**

- Need gradients for optimization
- Doing machine learning research
- Solving inverse problems
- Physics-informed neural networks
- Multi-GPU/TPU computing

**When to Use Other Frameworks:**

- **CuPy**: NumPy code without gradients
- **Numba**: Custom kernels, loop-heavy code
- **CUDA Python**: Low-level GPU control
- **PyTorch**: Standard deep learning

**Learning Path:**

1. Learn NumPy if you haven't already
2. Understand functional programming basics
3. Master `jit` and `grad` first
4. Then explore `vmap` and `pmap`
5. Apply to your domain (ML, physics, optimization)

JAX bridges the gap between high-level NumPy-style programming and high-performance computing with automatic differentiation, making it ideal for gradient-based scientific computing and machine learning research.
