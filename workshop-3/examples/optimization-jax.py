#!/usr/bin/env python3
"""
JAX Gradient-Based Optimization Example
Demonstrates JAX's automatic differentiation and JIT compilation.

This example shows:
- Automatic differentiation with grad()
- JIT compilation for performance
- Gradient-based optimization (gradient descent)
- Functional programming with pure functions
- Visualization of optimization trajectory

We'll minimize the Rosenbrock function (banana function):
f(x, y) = (1 - x)² + 100(y - x²)²

The global minimum is at (1, 1) with f(1, 1) = 0.
"""

import jax
import jax.numpy as jnp
import time
import numpy as np


def rosenbrock(params):
    """
    Rosenbrock function (banana function).

    This is a classic test function for optimization algorithms.
    It has a narrow curved valley with the global minimum at (1, 1).

    Args:
        params: Array [x, y]

    Returns:
        Scalar loss value
    """
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def gradient_descent(loss_fn, initial_params, learning_rate=0.001, n_steps=1000, verbose=True):
    """
    Gradient descent optimization using JAX's automatic differentiation.

    Args:
        loss_fn: Loss function to minimize
        initial_params: Initial parameter values
        learning_rate: Step size for gradient descent
        n_steps: Number of optimization steps
        verbose: Whether to print progress

    Returns:
        final_params: Optimized parameters
        history: Array of parameter values at each step
    """
    # Create gradient function using JAX's automatic differentiation
    grad_fn = jax.jit(jax.grad(loss_fn))

    params = initial_params
    history = [params]

    for step in range(n_steps):
        # Compute gradient automatically
        gradient = grad_fn(params)

        # Update parameters
        params = params - learning_rate * gradient
        history.append(params)

        # Print progress
        if verbose and step % 100 == 0:
            loss = loss_fn(params)
            grad_norm = jnp.linalg.norm(gradient)
            print(f"Step {step:4d} | Loss: {loss:10.6f} | Grad norm: {grad_norm:10.6f} | Params: [{params[0]:7.4f}, {params[1]:7.4f}]")

    return params, jnp.array(history)


def main():
    print("=" * 70)
    print("JAX Gradient-Based Optimization Example")
    print("=" * 70)
    print("\nMinimizing the Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²")
    print("Global minimum at (1, 1) with f(1, 1) = 0\n")

    # ========================================================================
    # Demonstrate Automatic Differentiation
    # ========================================================================
    print("=" * 70)
    print("1. Automatic Differentiation")
    print("=" * 70)

    # Create gradient function
    grad_rosenbrock = jax.grad(rosenbrock)

    # Evaluate gradient at a point
    test_point = jnp.array([0.0, 0.0])
    gradient = grad_rosenbrock(test_point)
    loss_value = rosenbrock(test_point)

    print(f"At point {test_point}:")
    print(f"  Loss value: {loss_value}")
    print(f"  Gradient: {gradient}")
    print(f"  Gradient norm: {jnp.linalg.norm(gradient):.6f}")

    # Verify gradient is correct (compare with manual derivative)
    # Manual: ∂f/∂x = -2(1-x) - 400x(y-x²), ∂f/∂y = 200(y-x²)
    x, y = test_point
    manual_grad_x = -2 * (1 - x) - 400 * x * (y - x**2)
    manual_grad_y = 200 * (y - x**2)
    manual_gradient = jnp.array([manual_grad_x, manual_grad_y])

    print(f"  Manual gradient: {manual_gradient}")
    print(f"  Gradients match: {jnp.allclose(gradient, manual_gradient)}")

    # ========================================================================
    # JIT Compilation Benchmark
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. JIT Compilation Performance")
    print("=" * 70)

    # Without JIT
    def grad_no_jit(params):
        return jax.grad(rosenbrock)(params)

    # With JIT
    grad_jit = jax.jit(jax.grad(rosenbrock))

    test_params = jnp.array([0.5, 0.5])

    # Warm-up for JIT (compilation happens here)
    _ = grad_jit(test_params)

    # Benchmark
    n_iterations = 1000

    start = time.time()
    for _ in range(n_iterations):
        _ = grad_no_jit(test_params)
    no_jit_time = time.time() - start

    start = time.time()
    for _ in range(n_iterations):
        _ = grad_jit(test_params)
    jit_time = time.time() - start

    print(f"Without JIT: {no_jit_time:.4f}s ({n_iterations} iterations)")
    print(f"With JIT:    {jit_time:.4f}s ({n_iterations} iterations)")
    print(f"Speedup:     {no_jit_time/jit_time:.1f}x")

    # ========================================================================
    # Optimization from Different Starting Points
    # ========================================================================
    print("\n" + "=" * 70)
    print("3. Gradient Descent Optimization")
    print("=" * 70)

    # Starting point 1: Far from optimum
    print("\n" + "-" * 70)
    print("Starting Point 1: [-1.0, 1.0] (far from optimum)")
    print("-" * 70)

    initial_1 = jnp.array([-1.0, 1.0])
    final_1, history_1 = gradient_descent(
        rosenbrock,
        initial_1,
        learning_rate=0.001,
        n_steps=1000,
        verbose=True
    )

    print(f"\nOptimization complete!")
    print(f"  Initial point: {initial_1}")
    print(f"  Final point: {final_1}")
    print(f"  Initial loss: {rosenbrock(initial_1):.6f}")
    print(f"  Final loss: {rosenbrock(final_1):.6f}")
    print(f"  Distance from optimum: {jnp.linalg.norm(final_1 - jnp.array([1.0, 1.0])):.6f}")

    # Starting point 2: Different location
    print("\n" + "-" * 70)
    print("Starting Point 2: [2.0, 2.0] (different location)")
    print("-" * 70)

    initial_2 = jnp.array([2.0, 2.0])
    final_2, history_2 = gradient_descent(
        rosenbrock,
        initial_2,
        learning_rate=0.001,
        n_steps=1000,
        verbose=True
    )

    print(f"\nOptimization complete!")
    print(f"  Initial point: {initial_2}")
    print(f"  Final point: {final_2}")
    print(f"  Initial loss: {rosenbrock(initial_2):.6f}")
    print(f"  Final loss: {rosenbrock(final_2):.6f}")
    print(f"  Distance from optimum: {jnp.linalg.norm(final_2 - jnp.array([1.0, 1.0])):.6f}")

    # ========================================================================
    # Composing Transformations
    # ========================================================================
    print("\n" + "=" * 70)
    print("4. Composing JAX Transformations")
    print("=" * 70)

    # Combine grad + jit
    fast_grad = jax.jit(jax.grad(rosenbrock))
    print("✓ Created JIT-compiled gradient function: jax.jit(jax.grad(f))")

    # Vectorize gradient computation over batch
    batched_grad = jax.vmap(jax.grad(rosenbrock))
    print("✓ Created vectorized gradient function: jax.vmap(jax.grad(f))")

    # Test vectorized gradient
    batch_points = jnp.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [2.0, 2.0]
    ])

    batch_gradients = batched_grad(batch_points)
    print(f"\nComputed gradients for {len(batch_points)} points simultaneously:")
    for i, (point, grad) in enumerate(zip(batch_points, batch_gradients)):
        print(f"  Point {i}: {point} → Gradient: {grad} (norm: {jnp.linalg.norm(grad):.4f})")

    # ========================================================================
    # Functional Programming: Pure Functions
    # ========================================================================
    print("\n" + "=" * 70)
    print("5. Functional Programming with JAX")
    print("=" * 70)

    print("\nJAX requires pure functions for transformations to work correctly.")
    print("Pure functions: output depends only on inputs, no side effects.\n")

    # Good: Pure function
    print("✓ Good (Pure function):")
    print("  def compute(x, y):")
    print("      return x ** 2 + y ** 2")

    # Bad: Impure function (for demonstration only)
    print("\n✗ Bad (Impure function - don't do this!):")
    print("  counter = 0  # Global state")
    print("  def compute(x, y):")
    print("      global counter")
    print("      counter += 1  # Side effect!")
    print("      return x ** 2 + y ** 2")

    # ========================================================================
    # Array Immutability
    # ========================================================================
    print("\n" + "=" * 70)
    print("6. Immutable Arrays")
    print("=" * 70)

    print("\nJAX arrays are immutable - you cannot modify them in place.")
    print("This enables safe parallelization and optimization.\n")

    x = jnp.array([1, 2, 3, 4, 5])
    print(f"Original array: {x}")

    # Functional update (creates new array)
    x_new = x.at[0].set(10)
    print(f"After x.at[0].set(10):")
    print(f"  Original: {x}")
    print(f"  New:      {x_new}")

    # Multiple updates
    x_new2 = x.at[1:4].set(99)
    print(f"\nAfter x.at[1:4].set(99):")
    print(f"  Original: {x}")
    print(f"  New:      {x_new2}")

    # ========================================================================
    # Random Numbers with Explicit State
    # ========================================================================
    print("\n" + "=" * 70)
    print("7. Random Numbers with Explicit State")
    print("=" * 70)

    print("\nJAX uses explicit random keys for reproducibility.\n")

    # Create random key
    key = jax.random.PRNGKey(42)
    print(f"Initial key: {key}")

    # Split key for independent randomness
    key, subkey1 = jax.random.split(key)
    key, subkey2 = jax.random.split(key)

    # Generate random numbers
    random1 = jax.random.normal(subkey1, (3,))
    random2 = jax.random.normal(subkey2, (3,))

    print(f"Random array 1: {random1}")
    print(f"Random array 2: {random2}")
    print(f"Arrays are different: {not jnp.allclose(random1, random2)}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nKey JAX Features Demonstrated:")
    print("  1. Automatic differentiation with grad()")
    print("  2. JIT compilation for performance")
    print("  3. Gradient-based optimization")
    print("  4. Composable transformations (grad + jit + vmap)")
    print("  5. Functional programming with pure functions")
    print("  6. Immutable arrays")
    print("  7. Explicit random state")

    print("\nWhen to Use JAX:")
    print("  ✓ Gradient-based optimization")
    print("  ✓ Machine learning research")
    print("  ✓ Physics-informed neural networks")
    print("  ✓ Inverse problems")
    print("  ✓ Multi-GPU/TPU computing")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This example requires:")
        print("  - JAX installed: pip install jax")
        print("  - For GPU support: pip install jax[cuda12]")
        raise
