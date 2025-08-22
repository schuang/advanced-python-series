# Workshop 3: Accelerating Your Code with GPUs

**Part of the series:** *From Scripts to Software: Practical Python for Reproducible Research*

Having learned how to write sustainable, object-oriented code and how to scale it across multiple processors, we will now explore how to dramatically accelerate your computations using Graphics Processing Units (GPUs).

## Part 1: The GPU Revolution

Modern GPUs are massively parallel processors that can perform millions of identical operations at once. This makes them incredibly powerful for a wide range of scientific workloads, from numerical simulations to deep learning. We will discuss:
*   The difference between CPU and GPU architectures.
*   The CUDA ecosystem, which is the foundation of GPU computing.
*   JAX, a modern Python library that provides a high-level, Pythonic interface to GPUs.

## Part 2: Thinking in JAX

JAX provides a NumPy-like API that will feel very familiar, but with the added ability to run on GPUs and TPUs. We will cover the basics of JAX, including its immutable array data structures.

*   **Introduction to JAX:** A hands-on introduction to the JAX API.
    *   See example: [01_jax_intro.py](workshop-3-examples/01_jax_intro.py)

## Part 3: The JAX "Superpowers"

JAX provides a set of powerful function transformations that are key to writing high-performance code.

*   **Just-In-Time (JIT) Compilation (`jax.jit`):** We will see how to use `jax.jit` to compile our Python functions into highly optimized machine code, leading to significant speedups.
    *   See example: [02_jax_jit.py](workshop-3-examples/02_jax_jit.py)

*   **Automatic Differentiation (`jax.grad`):** JAX can automatically compute the gradient of any Python function, which is a cornerstone of modern machine learning.
    *   See example: [03_jax_grad.py](workshop-3-examples/03_jax_grad.py)

*   **Automatic Vectorization (`jax.vmap`):** We will learn how to use `jax.vmap` to automatically vectorize our functions, allowing them to process batches of data efficiently.
    *   See example: [04_jax_vmap.py](workshop-3-examples/04_jax_vmap.py)

## Part 4: Applying JAX to the Golden Examples

Finally, we will apply these JAX concepts to our two "golden examples."

*   **Heat Equation:** We will rewrite the core computation of our heat equation solver in JAX and use `jax.jit` to accelerate the entire simulation.
    *   See example: [05_heat_equation_jax.py](workshop-3-examples/05_heat_equation_jax.py)

*   **Deep Learning:** We will rebuild our neural network from scratch in JAX and write a high-performance training loop using `jax.jit` and `jax.grad`.
    *   See example: [06_deep_learning_jax.py](workshop-3-examples/06_deep_learning_jax.py)
