# Numba and Classes: Optimizing Object-Oriented Code

A common question for developers moving from simple functions to more complex, object-oriented code is: "How do I use Numba to accelerate the methods of my class?"

The short answer is: you generally **don't** put `@jit` on a method of a regular Python class and expect it to work well. Numba needs to know the data types of everything it's working with, and it can't understand the complex, dynamic nature of a standard Python `self` object.

Instead, Numba provides a special decorator called **`@jitclass`** to create high-performance, compiled classes.

### Why a simple `@jit` on a method doesn't work well

Let's say you have a simple class representing a particle in a simulation:

```python
# A standard Python class
class Particle:
    def __init__(self, x, y, velocity):
        self.x = x
        self.y = y
        self.velocity = velocity

    # We want to speed this method up
    def advance(self, dt):
        # This is a loop-like operation we want to accelerate
        self.x += self.velocity[0] * dt
        self.y += self.velocity[1] * dt
```

If you were to just add `@jit(nopython=True)` to the `advance` method, Numba would fail. When it sees `self`, it has no idea what it is. It doesn't know that `self.x` is a number or that `self.velocity` is an array. The standard Python `self` is too dynamic for Numba to analyze and compile.

### The Numba Solution: `@jitclass`

The correct way to handle this is to tell Numba about the *entire class structure* upfront. You do this with the `@jitclass` decorator, which requires you to define a "specification" for the data types of all the attributes.

Here is how you would write the `Particle` class the Numba way:

```python
import numpy as np
from numba import jitclass
from numba import float64, int32

# 1. Define the "spec": a list of all attributes and their Numba types
spec = [
    ('x', float64),
    ('y', float64),
    ('velocity', float64[:]),  # A 1D array of float64
]

@jitclass(spec)  # 2. Apply the decorator to the class
class NumbaParticle:
    def __init__(self, x, y, velocity):
        self.x = x
        self.y = y
        self.velocity = velocity

    # 3. The methods are automatically compiled. No @jit needed here.
    def advance(self, dt):
        # Numba now understands this operation because it knows the types
        # of self.x, self.velocity, and dt.
        self.x += self.velocity[0] * dt
        self.y += self.velocity[1] * dt
```

### How to Use It and The Performance Gain

Using the jitted class is exactly the same as using a regular class, but the performance of its methods will be dramatically faster, especially if you have many particles and are calling the method in a loop.

```python
# Create a list of 1 million particles
particles = [
    NumbaParticle(float(i), float(i), np.random.rand(2))
    for i in range(1_000_000)
]

dt = 0.01

def run_simulation(particles, dt):
    # This loop is now calling the fast, compiled `advance` method
    for p in particles:
        p.advance(dt)

# %timeit run_simulation(particles, dt)
# This will be orders of magnitude faster than if you had used the plain Python class.
```

### When to Use `@jitclass` (and When Not To)

`@jitclass` is a powerful, specialized tool. It is not a replacement for regular Python classes.

**Use `@jitclass` when:**
1.  You have a class that is at the **core of a major performance bottleneck**, usually involving loops.
2.  You need to create **very large numbers** of these objects (e.g., thousands or millions of particles, agents, or cells in a simulation).
3.  The class is primarily a **container for numerical data**, and its methods perform mathematical operations on that data.

**Be aware of the trade-offs:**
*   **Inflexibility:** A `@jitclass` object is "sealed." Once it's created, you **cannot add new attributes** to it on the fly (e.g., `my_particle.charge = 5` would raise an error if `charge` wasn't in the original spec). This is a major difference from standard Python classes.
*   **Compatibility:** Jitted objects can be more difficult to use with other Python libraries that expect standard Python objects (e.g., pickling them or passing them to a GUI framework might not work as expected).

**The Verdict:**
The best practice is often to use regular Python classes for the high-level structure of your program and to use `@jitclass` as a precision tool for the low-level, performance-critical objects that are at the heart of your numerical calculations.
