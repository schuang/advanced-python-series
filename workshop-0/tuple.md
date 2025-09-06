# Tuples and Namedtuples

As scientists and researchers, we constantly work with data. Often, this data comes in small, fixed groups: a 3D coordinate (x, y, z), a measurement with its error (value, error), or the parameters of a model (alpha, beta, gamma). A common way to handle these in Python is using a `list`.

```python
# A list representing a 3D point
my_point = [1.0, 2.5, -3.3]
```

This works, but it has a hidden danger: lists are **mutable**. You can change them at any time.

```python
# Somewhere else in the code, we accidentally modify the point
my_point[1] = 9.9 # The original y-value is lost forever
```

This kind of bug can be incredibly hard to track down in a complex analysis script. You pass your data to a function, and it gets changed without you realizing it. This is where tuples come in.

## The Power of Immutability: `tuple`

A tuple is like a list that cannot be changed after it's created. Think of it as a "read-only" list. You create it with parentheses `()` instead of square brackets `[]`.

```python
# A tuple representing a 3D point
my_point = (1.0, 2.5, -3.3)

# You can access elements just like a list
print(f"X-coordinate: {my_point[0]}")

# But you CANNOT change them. This line will raise an error:
# my_point[1] = 9.9  # TypeError: 'tuple' object does not support item assignment
```

This "immutability" is a powerful feature for scientific programming. When you use a tuple for a fixed collection of data, you are giving a guarantee to anyone (including your future self) that this data structure will not be accidentally modified. It makes your code safer and easier to reason about.

### Tuple Unpacking

Another handy feature is "unpacking," which allows you to assign the elements of a tuple to individual variables in one go.

```python
# Tuple packing
point_3d = (5.2, 8.1, 2.0)

# Tuple unpacking
x, y, z = point_3d

print(f"The coordinates are x={x}, y={y}, z={z}")
```

This is very common when a function needs to return multiple values.

```python
import numpy as np

def calculate_stats(data):
    """Returns the mean and standard deviation of the data."""
    mean = np.mean(data)
    std_dev = np.std(data)
    return (mean, std_dev) # Return a tuple

# Unpack the results directly into variables
mean_val, std_val = calculate_stats([1, 2, 3, 4, 5])
```

## The Readability Problem: What does `[0]` mean?

Tuples are great for safety, but they can sometimes make code hard to read.

```python
# A tuple representing a particle measurement
# (particle_id, energy_keV, x_pos, y_pos)
particle = (101, 4.57, 12.3, -4.8)

# What is particle[1]? What is particle[2]?
# You have to remember the order.
if particle[1] > 4.5:
    print(f"High energy event from particle {particle[0]}")
```

This use of "magic numbers" (like `1` for energy) makes the code cryptic and error-prone. What if you later add a `z_pos` and the indices all shift?

## The Solution: `collections.namedtuple`

A `namedtuple` is the perfect solution. It gives you the best of both worlds: the immutability of a tuple and the readability of accessing fields by name.

First, you import it from the `collections` module. Then, you use it to create a custom "template" or "factory" for your data structure.

```python
from collections import namedtuple

# 1. Create the template for our particle data
#    Arguments: "TypeName", "field_names" (space-separated string)
Particle = namedtuple("Particle", "id energy x y")

# 2. Create an instance of our new Particle type
p1 = Particle(id=101, energy=4.57, x=12.3, y=-4.8)

# Now, you can access data by name!
print(f"Particle ID: {p1.id}")
print(f"Energy (keV): {p1.energy}")

# The code is now self-documenting and much clearer
if p1.energy > 4.5:
    print(f"High energy event from particle {p1.id}")

# You still get the benefits of a regular tuple:
# It's immutable
# p1.energy = 5.0 # This will raise an AttributeError

# You can still access by index if you need to
print(f"The y-coordinate is {p1[3]}")
```

### When to Use Them

*   **Use a `tuple`** when you have a short, simple sequence of items where the order is obvious and unlikely to change. Returning a pair of values like `(mean, std_dev)` is a perfect use case.
*   **Use a `namedtuple`** when you have a data structure with a few fixed fields, like a record from a file or a set of related parameters. It acts as a lightweight, immutable "mini-class," making your code significantly more readable and maintainable.

By using tuples and namedtuples appropriately, you can make your scientific code safer, clearer, and less prone to bugs, allowing you to focus more on the science and less on debugging.
