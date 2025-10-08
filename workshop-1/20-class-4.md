# Modern Python Classes: Decorators and Dataclasses

This section covers: 
- Further discussion of the decorators
- `@dataclass`: a decorator that can dramatically simplify your class definitions.

## Understanding Decorators

Throughout the previous tutorials, you have already been using decorators without a formal explanation. Every time you've written `@classmethod`, `@staticmethod`, or `@abstractmethod`, you were using a decorator.

Recall the `@` symbol:

- A decorator is a function that takes another function as input, adds some functionality to it, and returns the modified function. It's a way to "wrap" or "decorate" a function to give it extra powers without changing its core logic.

- It allows you to add reusable behaviors like logging, timing, or registration to many different functions in a clean, readable way, without cluttering up the logic of the functions themselves.

### Example: A `@timer` Decorator

In scientific computing, we often want to measure how long a specific function takes. A timer is a perfect use case for a decorator.

```python
import time

# 1. This is the decorator function.
#    It takes a function (`func`) as its input.
def timer(func):
    # 2. It defines a new "wrapper" function inside.
    #    *args and **kwargs are a standard way to accept any arguments.
    def wrapper(*args, **kwargs):
        # 3. Code to run BEFORE the original function.
        start_time = time.time()
        
        # 4. Call the original function and save its result.
        result = func(*args, **kwargs)
        
        # 5. Code to run AFTER the original function.
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed:.4f} seconds to run.")
        
        # 6. Return the original function's result.
        return result
    
    # 7. The decorator returns the newly defined wrapper function.
    return wrapper

# --- Now, let's use our decorator ---

@timer
def run_complex_calculation(n_points):
    """A placeholder for a long-running scientific task."""
    print(f"Running calculation with {n_points} points...")
    total = 0
    for i in range(n_points):
        total += i
    return total

# Now, when we call this function, it's actually the decorated version.
result = run_complex_calculation(10_000_000)
```
**Output:**
```
Running calculation with 10000000 points...
Function 'run_complex_calculation' took 0.3512 seconds to run.
```
The line `@timer` is just a clean, "Pythonic" shorthand for `run_complex_calculation = timer(run_complex_calculation)`. By understanding this simple pattern, you can see that decorators are not magic; they are just a standard way of wrapping one function with another.

This same pattern applies to the decorators you have already seen. `@classmethod` wraps your method to pass in the class (`cls`), `@staticmethod` wraps it to remove the `self` argument, and `@jit` from the Numba library wraps your function to send it to a just-in-time compiler. They are all functions that modify other functions.

## The Power of `@dataclass`

Now that you understand decorators, we can introduce one of the most useful decorators for scientific programming: `@dataclass`.

In `class-2.md`, we learned how to write a class manually, including the `__init__` method to store attributes. This involves a lot of boilerplate code.

**The "Before" Picture: A Manual Class**
```python
class ManualMolecule:
    def __init__(self, name, charge, num_atoms):
        self.name = name
        self.charge = charge
        self.num_atoms = num_atoms

    def __repr__(self):
        # We have to write our own representation method for printing.
        return f"ManualMolecule(name='{self.name}', charge={self.charge}, num_atoms={self.num_atoms})"

    def __eq__(self, other):
        # We have to write our own equality method for comparing.
        if not isinstance(other, ManualMolecule):
            return False
        return (self.name == other.name and
                self.charge == other.charge and
                self.num_atoms == other.num_atoms)
```
This is a lot of code just to create a simple data container.

**The "After" Picture: Using `@dataclass`**
The `@dataclass` decorator (from the built-in `dataclasses` module) can write all of that boilerplate for you automatically. All you have to do is declare the attributes using type hints.

```python
from dataclasses import dataclass

@dataclass
class Molecule:
    # Just declare the attributes and their types.
    # That's it!
    name: str
    charge: int
    num_atoms: int

# --- Let's see what we get for free ---

# 1. A proper __init__ method is automatically generated.
water = Molecule(name='Water', charge=0, num_atoms=3)

# 2. A beautiful __repr__ method is generated for easy debugging.
print(water)
# Output: Molecule(name='Water', charge=0, num_atoms=3)

# 3. A proper __eq__ method is generated for value-based comparison.
water2 = Molecule(name='Water', charge=0, num_atoms=3)
print(f"Are the two molecules equal? {water == water2}")
# Output: Are the two molecules equal? True
```

### Why `@dataclass` is Perfect for Scientific Computing
In research, we often work with simple objects whose primary purpose is to hold data (e.g., a simulation result, a set of experimental parameters, a record from a file). `@dataclass` is the perfect tool for this because:
1.  **It reduces boilerplate:** You can define a clean, readable data object in just a few lines of code.
2.  **It provides a useful `__repr__`:** The automatic representation makes your data objects easy to inspect and debug.
3.  **It makes testing easy:** The automatic `__eq__` method allows you to easily compare two data objects to see if they hold the same values, which is invaluable when writing tests.

By using `@dataclass`, you can write clearer, more concise, and more robust code, allowing you to focus on the science instead of on writing boilerplate methods.
