# Modern Python Classes: Dataclasses

This section covers `@dataclass`: a decorator that can simplify your class definitions.



## The Power of `@dataclass`

Now that you understand decorators, we can introduce one of the most useful decorators for scientific programming: `@dataclass`.

We have learned how to write a class manually, including the `__init__` method to store attributes. This involves a lot of boilerplate code.

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

In research, we often work with simple objects whose primary purpose is to hold data (e.g., simulation result, experimental parameters, record from a file). `@dataclass` is the perfect tool for this:

1. Reduces boilerplate: define clean, readable data object in just a few lines
2. Provides useful `__repr__`: automatic representation makes data objects easy to inspect and debug
3. Makes testing easy: automatic `__eq__` method allows easy comparison of two data objects to see if they hold same values—invaluable when writing tests

Using `@dataclass` lets you write clearer, more concise, and more robust code. Focus on the science instead of writing boilerplate methods.
