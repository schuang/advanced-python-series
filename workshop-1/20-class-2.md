# Python Classes: Syntax and Grammar

This guide covers fundamental syntax for creating and using classes in Python.

## The Core Concept: Blueprints and Instances

Two core concepts:

- Class: a blueprint defining structure and behavior
  - Example: `Molecule` class defines that all molecules have `name` and `coordinates`

- Object: a specific instance created from the blueprint
  - Example: `water` and `caffeine` objects created from `Molecule` class
  - Each holds its own unique data

### Why use classes?

Scripts with loose variables are error-prone. Classes provide:

- Single, coherent object (e.g., `water`) that knows its own data
- Call methods on the object (e.g., `water.calculate_center_of_mass()`)
- Bundle data and behavior together
- Safer, more organized, easier to debug and share

## The Basic Structure: `class` and `__init__`

Basic components:

- `class` keyword followed by class name
- Methods: functions that belong to a class
- `__init__`: the constructor
  - Called automatically when creating new instance
  - Sets up initial state by creating attributes

```python
class Molecule:
    # __init__ is the constructor method.
    # It runs when you create a new Molecule object.
    def __init__(self, name, charge, coordinates):
        print(f"Creating a new Molecule object for {name}...")
        
        # --- Attributes ---
        # These are variables that belong to the object.
        # They are created by assigning to `self`.
        self.name = name
        self.charge = charge
        self.coordinates = coordinates
```

## The `self` Keyword: The Most Important Concept

`self` is the first argument of any instance method.

Key points:

- Refers to the specific object (instance) the method is called on
- When you write `my_molecule.some_method()`, Python automatically passes `my_molecule` as `self`

Why is `self` needed?

- Methods are defined once on the class (blueprint)
- Multiple objects can exist (`water`, `caffeine`)
- When calling `water.calculate_center_of_mass()`, Python needs to know which object's data to use
- `self` makes that connection

Usage:

- Access object's own attributes and methods inside the method
- Example: `self.name = name` stores input `name` in this specific object's attribute

## Building a Class: A Step-by-Step Example

To understand the syntax, we will build up a single, coherent `Molecule` class from scratch.

### 1. The `__init__` Method and Instance Attributes

Goal: Create `Molecule` objects with unique `name`, `formula`, and `coordinates`.

- Instance attributes: unique to each object
- Class attribute: `_known_elements` set, shared across all `Molecule` objects

```python
import numpy as np

class Molecule:
    # This is a CLASS ATTRIBUTE, shared by all instances.
    _known_elements = set()

    def __init__(self, name, formula, coordinates):
        # --- INSTANCE ATTRIBUTES ---
        # These belong to the specific object being created (`self`).
        self.name = name
        self.formula = formula
        self.coordinates = np.array(coordinates)

        # Update the shared class data based on this instance's data
        elements_in_formula = ''.join(filter(str.isalpha, formula))
        for element in elements_in_formula:
            self.__class__._known_elements.add(element)
```

### 2. Instance Methods

Instance methods work with unique data of each object. All take `self` as first argument for access to object attributes.

```python
# Continuing the Molecule class...
    def calculate_center_of_mass(self):
        """Calculates the center of mass for this specific molecule."""
        return np.mean(self.coordinates, axis=0)

    def move(self, vector):
        """Moves this specific molecule by a given vector."""
        self.coordinates += vector
```

### 3. Class Methods

Class methods work with shared data across all instances. Take `cls` as first argument for access to class attributes. Can also serve as alternative constructors.

```python
# Continuing the Molecule class...
@classmethod
def get_all_known_elements(cls):
    """Returns the set of elements seen across all molecules."""
    return sorted(list(cls._known_elements))

@classmethod
def from_pdb_file(cls, filename):
    """Creates a Molecule instance by reading a PDB file."""
    # In a real function, you would parse the file.
    # Here, we'll just use placeholder data.
    print(f"Parsing {filename}...")
    parsed_name = "Caffeine"
    parsed_formula = "C8H10N4O2"
    parsed_coords = np.random.rand(24, 3) # Placeholder
    
    # The method calls the standard constructor `cls(...)` to create the object.
    return cls(parsed_name, parsed_formula, parsed_coords)
```

### 4. Static Methods

Static methods provide utility functions related to the class but don't need access to instance or class data.

```python
# Continuing the Molecule class...
@staticmethod
def get_atomic_mass(element_symbol):
    """A utility function to get atomic mass."""
    masses = {'H': 1.008, 'C': 12.011, 'O': 15.999, 'N': 14.007}
    return masses.get(element_symbol, 0.0)
```

Usage:
```python
# Call static method on the class itself
mass = Molecule.get_atomic_mass('C')  # Returns 12.011

# Can also call on an instance (but not recommended)
water = Molecule('Water', 'H2O', [[0,0,0]])
mass = water.get_atomic_mass('H')  # Returns 1.008
```

### The Complete Class
Here is the full, coherent `Molecule` class we have built. All the pieces work together.

```python
import numpy as np

class Molecule:
    # Class Attribute (shared)
    _known_elements = set()

    # Constructor and Instance Attributes
    def __init__(self, name, formula, coordinates):
        self.name = name
        self.formula = formula
        self.coordinates = np.array(coordinates)
        elements_in_formula = ''.join(filter(str.isalpha, formula))
        for element in elements_in_formula:
            self.__class__._known_elements.add(element)

    # Instance Methods
    def calculate_center_of_mass(self):
        return np.mean(self.coordinates, axis=0)

    def move(self, vector):
        self.coordinates += vector

    # Class Methods
    @classmethod
    def get_all_known_elements(cls):
        return sorted(list(cls._known_elements))

    @classmethod
    def from_pdb_file(cls, filename):
        print(f"Parsing {filename}...")
        parsed_name, parsed_formula, parsed_coords = "Caffeine", "C8H10N4O2", np.random.rand(24, 3)
        return cls(parsed_name, parsed_formula, parsed_coords)

    # Static Method
    @staticmethod
    def get_atomic_mass(element_symbol):
        masses = {'H': 1.008, 'C': 12.011, 'O': 15.999, 'N': 14.007}
        return masses.get(element_symbol, 0.0)
```

### Summary: Instance vs. Class vs. Static
The key is to distinguish between data that is unique to an instance versus data that is shared across the entire class.

| Method Type | Decorator | First Argument | Operates On... |
| :--- | :--- | :--- | :--- |
| **Instance Method** | (None) | `self` | Unique instance data (e.g., `self.coordinates`) |
| **Class Method** | `@classmethod` | `cls` | Shared class data (e.g., `cls._known_elements`) |
| **Static Method** | `@staticmethod`| (None) | Only its own inputs |

## Naming Conventions

Python doesn't strictly enforce naming rules, but strong community conventions improve readability.

- Class Names: `CamelCase` (first letter of each word capitalized)
  - Examples: `Molecule`, `SimulationAnalysis`

- Function and Method Names: `snake_case` (lowercase with underscores)
  - Examples: `calculate_center_of_mass`, `load_data`

- Variable and Attribute Names: `snake_case`
  - Examples: `self.coordinates`, `carbon_mass`

These conventions make it easy to distinguish classes, functions, and variables at a glance.

## Simulating Multiple Constructors (vs. C++)

In C++, you can define multiple constructors with different arguments (function overloading). Python doesn't allow this—a second `__init__` would overwrite the first.

Python offers two patterns to achieve the same goal.

### Pattern 1: Default Arguments in `__init__`

Single `__init__` with optional arguments (default to `None`). Use `if/elif` logic to handle different cases.

```python
class Molecule:
    def __init__(self, name, coordinates=None, filename=None):
        self.name = name
        if coordinates is not None:
            self.coordinates = np.array(coordinates)
        elif filename is not None:
            # In reality, you would parse the file here
            self.coordinates = np.random.rand(10, 3)
        else:
            raise ValueError("Must provide coordinates or a filename.")

# --- Usage ---
water = Molecule('Water', coordinates=[[0,0,0]])
caffeine = Molecule('Caffeine', filename='caffeine.pdb')
```

Pros: Simple for few variations  
Cons: `__init__` can become complex and messy

### Pattern 2: `@classmethod` Factories (Recommended)

Most powerful and Pythonic. Keep `__init__` simple and direct. Create separate, well-named class methods for each alternative constructor. Same "alternative constructor" pattern discussed earlier.

```python
class Molecule:
    # The main constructor is simple and direct
    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = np.array(coordinates)

    # This is an ALTERNATIVE constructor
    @classmethod
    def from_file(cls, name, filename):
        # 1. Perform special logic (e.g., read the file)
        coordinates_from_file = np.random.rand(10, 3) # Placeholder
        # 2. Call the main constructor `cls(...)` to create the final object
        return cls(name, coordinates_from_file)

# --- Usage ---
water = Molecule('Water', coordinates=[[0,0,0]])
caffeine = Molecule.from_file('Caffeine', filename='caffeine.pdb')
```

Pros: Clean, readable, scalable. `Molecule.from_file()` is self-documenting  
Cons: Requires slightly more lines of code

## Advanced Topic: Simulating Multiple Dispatch (vs. Julia)

While C++ uses function overloading, other scientific languages like Julia use a more powerful concept called **multiple dispatch**. This allows the system to choose which function to run based on the runtime types of *all* of its arguments. This is extremely useful for implementing different physical models for different types of interactions.

Python does not have this feature built-in, but you can achieve the same effect with an external library called `multipledispatch`.

### The Problem: Different Interactions
Imagine you have different particle types, and the `interact` function should change based on which types it receives.

```python
class Atom: pass
class Ion: pass

# interact(Atom(), Atom())   -> should be Lennard-Jones
# interact(Ion(), Ion())     -> should be Coulomb
# interact(Atom(), Ion())    -> should be Polarization
```

### The Solution: `multipledispatch`
First, install the library: `pip install multipledispatch`. Then, you can define your functions in a clean, declarative way.

```python
from multipledispatch import dispatch

class Atom: pass
class Ion: pass

@dispatch(Atom, Atom)
def interact(p1, p2):
    print("Calculating Lennard-Jones potential...")

@dispatch(Ion, Ion)
def interact(p1, p2):
    print("Calculating Coulomb potential...")

@dispatch(Atom, Ion)
def interact(p1, p2):
    print("Calculating polarization model...")

@dispatch(Ion, Atom)
def interact(p1, p2):
    # Handle commutative cases by calling the other version
    return interact(p2, p1)

# --- Usage ---
# The library dispatches to the correct function based on the types
interact(Atom(), Atom())
interact(Ion(), Ion())
interact(Atom(), Ion())
```
This pattern is far superior to a messy `if/isinstance` chain. It keeps your scientific logic clean, readable, and easily extensible.

## Key Takeaways

*   A **Class** is a blueprint for creating objects. An **Object** is a specific instance created from that blueprint, bundling its own unique data (attributes) and behaviors (methods).
*   The `__init__` method is the **constructor**, responsible for setting up an object's initial state by assigning values to `self`.
*   **Instance Methods** use `self` to operate on the unique data of a specific object.
*   **Class Methods** use `@classmethod` and `cls` to operate on shared data that belongs to the class blueprint itself.
*   **Static Methods** use `@staticmethod` for utility functions that are related to the class but don't depend on any instance or class state.
*   Python does not have C++-style multiple constructors, but the same effect is achieved more cleanly using **`@classmethod` factories**.

This syntax provides the complete toolkit for defining the behavior of your objects, from instance-specific actions to class-level operations and related utilities.

---

## Quick Quiz

Test your understanding of the key concepts.

**1. What is the primary difference between a class and an object (or instance)?**
<details>
  <summary>Answer</summary>
  A **class** is the blueprint or template. It defines the structure and behavior. An **object** is a specific, concrete instance created from that blueprint, holding its own unique data.
</details>

**2. You have a `Simulation` class. You want to add a method that calculates the kinetic energy, which depends on the unique velocities of the particles in that specific simulation. What kind of method should you write?**
<details>
  <summary>Answer</summary>
  You should write an **instance method**. Its first argument will be `self`, allowing it to access `self.velocities` to perform the calculation for that specific simulation instance.
</details>

**3. You want to add a method to your `Simulation` class that keeps a running count of how many total simulations have been created. What kind of method should you use to retrieve this count?**
<details>
  <summary>Answer</summary>
  You should use a **class method**. You would store the counter in a *class attribute* (e.g., `_simulation_count = 0`). The `__init__` method of each instance would increment this counter (`self.__class__._simulation_count += 1`). The class method (e.g., `get_total_count()`) would then access this shared counter via its `cls` argument (`return cls._simulation_count`).
</details>

**4. You want to add a function to your `Simulation` class that converts a temperature from Kelvin to Celsius. This calculation is logically related to simulations, but does not depend on any specific simulation's data or on the class itself. What is the most appropriate method type?**
<details>
  <summary>Answer</summary>
  A **static method** (using `@staticmethod`). It doesn't need `self` or `cls`, making it the perfect choice for a self-contained utility function that is namespaced within the class.
</details>

