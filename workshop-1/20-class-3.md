# Python Classes: Inheritance

## Building a Layered System

Research projects grow complex. Climate models have layered, interacting components: atmospheric, ocean, sea ice models. Bioinformatics pipelines chain distinct tools: alignment, variant calling, annotation.

Flat collections of functions and scripts don't represent these layered systems well. Hard to see relationships. Hard to reuse or replace parts without breaking everything.

Inheritance is the primary tool for managing this complexity. Build a hierarchy of classes that mirrors your scientific problem's logical structure. Define general, abstract concepts at the top (e.g., `ForceField`). Create specific, concrete implementations below (e.g., `LennardJonesForceField`, `CoulombForceField`).

Two main goals:

1. Build a logical hierarchy: code structure reflects problem structure
2. Maximize code reuse: write common logic once in parent class, reuse in all child classes

## The Problem: Code Duplication
Imagine you are writing a simulation. You start by creating a class for an `Atom`.

```python
class Atom:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)

    def advance_position(self, dt):
        self.position += self.velocity * dt
```

Now, you need to add `Ion`s to your simulation. An ion is just like an atom, but it also has a charge. Without inheritance, your first instinct might be to copy-paste:

```python
# The copy-paste approach (BAD PRACTICE)
class Ion:
    def __init__(self, mass, position, velocity, charge):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.charge = charge # The only new line!

    def advance_position(self, dt):
        # This method is identical to the one in Atom!
        self.position += self.velocity * dt
```

This is a recipe for disaster. You now have duplicated code. If you find a bug in `advance_position`, you have to remember to fix it in two places. This makes your code harder to maintain and less reproducible. Inheritance solves this problem by allowing you to define the common logic once and reuse it.

## Parent and Child Classes

Inheritance formalizes the relationship between your concepts.

- Parent class (base class, superclass): contains all common data and methods
  - Example: `Particle` class

- Child classes (derived classes, subclasses): inherit all attributes and methods from parent
  - Can add unique attributes and methods
  - Can modify inherited ones

Key concept: the "is-a" relationship. An `Ion` is a `Particle`. An `Atom` is a `Particle`. Anything you can do to a `Particle`, you can do to an `Ion`.

## Syntax and Examples

Let's rebuild our example the right way.

### 1. Creating the Parent Class
First, we define the general `Particle` class with all the shared logic.

```python
import numpy as np

class Particle:
    """The parent class for all simulation particles."""
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

    def advance_position(self, dt):
        """Advances the particle's position based on its velocity."""
        self.position += self.velocity * dt

    def calculate_kinetic_energy(self):
        """Calculates the kinetic energy."""
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)
```

### 2. Creating Child Classes
Now, we can create specialized child classes. The syntax is `class ChildClassName(ParentClassName):`.

```python
class Atom(Particle):
    """An Atom is a type of Particle. It inherits everything."""
    # We don't need to write anything here yet!
    # It automatically gets __init__, advance_position, etc. from Particle.
    pass

class Ion(Particle):
    """An Ion is a Particle that also has a charge."""
    def __init__(self, mass, position, velocity, charge):
        # --- Calling the Parent's Constructor ---
        # It's crucial to initialize the parent class first.
        # super() refers to the parent class (Particle).
        super().__init__(mass, position, velocity)
        
        # Now, add the new attribute that is unique to the Ion.
        self.charge = charge
```

The `super().__init__(...)` line is critical. It says, "run the `__init__` method of my parent class," which handles setting up the `mass`, `position`, and `velocity`. We then only need to add the one new line to handle the `charge`.

### 3. Using the Inherited Classes

```python
# Create an instance of the Atom child class
argon_atom = Atom(mass=39.9, position=[0,0,0], velocity=[1,0,0])

# Create an instance of the Ion child class
sodium_ion = Ion(mass=22.9, position=[5,5,5], velocity=[0,1,0], charge=1)

# We can call the method defined in the PARENT class on both objects
argon_atom.advance_position(0.1)
sodium_ion.advance_position(0.1)

print(f"Argon atom new position: {argon_atom.position}")
print(f"Sodium ion new position: {sodium_ion.position}")

# The sodium_ion object has the extra attribute
print(f"Sodium ion charge: {sodium_ion.charge}")
```

### 4. Overriding and Extending Methods

What if we want a child class to behave differently? We can **override** a parent's method by simply defining a method with the same name in the child class.

Let's add a `describe` method.

```python
class Particle:
    # ... (previous methods) ...
    def describe(self):
        return f"A particle with mass {self.mass:.2f}."

class Ion(Particle):
    # ... (__init__ from above) ...
    
    # This OVERRIDES the parent's describe method
    def describe(self):
        # But we can still call the parent's version first using super()
        # This is called EXTENDING the method.
        parent_description = super().describe()
        return f"{parent_description} And it has a charge of {self.charge}."

# --- Comparing the behavior ---
p = Particle(1.0, [0,0,0], [0,0,0])
ion = Ion(22.9, [5,5,5], [0,1,0], 1)

print(p.describe())
print(ion.describe())
```
**Output:**
```
A particle with mass 1.00.
A particle with mass 22.90. And it has a charge of 1.
```
The `Ion` class first calls the original `describe` method from the `Particle` class using `super().describe()` and then *extends* it by adding its own specific information.

By using inheritance, you create a logical and intuitive structure for your code that eliminates duplication and makes your scientific models easier to build, maintain, and share.

See the complete code: `src/atom.py`.

## Abstract Classes: Defining a Contract (The Python Equivalent of C++ Virtual Classes)

Core idea: create a "contract" or "template" for other classes to follow. Define a general parent class that should not be used on its own, but serves as a blueprint guaranteeing all child classes have consistent structure and specific methods. Cornerstone of building large, reliable software systems.

C++ does this with "abstract base classes" and "pure virtual functions." Python provides the same functionality through the built-in `abc` (Abstract Base Class) module.

### Defining a "Contract"

Modeling different force fields. Every valid force field must calculate the energy of a system. We want to enforce this rule in code.

Create an abstract `ForceField` class with an "abstract method" called `calculate_energy`. This class acts as a contract. Cannot create an instance of `ForceField` itself. Can only create instances of child classes that fulfill the contract by providing their own concrete implementation of `calculate_energy`.

### The Syntax: `ABC` and `@abstractmethod`

```python
from abc import ABC, abstractmethod

# 1. Inherit from ABC to mark this as an abstract base class.
class ForceField(ABC):
    """
    An abstract base class (a contract) for all force fields.
    It cannot be instantiated directly.
    """
    def __init__(self, parameters):
        self.parameters = parameters

    # 2. Use the @abstractmethod decorator.
    # This declares that any concrete child class MUST implement this method.
    @abstractmethod
    def calculate_energy(self, positions):
        """Calculates the potential energy of the system."""
        pass

# --- Now, we create concrete child classes that fulfill the contract ---

class LennardJones(ForceField):
    """A concrete implementation for the Lennard-Jones potential."""
    
    # We MUST implement this method, or Python will raise an error.
    def calculate_energy(self, positions):
        # In a real scenario, this would be a complex calculation.
        print("Calculating energy using the Lennard-Jones potential...")
        return np.sum(positions**2) # Placeholder

class Coulomb(ForceField):
    """A concrete implementation for the Coulomb potential."""
    
    # We also MUST implement this method.
    def calculate_energy(self, positions):
        print("Calculating energy using the Coulomb potential...")
        return np.sum(np.abs(positions)) # Placeholder
```

### Enforcing the Contract
The key feature is that Python will prevent you from creating an object that doesn't follow the rules.

```python
# This works perfectly. LennardJones fulfills the contract.
lj_potential = LennardJones(parameters={'sigma': 1.0, 'epsilon': 0.1})
lj_potential.calculate_energy(np.random.rand(10, 3))

# This will FAIL with a TypeError!
# You cannot create an instance of an abstract class.
try:
    invalid_ff = ForceField(parameters={})
except TypeError as e:
    print(f"\nError: {e}")
```
**Output:**
```
Calculating energy using the Lennard-Jones potential...

Error: Can't instantiate abstract class ForceField with abstract method calculate_energy
```
This is the same behavior you would get from a C++ class with a pure virtual function. The `abc` module is the "Pythonic" way to define a mandatory interface for a family of related classes, which is a cornerstone of robust, large-scale software design.

## Multi-Level Inheritance

Inheritance is not limited to a single parent-child level. You can build entire family trees of classes, where a child class becomes a parent to a grandchild class. This allows you to create highly specialized classes that inherit and combine features from their entire ancestry.

### A Simple Example: The Animal Kingdom

Let's model a simple hierarchy: `Animal` -> `Mammal` -> `Dog`.

1.  **The Base Class**

    This class is very general. It has a feature common to all animals: they have an age.
    ```python
    class Animal:
        def __init__(self, age):
            self.age = age
        
        def report_age(self):
            return f"I am {self.age} years old."
    ```

2.  **The Parent Class (Intermediate Class)**

    This class inherits from `Animal`. It adds a feature common to all mammals: they have fur.
    ```python
    class Mammal(Animal):
        def __init__(self, age, fur_color):
            # Initialize the parent class to set the age
            super().__init__(age)
            self.fur_color = fur_color
            
        def describe_fur(self):
            return f"I have {self.fur_color} fur."
    ```

3.  **The Child Class (Concrete Class)**
    This class inherits from `Mammal`. It adds a feature unique to dogs: they can bark.
    ```python
    class Dog(Mammal):
        def __init__(self, age, fur_color, breed):
            # Initialize the parent class (Mammal)
            # The Mammal's __init__ will in turn initialize the Animal class.
            super().__init__(age, fur_color)
            self.breed = breed
            
        def bark(self):
            return "Woof!"
    ```

### Using the Grandchild Class
An instance of the `Dog` class now has access to the methods and attributes from its entire inheritance chain: `Dog`, `Mammal`, and `Animal`.

```python
my_dog = Dog(age=5, fur_color='brown', breed='Golden Retriever')

# Method from the Dog class
print(my_dog.bark())

# Method from the Mammal class
print(my_dog.describe_fur())

# Method from the Animal class
print(my_dog.report_age())

# Attributes from all levels
print(f"Breed: {my_dog.breed}, Fur: {my_dog.fur_color}, Age: {my_dog.age}")
```
**Output:**
```
Woof!
I have brown fur.
I am 5 years old.
Breed: Golden Retriever, Fur: brown, Age: 5
```
This layered approach is extremely powerful for organizing complex scientific models. You can define general physical laws in a base class, add more specific behaviors in intermediate classes, and create highly specialized, concrete classes for your final simulations, all while maximizing code reuse and maintaining a clear, logical structure.

## Python's Inheritance vs. C++ and Julia

Coming from other scientific computing languages? Python's inheritance is simpler and more flexible—significant advantage for rapid research and development.

- Compared to C++: Python's inheritance is much less verbose
  - No Header Files: define class and methods in single `.py` file
  - No `virtual` Keyword: all methods are "virtual" by default. When you call `my_object.my_method()`, Python runs the version from the most specific child class. No explicit declaration needed
  - Multiple Inheritance: Python supports inheriting from multiple parents (`class C(A, B):`). Powerful but complex. C++ supports this too, but Python's dynamic nature makes it easier to manage

- Compared to Julia: Julia's design is fundamentally different
  - Composition over Inheritance: Julia strongly favors "composition" over inheritance. Instead of `Dog` being an `Animal`, create a `Dog` struct that has an `Animal` struct inside it
  - No Shared Behavior: cannot inherit methods in Julia (can inherit types). Primary mechanism for sharing behavior is multiple dispatch—define functions that operate on different data types

In Julia, this would look like:

```julia
# You can define a type hierarchy
abstract type Animal end

struct Dog <: Animal  # Dog is a subtype of Animal
    name::String
end

struct Cat <: Animal
    name::String
end

# methods are defined OUTSIDE the types
function speak(a::Animal)
    println("Some generic animal sound")
end

function speak(d::Dog)
    println("$(d.name) says Woof!")
end

function speak(c::Cat)
    println("$(c.name) says Meow!")
end
```


**The Advantage for Scientific Prototyping:**

- Python hits a sweet spot for scientific computing
- Inheritance model is powerful enough to build logical hierarchies needed for complex models
- Avoids the rigid, boilerplate-heavy syntax of C++
- Faster to prototype, easier to read, and more straightforward to refactor
- Flexibility is crucial as your scientific understanding of the problem evolves


## Polymorphism in Scientific Computing

All the concepts about inheritance, method overriding, and abstract classes build towards a powerful design principle: polymorphism. Write flexible, high-level code that operates on a wide variety of different objects in a uniform way.

Core idea: single function accepts objects of different classes, as long as they all "look" the same by adhering to a common interface.

### The Problem

Run the same simulation pipeline, but test three different force field models: `LennardJones`, `Coulomb`, and a new `AI_Potential`. Without polymorphism, you might write an `if/elif/else` block:

```python
# The NON-polymorphic way (BAD PRACTICE)
def run_simulation(force_field, positions):
    if isinstance(force_field, LennardJones):
        energy = force_field.calculate_energy(positions)
    elif isinstance(force_field, Coulomb):
        energy = force_field.calculate_energy(positions)
    # ... you have to modify this function every time you add a new model!
```
This is not scalable and is hard to maintain because:

- You must explicitly check the type with `isinstance()`

- You must modify this function every time you add a new force field class!

- The function is tightly coupled to specific class names


### The Polymorphic Solution

By using the `ForceField` abstract base class we defined earlier, we create a contract that all force fields must follow. This allows us to write a single, clean, high-level function that doesn't need to know the specific details of the model it's working with.

```python
# This is a POLYMORPHIC function.
# It can accept any object that fulfills the ForceField contract.
def run_simulation(force_field: ForceField, positions):
    """Runs a simulation step using a given force field."""
    print(f"\n--- Running simulation with {force_field.__class__.__name__} ---")
    # It doesn't know the specific type, it just trusts the contract.
    energy = force_field.calculate_energy(positions)
    print(f"Calculated Energy: {energy:.2f}")

# --- Now we can pass in different types of objects ---
lj = LennardJones(parameters={})
coulomb = Coulomb(parameters={})
# Imagine we wrote a third class, AI_Potential, that also inherits from ForceField

run_simulation(lj, np.random.rand(10,3))
run_simulation(coulomb, np.random.rand(10,3))
# run_simulation(AI_Potential(model_file='...'), positions)
```

So polymorphism in Python is more about design philosophy (trusting interfaces) than syntax. The syntax difference is simply: with or without `isinstance()` checks!


This is the ultimate benefit of object-oriented design. Your high-level scientific workflow is now completely **decoupled** from the specific models you are using. You can develop and test new `ForceField` models without ever having to change a single line of your main `run_simulation` function. This makes your research more flexible, your code more robust, and your science more reproducible.

---

## Key Takeaways

- Inheritance: primary tool for building logical hierarchy of classes and maximizing code reuse

- Child class inherits all methods and attributes from parent class. Use `class Child(Parent):` to define this relationship

- Use `super().__init__(...)` in child's constructor to ensure parent class is properly initialized

- Child class can override parent's method by defining method with same name. Can extend parent's method by calling `super().method_name()` inside new implementation

- Abstract Base Classes (using `ABC` and `@abstractmethod`) define a "contract" or interface, forcing child classes to implement specific methods. Python equivalent of C++ pure virtual functions

- Polymorphism: the payoff for using inheritance correctly. Write clean, high-level functions that operate on any object adhering to a common interface, making code flexible and extensible

---

## Quick Quiz

**1. What are the two primary benefits of using inheritance?**
<details>
  <summary>Answer</summary>
  1.  **Code Reuse:** It allows you to write common logic once in a parent class and reuse it in multiple child classes, avoiding code duplication.
  2.  **Logical Hierarchy:** It allows you to structure your code in a way that reflects the natural, hierarchical relationships of the concepts in your scientific model.
</details>

**2. You have a parent class `Experiment` and a child class `CalorimetryExperiment`. Both have an `__init__` method. What is the command you must use in the child's `__init__` method to ensure the parent's `__init__` method is also run?**
<details>
  <summary>Answer</summary>
  You must call `super().__init__(...)`, passing along any arguments the parent's constructor needs. This ensures that the setup logic in the parent class is executed before the child class adds its own specific attributes.
</details>

**3. You are writing a high-level function that needs to work with different types of scientific data readers (e.g., `PDBReader`, `CSVReader`, `HDF5Reader`). You want to guarantee that every reader object has a `.read_data()` method. What is the best way to enforce this contract?**
<details>
    <summary>Answer</summary>
    The best way is to create an **abstract base class** called `DataReader` that inherits from `ABC`. Inside this class, you would define an abstract method:
    ```python
    from abc import ABC, abstractmethod

    class DataReader(ABC):
            @abstractmethod
            def read_data(self):
                    pass
    ```
    You would then make sure that `PDBReader`, `CSVReader`, and `HDF5Reader` all inherit from `DataReader` and provide their own concrete implementations of the `read_data` method.
</details>

**4. What is polymorphism?**
<details>
  <summary>Answer</summary>
  Polymorphism is the ability to write a single, high-level function that can operate on objects of different classes. This is possible as long as those different objects all share a common interface (e.g., they all inherit from the same base class or have methods with the same names). It allows you to decouple your main workflow from the specific implementations of your models, making your code more flexible and extensible.
</details>
