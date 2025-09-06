# Python Classes: Inheritance

## The "Why": Building a Layered System
As a research project grows, the complexity can become overwhelming. A climate model isn't just one big equation; it's a layered system of interacting components: an atmospheric model, an ocean model, a sea ice model, and so on. A bioinformatics pipeline isn't a single script; it's a series of distinct tools for alignment, variant calling, and annotation.

A flat collection of dozens of functions and scripts is not a good way to represent these complex, layered systems. It's hard to see the relationships between the components, and it's difficult to reuse or replace a single part without breaking the whole.

**Inheritance** is the primary tool that object-oriented programming gives us to manage this complexity. It allows us to build a **hierarchy** of classes that mirrors the logical structure of our scientific problem. It lets us define a general, abstract concept at the top (e.g., a `ForceField`) and then create more specific, concrete implementations below (e.g., a `LennardJonesForceField` and a `CoulombForceField`).

This approach has two main goals:
1.  **To build a logical hierarchy:** It makes the structure of your code reflect the structure of your problem, which makes it easier to reason about.
2.  **To maximize code reuse:** It allows you to write the common, shared logic once in a general "parent" class and then reuse that logic in all of your more specific "child" classes.

This tutorial will cover the syntax and concepts for building these powerful, layered systems.

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

## The "What": Parent and Child Classes

Inheritance formalizes the relationship between your concepts.

*   You create a general **parent class** (also called a "base class" or "superclass") that contains all the common data and methods. In our case, this would be a `Particle` class.
*   You then create **child classes** (also called "derived classes" or "subclasses") that **inherit** all the attributes and methods from the parent. These child classes can then add their own unique attributes and methods, or modify the ones they inherited.

The most important concept is the **"is-a" relationship**. An `Ion` **is a** `Particle`. An `Atom` **is a** `Particle`. This means that anything you can do to a `Particle`, you can also do to an `Ion`.

## The "How": Syntax and Examples

Let's rebuild our example the right way.

### 1. Creating the Parent Class
First, we define the general `Particle` class with all the shared logic.

```python
import numpy as np

class Particle:
    """The parent class for all simulation particles."""
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)

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

## Abstract Classes: Defining a Contract (The Python Equivalent of C++ Virtual Classes)

For those unfamiliar with the C++ term, the core idea is about creating a "contract" or a "template" for other classes to follow. Sometimes you want to define a general parent class that should not be used on its own, but instead serves as a blueprint that guarantees all of its child classes will have a consistent structure and a specific set of methods. This is a cornerstone of building large, reliable software systems.

In C++, this is often done using "abstract base classes" with "pure virtual functions." Python provides the exact same functionality through its built-in `abc` (Abstract Base Class) module.

### The Concept: Defining a Contract
Imagine we are modeling different force fields. We know that every valid force field must be able to calculate the energy of a system. We want to enforce this rule in our code.

We can create an abstract `ForceField` class that has an "abstract method" called `calculate_energy`. This class acts as a contract. You cannot create an instance of `ForceField` itself. You can only create instances of child classes that fulfill the contract by providing their own concrete implementation of `calculate_energy`.

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

1.  **The Grandparent Class (Base Class)**
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

For those coming from other scientific computing languages, Python's approach to inheritance can feel simpler and more flexible, which is a significant advantage for rapid research and development.

*   **Compared to C++:** Python's inheritance is much less verbose.
    *   **No Header Files:** You define the class and its methods in a single `.py` file.
    *   **No `virtual` Keyword:** In Python, all methods are "virtual" by default. This means that when you call `my_object.my_method()`, Python will always run the version of the method from the most specific child class, which is almost always the behavior you want. You don't need to explicitly declare it.
    *   **Multiple Inheritance:** Python supports inheriting from multiple parent classes (`class C(A, B):`), which can be a powerful but complex feature. C++ also supports this, but Python's dynamic nature often makes it easier to manage.

*   **Compared to Julia:** Julia's design is fundamentally different.
    *   **Composition over Inheritance:** Julia's community and design strongly favor a pattern called "composition" over the "inheritance" we have discussed here. Instead of a `Dog` *being an* `Animal`, you would create a `Dog` struct that *has an* `Animal` struct inside it.
    *   **No Shared Behavior:** In Julia, you cannot inherit methods. The primary mechanism for sharing behavior is through multiple dispatch, where you define functions that operate on different data types.

**The Advantage for Scientific Prototyping:**
Python hits a sweet spot for scientific computing. Its inheritance model is powerful enough to build the logical hierarchies needed for complex models, but it avoids the rigid, boilerplate-heavy syntax of C++. This makes it faster to prototype, easier to read, and more straightforward to refactor as your scientific understanding of the problem evolves.

## The Payoff: Polymorphism in Scientific Computing

All the concepts in this tutorial—inheritance, method overriding, and abstract classes—build towards a single, powerful design principle: **polymorphism**. This is the concept that allows you to write flexible, high-level code that can operate on a wide variety of different objects in a uniform way.

The core idea is to allow a single function to accept objects of different classes, as long as they all "look" the same by adhering to a common interface.

### The Scientific Problem
Imagine you want to run the same simulation pipeline, but test three different force field models: `LennardJones`, `Coulomb`, and a new `AI_Potential`. Without polymorphism, you might be tempted to write an `if/elif/else` block:

```python
# The NON-polymorphic way (BAD PRACTICE)
def run_simulation(force_field, positions):
    if isinstance(force_field, LennardJones):
        energy = force_field.calculate_energy(positions)
    elif isinstance(force_field, Coulomb):
        energy = force_field.calculate_energy(positions)
    # ... you have to modify this function every time you add a new model!
```
This is not scalable and is hard to maintain.

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
This is the ultimate benefit of object-oriented design. Your high-level scientific workflow is now completely **decoupled** from the specific models you are using. You can develop and test new `ForceField` models without ever having to change a single line of your main `run_simulation` function. This makes your research more flexible, your code more robust, and your science more reproducible.

---

## Key Takeaways

*   **Inheritance** is the primary tool for building a logical **hierarchy** of classes and maximizing **code reuse**.
*   A **child class** inherits all methods and attributes from its **parent class**. Use `class Child(Parent):` to define this relationship.
*   Use `super().__init__(...)` in the child's constructor to ensure the parent class is properly initialized.
*   A child class can **override** a parent's method by defining a method with the same name. It can **extend** the parent's method by calling `super().method_name()` inside the new implementation.
*   **Abstract Base Classes** (using `ABC` and `@abstractmethod`) allow you to define a "contract" or an interface, forcing child classes to implement specific methods. This is the Python equivalent of C++ pure virtual functions.
*   **Polymorphism** is the payoff for using inheritance correctly. It allows you to write clean, high-level functions that can operate on any object that adheres to a common interface, making your code flexible and extensible.

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
