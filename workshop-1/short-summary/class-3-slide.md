# Python Classes: Inheritance

- **Why Inheritance?**
  - Organizes complex, layered systems (e.g., climate models, bioinformatics pipelines)
  - Builds logical hierarchies that mirror scientific problems
  - Maximizes code reuse; avoids duplication

- **Problem Without Inheritance**
  - Copy-paste leads to duplicated code and maintenance issues
  - Bugs must be fixed in multiple places

- **Parent & Child Classes**
  - Parent/base class: defines shared data & methods
  - Child/subclass: inherits from parent, adds/overrides features
  - "is-a" relationship: e.g., `Ion` is a `Particle`

- **Syntax**
  - `class Child(Parent):` — child inherits all parent methods/attributes
  - Use `super().__init__(...)` in child to initialize parent

- **Method Overriding & Extending**
  - Child can override parent methods
  - Use `super().method()` to extend parent behavior

- **Abstract Base Classes (ABC)**
  - Define a contract/interface for child classes
  - Use `from abc import ABC, abstractmethod`
  - Abstract methods must be implemented by child classes
  - Prevents instantiation of incomplete base classes

- **Multi-Level Inheritance**
  - Classes can inherit in chains (e.g., `Animal` → `Mammal` → `Dog`)
  - Grandchild class inherits all features from ancestors

- **Python vs. C++/Julia**
  - Python: simple, flexible, all methods are "virtual" by default
  - C++: more boilerplate, explicit virtual methods
  - Julia: favors composition over inheritance

- **Polymorphism**
  - Write high-level functions that work with any class following a common interface
  - Decouples workflow from specific implementations
  - Enables flexible, extensible scientific code

- **Key Takeaways**
  - Inheritance = hierarchy + code reuse
  - Use `super()` for parent initialization and method extension
  - Abstract base classes enforce consistent interfaces
  - Polymorphism enables flexible, maintainable workflows

---

## Quick Quiz (for discussion)
- What are the two main benefits of inheritance?
- How do you ensure a child class runs the parent's `__init__`?
- How do you enforce a method contract for subclasses?
- What is polymorphism and why is it useful?
