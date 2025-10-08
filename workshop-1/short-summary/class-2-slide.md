# Python Classes: Syntax & Grammar

- **Class vs. Object**
  - Class: blueprint/template for objects
  - Object: instance with its own data

- **Why Use Classes?**
  - Bundles data and behavior
  - Safer, more organized, easier to debug/share

- **Basic Structure**
  - `class` keyword defines a class
  - `__init__`: constructor, sets up instance attributes
  - `self`: refers to the specific object instance

- **Attributes**
  - Instance attributes: unique to each object (`self.name`)
  - Class attributes: shared by all instances (`_known_elements`)

- **Methods**
  - Instance methods: operate on unique data (`self`)
  - Class methods: operate on shared data (`@classmethod`, `cls`)
  - Static methods: utility functions, no access to instance/class (`@staticmethod`)

- **Naming Conventions**
  - Class names: CamelCase
  - Methods/variables: snake_case

- **Multiple Constructors**
  - Use default arguments in `__init__` for simple cases
  - Use `@classmethod` factory methods for alternatives

- **Advanced: Multiple Dispatch**
  - Use `multipledispatch` library for function overloading by argument types

- **Key Takeaways**
  - Class = blueprint, Object = instance
  - `__init__` sets up object state
  - Use instance/class/static methods appropriately
  - Prefer `@classmethod` factories for multiple constructors

---

## Quick Quiz (for review)
- Class vs. object: what's the difference?
- When to use instance, class, or static methods?
- How to create multiple constructors in Python?
- How to organize shared vs. unique data?
