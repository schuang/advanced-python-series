# Modern Python Classes: Decorators & Dataclasses

- **Decorators: The @ Symbol**
  - Decorators are functions that "wrap" other functions or methods
  - Add extra behavior (e.g., logging, timing, registration) without changing core logic
  - Syntax: `@decorator_name` above a function or method
  - Common built-in decorators: `@classmethod`, `@staticmethod`, `@abstractmethod`, `@jit`

- **How Decorators Work**
  - Decorator takes a function, returns a modified version
  - Example: `@timer` measures and prints how long a function takes
  - `@decorator` is shorthand for: `func = decorator(func)`

- **Practical Example: Timing a Function**
  - Use `@timer` to measure runtime of scientific calculations
  - Decorators accept any arguments via `*args, **kwargs`

- **Why Use Decorators?**
  - Clean, reusable way to add features to many functions
  - Keeps code readable and uncluttered

---

- **Dataclasses: Simplifying Data Containers**
  - `@dataclass` automatically generates `__init__`, `__repr__`, and `__eq__` methods
  - Reduces boilerplate for classes that mainly store data
  - Just declare attributes with type hints

- **Manual vs. Dataclass Example**
  - Manual class: must write all methods yourself
  - Dataclass: just list attributes, get useful methods for free

- **Benefits for Scientific Computing**
  - Fast, readable data objects for simulations, experiments, file records
  - Easy debugging and testing (automatic comparison and printing)
  - Focus on science, not boilerplate

---

- **Key Takeaways**
  - Decorators add reusable features to functions and methods
  - `@dataclass` is ideal for simple, data-focused classes
  - Both features help write cleaner, more maintainable scientific code
