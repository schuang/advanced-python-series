
# Workshop 0: A Quick Review of Python for Scientific Computing

**Part of the series:** *From Scripts to Software: Practical Python for Reproducible Research*

This optional workshop is for those new to Python or who need a refresher on the basics. We will cover the fundamental syntax and concepts required to follow along with the rest of the workshop series.

### Learning Objectives

*   Understand basic Python data types and data structures.
*   Use control flow (loops, conditionals) to direct the logic of your scripts.
*   Write and call functions to create reusable code.
*   Get a first look at the structure of classes and objects.

## Part 1: Python Basics

*   **Variables and Data Types:** `int`, `float`, `str`, `bool`
    *   See example: [workshop-0/01_data_types.py](workshop-0/01_data_types.py)
*   **Data Structures:** `lists`, [`dictionaries`](workshop-0/dict.md).
    *   See example: [workshop-0/02_data_structures.py](workshop-0/02_data_structures.py)

## Part 2: Control Flow

*   **Conditional Statements:** `if`, `elif`, `else`
    *   See example: [workshop-0/03_conditionals.py](workshop-0/03_conditionals.py)
*   **Loops:** `for`, `while`
    *   See example: [workshop-0/04_loops.py](workshop-0/04_loops.py)

## Part 3: Reusing Code with Functions

*   **Defining and Calling Functions:**
    *   See example: [workshop-0/05_functions.py](workshop-0/05_functions.py)

## Part 4: A First Look at Objects

*   **Classes and Objects:** A brief introduction to the basic structure of a class.
    *   See example: [workshop-0/06_simple_class.py](workshop-0/06_simple_class.py)


## Part 5: other important topics

- Python Constructs
   * [tuple](workshop-0/tuple.md) and namedtuple
   * [set](workshop-0/set.md)
   * [Functions](workshop-0/function.md): Pass functions as arguments, return them from other functions. Decorators.
   * Comprehensions & Generators

- Structuring for Reusability
   * Modules and Packages: Organize code into separate files (modules) and directories (packages) with __init__.py to create a clear, importable structure.
   * Object-Oriented Programming (OOP) Basics:
       * Classes & Objects: Encapsulate data (attributes) and behavior (methods).
       * Special Methods (Dunders): __init__, __repr__, __str__, __len__. Make your objects behave like standard Python types.
       * Properties: Control attribute access with @property for cleaner APIs.
   * Decorators: A simple, powerful way to add functionality (e.g., logging, timing, caching) to functions or methods without modifying their core logic.
   * Context Managers (`with`): Ensure resources are properly managed and released.

- Ensuring Code Quality
   * [Error Handling](workshop-0/exception.md): `try...except...else...finally`.
   * [Type hints](workshop-0/type_hint.md) Use type hints to improve code clarity, catch bugs early with static analysis tools (like mypy), and enable better IDE support. This makes complex data structures (e.g., a dict of lists of tuples) far more understandable.
   * [`pytest`](workshop-0/pytest.md): Write simple, effective tests.
   * Docstrings: Document your code's API using a standard format (e.g., NumPy/SciPy or Google style) so tools can auto-generate documentation.

- Environment and Dependency Management
   * Virtual Environments: [`venv`](workshop-0/venv.md), [`conda`](workshop-0/conda.md) and [`uv`](workshop-0/uv.md).
   * Dependency Management: `requirements.txt` and `pyproject.toml` file.

- Data and numerical computing
   * [Numpy](workshop-0/numpy.md)
   * [Numba](workshop-0/numba.md)


## Next Steps

With the working knoledge of these fundamentals, you are now ready to proceed to **Workshop 1: Foundations for Sustainable Research Software**, where we will explore how to use these building blocks to write more robust and maintainable scientific code.

## References

- [Python book](https://assets.openstax.org/oscms-prodcms/media/documents/Introduction_to_Python_Programming_-_WEB.pdf) by OpenStax
