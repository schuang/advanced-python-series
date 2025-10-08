# Python Functions

Functions vs. Linear Scripts:

- Linear scripts are harder to scale and modify.
- Functions organize code into reusable, testable blocks.
- Avoids repetition and makes code easier to debug and maintain.
- Hide local variables inside functions.

"Modern" Python features to use:

- Type hints
- Pass by value and pass by reference
- Function decorators
- Calling functions in other files
- Default argument values
- Keyword-only arguments (using *, in function signature)
- Flexible interface with `*args` and `**kwargs`
- Return multiple values (tuples, dataclasses)
- Type aliases for complex types
- Context managers
- Exception handling within functions
- Use of functools utilities (e.g., @lru_cache, partial)

Other features:

- Docstrings for documentation
- Lambda (anonymous) functions
- Use of built-in decorators (@staticmethod, @classmethod, @property)
- Support for asynchronous functions (async def, await)


---------------------------------------------------------



## Type Hinting

Improves code readability and helps with static analysis tools.

Introduced in Python 3.5 (PEP 484) in 2015.


```python
count: int = 0
name: str = "Alice"
data: list[float] = [1.0, 2.5, 3.7]

from typing import Any
mixed: list[Any] = [1, "two", 3.0, [4, 5]]

def add(x: int, y: int) -> int:
  return x + y
```

#### Pros and Cons of Type Hints

**Pros:**

- Improve code readability and documentation.

- Help catch bugs early with static analysis tools.

- Make code easier to understand for collaborators.

- Enable better IDE support (auto-completion, type checking).

- Facilitate refactoring and maintenance in large projects.

**Cons:**

- Slightly increase code verbosity.

- Do not enforce types at runtime (unless using extra tools).

- May require extra effort for complex types or dynamic code.

- Can be confusing for beginners unfamiliar with typing syntax.

- Some third-party libraries may not have complete type stubs.

**Note:** Type hints do not affect compiled Python code in libraries like Numba or JAX. These libraries ignore Python type hints and use their own mechanisms (such as decorators or explicit type annotations) for optimization and compilation. Type hints are mainly for static analysis, documentation, and editor support—not for runtime or compilation behavior in Numba/JAX.

### Type Hint Examples with NumPy
```python
import numpy as np
from typing import Tuple, Optional

# Calculate mean of a 1D array
def mean(arr: np.ndarray) -> float:
  return float(np.mean(arr))

# Normalize a vector
def normalize(vec: np.ndarray) -> np.ndarray:
  return vec / np.linalg.norm(vec)

# Compute dot product of two arrays
def dot(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.dot(a, b))

# Return shape and dtype of an array
def array_info(arr: np.ndarray) -> Tuple[Tuple[int, ...], str]:
  return arr.shape, str(arr.dtype)

# Optional return type: can return None if input is empty
def safe_mean(arr: np.ndarray) -> Optional[float]:
  if arr.size == 0:
    return None
  return float(np.mean(arr))

# Function accepting and returning 2D arrays
def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  return np.matmul(a, b)
```

---------------------------------------------------------


## Function signature

A Python function is defined by its signature, which includes:

- The function name
- The list of arguments (parameters)
- The return type (optional, with type hints)

### Positional Arguments vs Keyword Arguments

**Positional arguments** are specified by their position in the function call:
  
```python
def add(x, y):
    return x + y
add(2, 3)  # x=2, y=3
```

**Keyword arguments** are specified by name:

```python
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")
greet(name="Alice", greeting="Hi")
```

You can mix positional and keyword arguments, but positional arguments must come first.

### Function Argument Types

- Arguments can be any Python object: int, float, str, list, dict, custom class, etc.

- You can use type hints to specify expected types:

```python
def scale(x: float, factor: float) -> float:
    return x * factor
```

### Return Type

- Functions can return any Python object.

- Use type hints to specify the return type:

```python
def get_name() -> str:
    return "Alice"
```


---------------------------------------------------------




## Pass by Value vs. Pass by Reference

- Python uses "pass by object reference" (sometimes called "pass by assignment").

- Mutable objects (lists, dicts) can be changed inside functions.

- Immutable objects (ints, strings, tuples) cannot be changed inside functions.

- Example:
  ```python
  def modify_list(lst):
      lst.append(1)
  my_list = []
  modify_list(my_list)
  print(my_list)  # [1]
  ```

---------------------------------------------------------



## Global Variables

Avoid using `global` keyword unless absolutely necessary.

- Global variables are accessible throughout the module - can lead to bugs and hard-to-maintain code.
- Prefer passing variables as function arguments.
- less modular, harder to test, risk of accidental modification.




---------------------------------------------------------


## Type Aliases for Complex Types

Type aliases make code easier to read and maintain, especially for complex or repeated type hints.

Example:
```python
from typing import List, Tuple, Dict

# Define a type alias for a list of 2D coordinates
Coordinates = List[Tuple[float, float]]

# Use the alias in a function signature
def total_distance(points: Coordinates) -> float:
  # ... implementation ...
  pass

# Alias for a dictionary mapping strings to lists of integers
StrToIntList = Dict[str, List[int]]
```

Type aliases are especially useful in scientific code, where data structures can be complex and reused across many functions.


## Exception handling within functions

Exception handling makes your functions robust and user-friendly by catching and managing errors gracefully.

When an exception is raised, control is immediately returned to the caller (or to the nearest enclosing `except` block), and the remaining lines in the function are not executed.

If an exception is not handled anywhere in your code, the program will abort and control returns to the operating system. By using exception handling, you can catch errors and keep your program running, instead of having it terminate unexpectedly.

Use `try`, `except`, and optionally `finally` blocks to handle exceptions:

```python
def safe_divide(x: float, y: float) -> float:
  try:
    return x / y
  except ZeroDivisionError:
    print("Error: Division by zero!")
    return float('inf')

def read_file(filename: str) -> str:
  try:
    with open(filename) as f:
      return f.read()
  except FileNotFoundError:
    print(f"File not found: {filename}")
    return ""
```

You can also raise exceptions to signal errors to the caller:

```python
def get_item(lst: list, idx: int):
  if idx < 0 or idx >= len(lst):
    raise IndexError("Index out of range")
  return lst[idx]
```

## Default argument values

Default argument values let you specify a value for a parameter if the caller does not provide one. This makes functions more flexible and easier to use.

Example:
```python
def greet(name, greeting="Hello"):
  print(f"{greeting}, {name}!")

greet("Alice")           # Output: Hello, Alice!
greet("Bob", greeting="Hi")  # Output: Hi, Bob!
```

You can set default values for any parameter except those before a required positional argument.


## Keyword-only arguments

Keyword-only arguments are parameters that must be specified by name (as a keyword) when calling the function. In Python, you define them by placing a `*` in the function signature before those arguments.

Example:
```python
def example(a, b, *, c, d=5):
  print(a, b, c, d)

example(1, 2, c=3)      # c must be specified as a keyword
example(1, 2, d=7, c=4) # both c and d must be specified as keywords
```

This helps make code clearer and prevents mistakes from passing arguments in the wrong order. All arguments after `*` must be given as keywords.



## *args and **kwargs

- `*args`: allows a function to accept any number of positional arguments (as a tuple).

- `**kwargs`: allows a function to accept any number of keyword arguments (as a dict).

Example:

```python
def demo(*args, **kwargs):
    print(args)
    print(kwargs)
demo(1, 2, 3, a=4, b=5)
# Output: (1, 2, 3) {'a': 4, 'b': 5}
```


## Context managers (with statement, custom via __enter__/__exit__)

Context managers are a Python feature that help you manage resources (like files, network connections, or locks) safely and automatically. They ensure setup and cleanup code runs reliably, even if errors occur.

The most common way to use a context manager is with the `with` statement.

### File Handling

You will never forget to "close" a file:

```python
with open("data.txt") as f:
  data = f.read()
# File is automatically closed when the block ends
```

### Connecting to a database

```python

import sqlite3

# Using the built-in context manager for sqlite3
with sqlite3.connect("example.db") as conn:
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER, name TEXT)")
    cursor.execute("INSERT INTO users VALUES (?, ?)", (1, "Alice"))
    conn.commit()
    cursor.execute("SELECT * FROM users")
    print(cursor.fetchall())
# Connection is automatically closed when the block ends

```

## Function Decorators

- Decorators are functions that modify the behavior of other functions.
 
- Common uses: logging, timing, access control.

### Example 1
  ```python
  def my_decorator(func):
      def wrapper(*args, **kwargs):
          print("Before function call")
          result = func(*args, **kwargs)
          print("After function call")
          return result
      return wrapper

  @my_decorator
  def greet(name):
      print(f"Hello, {name}!")
  greet("Alice")
  ```

### timing example

```python
import time

def timing_decorator(func):
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"{func.__name__} took {end - start:.4f} seconds")
    return result
  return wrapper

@timing_decorator
def slow_function():
  time.sleep(1)
  print("Done!")

slow_function()
```

---------------------------------------------------------


## Calling Functions from Other Files

- Use `import` to access functions defined in other Python files (modules).
- Example:
  - In `utils.py`:
    ```python
    def helper():
        print("Helping!")
    ```
  - In another file:
    ```python
    from utils import helper
    helper()
    ```