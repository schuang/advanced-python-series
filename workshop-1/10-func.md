# Python Functions

- Functions organize code into reusable, testable blocks

- Eliminate repetition and simplify debugging and maintenance

- Encapsulate local variables within function scope

We will explore several essential features of Python functions.

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

- Improve readability and documentation

- Catch bugs early with static analysis

- Enable better IDE support (auto-completion, type checking)

- Facilitate refactoring in large projects

**Cons:**

- Increase code verbosity

- Not enforced at runtime (without extra tools)

- Require extra effort for complex types

- Third-party libraries may lack type stubs

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

When calling a function, you can mix positional and keyword arguments, but positional arguments must come first in the call.

```python
# Valid
greet("Alice", greeting="Hi")  # positional, then keyword

# Invalid - causes SyntaxError
greet(name="Alice", "Hi")      # keyword before positional
```

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



## Passing values to a function

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

- Global variables are accessible throughout the module and can lead to bugs and hard-to-maintain code

- Prefer passing variables as function arguments

- Less modular, harder to test, and risk accidental modification




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

---------------------------------------------------------

## Exception handling within functions

Exception handling makes functions robust by catching and managing errors gracefully.

Key behaviors:

- When an exception is raised, control immediately **returns to the caller** (or nearest `except` block)

- Remaining lines in the function are not executed

- Unhandled exceptions abort the program and return control to the operating system

- Exception handling **keeps programs running** instead of terminating unexpectedly


Use `try`, `except`, and optionally `finally` blocks:

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
---------------------------------------------------------


## Default argument values

Default argument values allow parameters to have fallback values when not provided by the caller. This makes functions more flexible and easier to use.

Example:
```python
def greet(name, greeting="Hello"):
  print(f"{greeting}, {name}!")

greet("Alice")           # Output: Hello, Alice!
greet("Bob", greeting="Hi")  # Output: Hi, Bob!
```

Required parameters must come before parameters with default values in the function definition.

```python
# Valid
def greet(name, greeting="Hello"):  # required first, then default
    pass

# Invalid - SyntaxError
def greet(greeting="Hello", name):  # default before required
    pass
```

---------------------------------------------------------


## Keyword-only arguments

Keyword-only arguments must be specified by name when calling the function. Define them by placing a `*` in the function signature before those parameters.

Examples:

```python
def example(a, b, *, c, d=5):
  print(a, b, c, d)

example(1, 2, c=3)      # c must be specified as a keyword
example(1, 2, d=7, c=4) # both c and d must be specified as keywords
```

```python
def greet(name, *, greeting):  # greeting is keyword-only
    print(f"{greeting}, {name}!")

greet("Alice", "Hello")           # ERROR! Can't pass greeting positionally
greet("Alice", greeting="Hello")  # Required - must use keyword
```

Benefits: improves code clarity and prevents argument order mistakes. All parameters after `*` must be given as keywords.


---------------------------------------------------------

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

---------------------------------------------------------

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

### Custom Context Manager

You can create your own context managers by defining `__enter__` and `__exit__` methods:

```python
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        print("Timer started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start
        print(f"Timer stopped. Elapsed time: {elapsed:.2f}s")
        return False  # Don't suppress exceptions

# Usage
with Timer():
    # Do some work
    sum([i**2 for i in range(1000000)])
```

- The `__enter__` method runs when entering the `with` block and can return a value. 

- The `__exit__` method runs when exiting the block (even if an error occurs) and receives exception information if an error happened.


---------------------------------------------------------


## Function Decorators

- Decorators are functions that modify the behavior of other functions.
 
- Common uses: logging, timing, access control.

### Example

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

### Example

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

In `utils.py`:

```python
def helper():
    print("Helping!")
```

In another file:

```python
from utils import helper
helper()
```