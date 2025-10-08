# Calling External Code from Python: A Scientist's Guide

As computational scientists, we often stand on the shoulders of giants. Many powerful simulation engines and numerical libraries are written in compiled languages like C, Fortran, or modern high-performance languages like Julia. This tutorial explains the standard Python tools for interfacing with this external code.

---

## Part 1: Calling Compiled C & Fortran with `ctypes`

The most common task is calling functions from existing, compiled shared libraries (`.so`, `.dll`, `.dylib`). Python's built-in **`ctypes`** library is the perfect tool for this. It acts as a direct, low-level bridge to these libraries without requiring any extra installation or compilation steps for your Python code.

### The "Why": When to Use `ctypes`
Use `ctypes` when you have a pre-compiled library and you want to call functions from it directly, especially when you want to avoid setting up a complex build system.

### Practical Example 1: Calling C

Let's start with a simple C function.

**1. The C Code (`libadd.c`)**
```c
double add_doubles(double a, double b) {
    return a + b;
}
```

**2. Compile to a Shared Library**
```bash
gcc -shared -fPIC -o libadd.so libadd.c
```

**3. The Python `ctypes` Code**
The Python script must load the library, define the function's argument and return types, and then call it.

```python
import ctypes
import os

# 1. Load the library
lib_path = os.path.join(os.path.dirname(__file__), 'libadd.so')
my_c_library = ctypes.CDLL(lib_path)

# 2. Define the function signature
add_doubles_func = my_c_library.add_doubles
add_doubles_func.argtypes = [ctypes.c_double, ctypes.c_double]
add_doubles_func.restype = ctypes.c_double

# 3. Call the function
result = add_doubles_func(10.5, 20.2)
print(f"Result from C library: {result}")
```

### Practical Example 2: Calling Fortran

Calling Fortran is similar, but with two key differences: **name mangling** (the function name is often changed to lowercase with a trailing underscore) and **pass-by-reference** (all arguments are passed as pointers).

**1. The Fortran Code (`libadd_f.f90`)**
```fortran
subroutine add_doubles_f(a, b, result)
    real(8), intent(in) :: a, b
    real(8), intent(out) :: result
    result = a + b
end subroutine add_doubles_f
```

**2. Compile to a Shared Library**
```bash
gfortran -shared -fPIC -o libadd_f.so libadd_f.f90
```

**3. The Python `ctypes` Code**
```python
import ctypes
import os

# 1. Load the library
lib_path = os.path.join(os.path.dirname(__file__), 'libadd_f.so')
my_f_library = ctypes.CDLL(lib_path)

# 2. Define the function signature (note the name change and pointer types)
add_doubles_func = my_f_library.add_doubles_f_
add_doubles_func.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double)
]
add_doubles_func.restype = None

# 3. Prepare arguments and call by reference
a = ctypes.c_double(10.5)
b = ctypes.c_double(20.2)
result = ctypes.c_double()
add_doubles_func(ctypes.byref(a), ctypes.byref(b), ctypes.byref(result))

print(f"Result from Fortran library: {result.value}")
```

---

## Part 2: Bridging High-Level Languages

When interfacing with other modern scientific languages like Julia or R, `ctypes` is not the right tool. Instead, we use dedicated high-level libraries that manage the communication between the two language runtimes.

### Calling Julia from Python with `PyJulia`

The `julia` library (`pip install julia`) starts a Julia session in the background and provides a seamless bridge.

**1. The Julia Code (`my_analysis.jl`)**
```julia
function process_data(data, constant)
    return data.^2 .+ constant
end
```

**2. The Python Code**
```python
import numpy as np
from julia import Main as jl

# This starts the Julia VM (can be slow the first time)
jl.include("my_analysis.jl")

python_array = np.array([1.0, 2.0, 3.0])
# PyJulia handles the conversion of NumPy arrays automatically
result_array = jl.process_data(python_array, 100.0)
print(f"Result from Julia: {result_array}")
```

### Calling R from Python with `rpy2`

Similarly, the `rpy2` library (`pip install rpy2`) embeds an R process in Python, with excellent support for converting pandas DataFrames.

```python
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
stats = importr('stats')

py_df = pd.DataFrame({'x': [1,2,3,4,5], 'y': [1.1,1.9,3.2,4.3,5.1]})

# rpy2 converts the pandas DataFrame and calls R's lm() function
linear_model = stats.lm("y ~ x", data=py_df)
coefficients = stats.coef(linear_model)
print(f"Slope from R linear model: {coefficients[1]:.4f}")
```

---

## Part 3: Choosing the Right Tool - The Interoperability Ecosystem

`ctypes` is just one tool in a rich ecosystem. Here is a guide to help you choose the right one for your task.

*   **To accelerate your Python code:**
    *   **Numba:** The best choice for speeding up `for` loops over NumPy arrays using a JIT compiler.
    *   **Cython:** A more powerful tool that compiles a Python-like language to C. Use it for complex acceleration tasks or for creating detailed C-level bindings.

*   **To call existing C/C++/Fortran libraries:**
    *   **`ctypes` (this tutorial):** The best choice when you need a quick, simple way to call a few functions from a pre-compiled library without any extra build steps. It's built-in and easy to use.
    *   **`pybind11`:** The modern standard for creating clean, high-quality bindings for C++ libraries. This is the tool of choice if you are a C++ developer looking to expose your code to Python.
    *   **Cython:** Can also be used to create very powerful, high-performance bindings, giving you more control than `ctypes`.

*   **To interface with other high-level languages:**
    *   **`PyJulia`:** The standard for calling Julia.
    *   **`rpy2`:** The standard for calling R.

By understanding this landscape, you can choose the most appropriate tool for your specific scientific computing challenge.