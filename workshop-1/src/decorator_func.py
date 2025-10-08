import time

# 1. This is the decorator function.
#    It takes a function (`func`) as its input.
def timer(func):
    # 2. It defines a new "wrapper" function inside.
    #    *args and **kwargs are a standard way to accept any arguments.
    def wrapper(*args, **kwargs):
        # 3. Code to run BEFORE the original function.
        start_time = time.time()
        
        # 4. Call the original function and save its result.
        result = func(*args, **kwargs)
        
        # 5. Code to run AFTER the original function.
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed:.4f} seconds to run.")
        
        # 6. Return the original function's result.
        return result
    
    # 7. The decorator returns the newly defined wrapper function.
    return wrapper

# --- Now, let's use our decorator ---

@timer
def run_complex_calculation(n_points):
    """A placeholder for a long-running scientific task."""
    print(f"Running calculation with {n_points} points...")
    total = 0
    for i in range(n_points):
        total += i
    return total

# Now, when we call this function, it's actually the decorated version.
result = run_complex_calculation(10_000_000)