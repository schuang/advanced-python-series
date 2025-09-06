# Pytest

As scientists, we are trained to be skeptical of our results. We run replicates, perform statistical checks, and demand rigorous proof. Yet, we often place blind faith in the correctness of our own code. We run it once on our data, it produces a plausible-looking graph, and we move on. This is a dangerous habit. A small bug in a data processing script can silently corrupt an entire dataset, potentially invalidating months of work.

This is where systematic testing comes in. Writing tests is the computational equivalent of running a control experiment. It is the single most effective way to ensure your code is—and remains—correct.

## Why the "Extra" Effort?

Frankly, writing tests feels like extra work. But the real question is, "compared to what?"
*   Compared to spending a week tracking down a bug that a simple test would have caught in seconds?
*   Compared to retracting a paper because your analysis code had a flaw?
*   Compared to being unable to modify or improve your code for fear of breaking it?

Testing is not about adding work; it's about **saving future work**. It provides:
1.  **A Safety Net:** It allows you to refactor, optimize, and change your code with confidence, knowing that if you break something, a test will fail immediately.
2.  **Reproducibility:** A comprehensive test suite is the ultimate proof that your code is behaving as you claim. It's a cornerstone of reproducible research.
3.  **Living Documentation:** Tests are concrete, runnable examples of how your functions are supposed to be used.

## Getting Started with `pytest`

`pytest` is a popular Python testing framework because it's powerful yet simple. It's designed to get out of your way.

**The Core Idea:** You write simple functions that `assert` that your code's output is what you expect.

Let's say you have a utility function in a file named `analysis_tools.py`:

```python
# analysis_tools.py
import numpy as np

def calculate_mean(data):
    """Calculates the arithmetic mean of a list or array."""
    return np.mean(data)
```

### Your First Test

Create a new file in the same directory called `test_analysis_tools.py`. `pytest` will automatically discover files that start with `test_` or end with `_test.py`.

```python
# test_analysis_tools.py
from . import analysis_tools # Import our code to be tested
import numpy as np

def test_calculate_mean_basic():
    """Test the mean calculation with a simple list."""
    input_data = [1, 2, 3, 4, 5]
    expected_result = 3.0
    # The core of the test: assert that the actual result equals the expected result.
    actual_result = analysis_tools.calculate_mean(input_data)
    assert actual_result == expected_result
```

That's it! Your test function must start with `test_`. Inside, you prepare your input, call your function, and use a plain `assert` statement.

To run your tests, open your terminal in that directory and simply run:
```bash
pytest
```

`pytest` will scan for test files and functions, run them, and give you a simple report. A passing test will show a green dot (`.`).

### What a Failing Test Looks Like

Let's add a test that we know will fail to see what happens.

```python
# test_analysis_tools.py (continued)
def test_calculate_mean_fail_example():
    """A test designed to fail to show the output."""
    input_data = [1, 2, 3]
    expected_result = 99.0 # Deliberately wrong
    actual_result = analysis_tools.calculate_mean(input_data)
    assert actual_result == expected_result
```

When you run `pytest`, you'll get a detailed, red report that tells you exactly what went wrong. It will show you the line that failed and the values of the variables involved. This immediate, precise feedback is what makes testing so powerful for debugging.

## Essential Testing Patterns for Scientists

### 1. Handling Floating-Point Numbers

Due to the nature of binary floating-point representation, direct comparison is often unsafe. `0.1 + 0.2` does not exactly equal `0.3` in Python. `pytest` provides a helper for this.

```python
from pytest import approx

def test_float_calculation():
    # Some complex calculation
    actual_result = 0.1 + 0.2
    expected_result = 0.3
    # assert actual_result == expected_result # This will FAIL!
    assert actual_result == approx(expected_result) # This will PASS.
```
Always use `pytest.approx` when comparing floating-point numbers.

### 2. Testing for Expected Errors

What should your function do if it receives bad input? For example, `calculate_mean` should probably fail if it gets an empty list. A function that fails silently is dangerous. It's better for it to raise an error. And you should test for that.

Let's modify our function to be more robust:
```python
# analysis_tools.py
def calculate_mean(data):
    """Calculates the arithmetic mean of a list or array."""
    if not data:
        raise ValueError("Input data cannot be empty")
    return np.mean(data)
```

Now, we can write a test to ensure this error is raised when it should be.

```python
# test_analysis_tools.py
import pytest

def test_calculate_mean_empty_list():
    """Test that our function correctly raises an error for empty input."""
    with pytest.raises(ValueError):
        analysis_tools.calculate_mean([])
```
This test passes *only if* `calculate_mean([])` raises a `ValueError`. If it doesn't raise an error, or raises a different one, the test will fail.

### 3. Fixtures: Reusing Test Setups

Often, many of your tests will need the same starting object, like a specific NumPy array, a pandas DataFrame loaded from a file, or a complex custom object. Instead of creating this object in every single test function, `pytest` lets you create a **fixture**.

A fixture is a function that prepares a resource for your tests to use.

```python
# test_analysis_tools.py

@pytest.fixture
def sample_dataframe():
    """Creates a sample pandas DataFrame for testing."""
    import pandas as pd
    data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
    return pd.DataFrame(data)

# Now, any test function that includes 'sample_dataframe' as an argument
# will automatically receive the result of that fixture.
def test_some_dataframe_operation(sample_dataframe):
    # The 'sample_dataframe' argument is the DataFrame from our fixture
    assert len(sample_dataframe) == 3
    assert sample_dataframe['col1'].sum() == 6

def test_another_dataframe_operation(sample_dataframe):
    assert sample_dataframe['col2'].mean() == 5.0
```
Fixtures are a powerful way to make your tests cleaner, faster, and more maintainable.

## A Change in Mindset

Writing tests forces you to think critically about your code: What are its inputs? What are its outputs? What could go wrong? This process alone will make you a better programmer. Start small. The next time you write a function, write one simple test for it. The confidence you gain from having a suite of tests that verify your code's correctness is invaluable and will fundamentally improve the quality and reliability of your research.

## An Advanced Technique: Test-Driven Development (TDD)

Once you are comfortable writing tests for your existing code, you might consider a more advanced and powerful workflow known as Test-Driven Development (TDD). The core idea is simple but transformative: **write the test *before* you write the code.**

The workflow follows a simple cycle:

1.  **Red:** Write a new test for a feature you haven't implemented yet. Run your tests. The new test will fail (turn red), because the code doesn't exist.
2.  **Green:** Write the absolute minimum amount of code necessary to make the test pass (turn green). Don't worry about making it elegant or efficient yet; just make it work.
3.  **Refactor:** Now that you have a passing test as a safety net, you can clean up your code. Improve the implementation, remove duplication, and make it more readable, running the tests frequently to ensure you haven't broken anything.

For a scientist, this approach has a profound benefit: it forces you to precisely define your function's requirements and expected outcomes *before* you get lost in the implementation details. It's the programmatic equivalent of clearly stating your hypothesis before you design the experiment. It shifts the focus from "What code should I write?" to "What result do I need to produce?"

While it can feel counter-intuitive at first, many people find that TDD leads to better-designed, more robust, and more modular code in the long run. It's a powerful skill to aspire to as you grow more confident in your testing practices.
