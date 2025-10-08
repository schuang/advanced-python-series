# Python Classes

As researchers, we often start with a script. We load some data, define a dozen variables for our parameters, and then run a series of functions to get a result. This works, but it's fragile. The data and the parameters are just "floating" around, and it's easy to accidentally use the wrong variable or modify data in a way that corrupts a later step.

As our analysis grows more complex, we face a critical organizational challenge: how do we bundle the **data** of an experiment with the specific **logic** used to analyze it? This is the problem that Python's **classes** were designed to solve.

A class is a blueprint for creating **objects**. Think of it as a self-contained "mini-program" that packages together related data (attributes) and the functions that operate on that data (methods).

## The Problem: "Floating" Variables and Reproducibility

Imagine a script to analyze a simulation trajectory. You might have:

```python
# A collection of disconnected variables
trajectory_file = 'sim_run_01.xtc'
topology_file = 'system.pdb'
temperature = 300.0
pressure = 1.0
results = None # Will be populated later

def load_data(traj, top):
    # ... loads data ...
    return loaded_data

def calculate_rmsd(data, temp):
    # ... does a calculation ...
    return rmsd_values

# The script runs top-to-bottom
sim_data = load_data(trajectory_file, topology_file)
results = calculate_rmsd(sim_data, temperature)
```
This is hard to reproduce. What if you forget to pass `temperature` to the `calculate_rmsd` function? What if another part of the script accidentally changes the value of `sim_data`? The state of your analysis is fragile and spread out.

### An Alternative: The "Composite Function"
A common instinct for organizing this workflow without classes is to create a single "master" function that calls the other functions in sequence.

```python
def run_full_analysis(trajectory_file, topology_file, temperature, pressure):
    """Runs the entire analysis pipeline."""
    sim_data = load_data(trajectory_file, topology_file)
    results = calculate_rmsd(sim_data, temperature)
    # ... maybe a dozen more steps ...
    return results
```
This is certainly an improvement over a loose script, but it has a critical weakness that highlights the true value of classes.

### The Problem with the Composite Function: Opaque State
The composite function is a **black box**. You put parameters in, and you get a final result out. But what if you want to inspect the intermediate steps?
*   How do you access the `sim_data` after it's been loaded?
*   What if you want to run an additional, exploratory analysis on the loaded data without re-running the whole pipeline?
*   What if you want to change a parameter and only re-run the last step?

With a composite function, you can't. The intermediate data (`sim_data`) is created and then immediately thrown away inside the function's local scope. The entire state of the analysis is temporary and inaccessible.

## The Solution: A Self-Contained `Simulation` Class

A class lets us bundle all of this into a single, coherent unit.

```python
import numpy as np

class SimulationAnalysis:
    """A class to hold and analyze a single simulation run."""

    def __init__(self, trajectory_file, topology_file, temperature, pressure):
        """The constructor: sets up the object's initial state."""
        # --- Attributes: The data of the object ---
        self.trajectory_file = trajectory_file
        self.topology_file = topology_file
        self.temperature = temperature
        self.pressure = pressure
        
        # Internal state, starts empty
        self.data = None
        self.rmsd_results = None

    def load_data(self):
        """A method that operates on the object's own data."""
        print(f"Loading data from {self.trajectory_file}...")
        # In a real scenario, this would use a library like MDAnalysis or mdtraj
        # self.data = ... load from self.trajectory_file ...
        self.data = np.random.rand(100, 3) # Placeholder for real data

    def calculate_rmsd(self):
        """Another method. It uses the object's internal state."""
        if self.data is None:
            raise ValueError("Data not loaded. Please run .load_data() first.")
        
        # Notice how it uses its own attributes, like self.temperature
        print(f"Calculating RMSD at {self.temperature} K...")
        # self.rmsd_results = ... perform calculation on self.data ...
        self.rmsd_results = np.mean(self.data, axis=1) # Placeholder calculation
```

### Using the Class
Now, our analysis is an "object." All the data and logic are neatly packaged together.

```python
# Create an instance of our analysis for a specific run
sim_run_1 = SimulationAnalysis(
    trajectory_file='sim_run_01.xtc',
    topology_file='system.pdb',
    temperature=300.0,
    pressure=1.0
)

# Call the methods in a logical order
sim_run_1.load_data()
sim_run_1.calculate_rmsd()

# The results are stored safely inside the object
print(f"First 5 RMSD values: {sim_run_1.rmsd_results[:5]}")
```

## Why This is Critical for Reproducible Science

Funding agencies (NIH, NSF), publishers (Nature, Science), and the scientific community at large are increasingly demanding that computational research be **reproducible**. This means more than just getting the same answer twice. It means your code must be clear, robust, and easy for others to run and understand.

This is what using classes helps you achieve, in ways that are very difficult without them:

1.  **Encapsulation for Integrity:** By bundling data and methods, a class protects the integrity of your analysis. The `rmsd_results` are stored inside the `sim_run_1` object. It's much harder for an unrelated piece of code to accidentally modify it. The object acts as a "snapshot" of a specific analysis, with a well-defined state.

2.  **Clarity and Reusability:** The class makes your workflow explicit. You can now easily create a second, independent analysis object for a different dataset, and it will not interfere with the first one.
    ```python
    # A second, completely independent analysis
    sim_run_2 = SimulationAnalysis('sim_run_02.xtc', 'system.pdb', 310.0, 1.0)
    sim_run_2.load_data()
    sim_run_2.calculate_rmsd()
    ```
    This is nearly impossible to manage safely with dozens of floating variables.

3.  **A Contract for Your Science:** A well-defined class is like a contract. Its `__init__` method defines exactly what parameters are needed to run the analysis. Its methods define the exact sequence of operations. This makes it far easier for a colleague (or a reviewer) to understand and verify your methodology.

4.  **Enabling Safe Refactoring:**
    Throughout these tutorials, the term "refactor" has been mentioned. This is a critical concept in software engineering that is highly relevant to scientific computing.
    *   **What is refactoring?** It is the process of restructuring existing code to improve its internal design, clarity, and maintainability, **without changing its external behavior.**
    *   **What it is NOT:** Refactoring is not about fixing bugs or adding new features.
    The best analogy for a researcher is **revising a manuscript**. When you revise a paper, you aren't adding new data (new features) or correcting a factual error (fixing a bug). You are improving the logical flow and clarifying sentences to make the paper easier to understand. Refactoring is the code equivalent of that revision process.
    A tangled scientific script is terrifying to change. Improving one part risks breaking ten other things. Classes make refactoring safe. Because a class **encapsulates** its data and methods, you can freely change the *internal* implementation of a method, and as long as you don't change its name or what it returns, you can be confident you haven't broken other parts of your code. This ability to safely improve your code over time is essential for long-term research projects.

### Beyond Code: Reproducing the Computing Environment
True reproducibility, however, requires more than just well-structured code. A script that runs perfectly on your machine might fail on a colleague's -- the classic "it works on my computer" syndrome -- not because the code is wrong, but because the **environment** is different. They might have a different version of NumPy, a missing library, or even a different operating system.

To achieve full reproducibility, the computing environment itself must be captured and shared. This is a layered problem:
*   **Python Dependencies:** At a minimum, you must be able to recreate the exact set of Python packages and their versions. This is handled by tools like `pip` with `requirements.txt` files or, more robustly, by environment managers like `conda` or `uv`.
*   **System-Level Dependencies:** What if your code depends on a specific C library or system tool? This is where environment managers like `conda` shine, as they can package non-Python software.
*   **The Entire Operating System:** For the highest level of reproducibility, you can use **containers** (like Docker or Apptainer). A container packages your code, all its dependencies, and a snapshot of the entire operating system into a single, portable image that will run identically on any machine.

Using classes to structure your code is the first and most critical step. It organizes your scientific logic. The next step is to place that well-structured code into a well-defined, reproducible environment.

Without classes, you are essentially presenting a script and a loose collection of variables, and asking others to trust that they are all used correctly. With classes, you are presenting a self-contained, verifiable **scientific instrument**. This shift from a script to an object-oriented model is the cornerstone of writing professional, reproducible scientific software.
