# Python Classes

## The Problem with Scripts

As researchers, we often start with a script:

- Load data
- Define variables for parameters
- Run functions to get results

This works, but has serious limitations:

- Data and parameters "float" around without structure
- Easy to accidentally use wrong variables
- Simple to corrupt data in ways that break later steps


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



------------------------------------------------


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

------------------------------------------------


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

## Why Classes Are Critical for Reproducible Science

Funding agencies, publishers, and the scientific community increasingly demand reproducible computational research. This means code must be clear, robust, and easy for others to run and understand.

Classes achieve this in ways that are difficult without them:

1.  Encapsulation for Integrity
    - Bundles data and methods together
    - Protects analysis integrity
    - Results stored safely inside objects
    - Harder to accidentally modify
    - Object acts as a "snapshot" of specific analysis

2.  Clarity and Reusability
    - Makes workflow explicit
    - Easy to create independent analysis objects
    - Multiple datasets without interference
    ```python
    sim_run_2 = SimulationAnalysis('sim_run_02.xtc', 'system.pdb', 310.0, 1.0)
    sim_run_2.load_data()
    sim_run_2.calculate_rmsd()
    ```
    - Nearly impossible to manage safely with floating variables

3.  A Contract for Your Science
    - `__init__` defines required parameters
    - Methods define operation sequence
    - Easier for colleagues and reviewers to verify methodology

4.  Enabling Safe Refactoring
    - Refactoring: restructure code to improve design without changing behavior
    - Like revising a manuscript: improve flow and clarity, not content
    - Tangled scripts are terrifying to change
    - Classes make refactoring safe through encapsulation
    - Change internal implementation without breaking other code
    - Essential for long-term research projects

### Beyond Code: Reproducing the Computing Environment

True reproducibility requires more than well-structured code. The classic "it works on my computer" syndrome occurs when environments differ.

The environment must be captured and shared at multiple levels:

- Python Dependencies
  - Recreate exact package versions
  - Tools: `pip` with `requirements.txt`, `conda`, or `uv`

- System-Level Dependencies
  - C libraries or system tools
  - Tools: `conda` for non-Python software

- Entire Operating System
  - Highest level of reproducibility
  - Tools: containers (Docker, Apptainer)
  - Package code, dependencies, and OS into portable image

Two critical steps for reproducibility:
1. Use classes to organize scientific logic
2. Place code into well-defined, reproducible environment

Bottom line: Without classes, you present a script and loose variables, asking others to trust correct usage. With classes, you present a self-contained, verifiable scientific instrument.
