# Heat1D Simulation

A simple 1D heat equation solver for scientific computing.

## Installation

```bash
pip install heat1d-simulation
```

## Usage

```python
from heat1d import Heat1D

# Create a simulation
sim = Heat1D(length=1.0, nx=100, alpha=0.01)

# Run the simulation
sim.solve(nt=1000, dt=0.0001)

# Plot results
sim.plot()
```

## Requirements

- Python >= 3.8
- numpy
- matplotlib
