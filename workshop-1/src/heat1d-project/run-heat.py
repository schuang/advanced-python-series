# run this after installing "heat1d" in the current python venv

import numpy as np
from heat1d.heat1d_class import Grid, HeatSolver
from heat1d.heat_plotter import plot_heat_comparison

nt = 2000
dt = 0.001
nx = 50
t_final = nt * dt

grid = Grid(nx=nx, L=1.0)
source = np.sin(2 * np.pi * grid.x)
solver = HeatSolver(grid, alpha=0.01, dt=dt, source_term=source)
initial_condition = np.zeros(nx)
solver.set_initial_condition(initial_condition)

# run the simulation
print(f"Running simulation for {nt} time steps...")
print(f"Grid points: {nx}, Time step: {dt}, Final time: {t_final}")
for n in range(nt):
    solver.step()
    if (n + 1) % 500 == 0:
        print(f"  Step {n+1}/{nt} complete")

print(f"Simulation complete!")

# Plot the results
numerical_solutions = {
    "2nd Order": solver.T
}

plot_heat_comparison(
    grid=grid,
    t_final=t_final,
    filename="heat1d_result.png",
    initial_condition=initial_condition,
    numerical_solutions=numerical_solutions
)

print(f"\nFinal temperature range: [{solver.T.min():.4f}, {solver.T.max():.4f}]")
