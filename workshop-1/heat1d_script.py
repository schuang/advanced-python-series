import numpy as np
import matplotlib.pyplot as plt

# heat1d_script.py
# A simple script to solve the 1D heat equation using explicit Euler
# and second-order finite differences. This demonstrates the "floating
# variables" anti-pattern discussed in class-1.md.

# --- 1. Parameters (Floating Variables) ---
L = 1.0          # Length of the domain
nx = 21          # Number of grid points
dx = L / (nx - 1) # Grid spacing
alpha = 0.01     # Thermal diffusivity
nt = 2000        # Number of time steps
dt = 0.001       # Time step size

# Stability check (CFL condition for this scheme)
cfl = alpha * dt / dx**2
if cfl >= 0.5:
    print(f"Warning: CFL condition not met! CFL = {cfl:.2f}. Solution may be unstable.")

# --- 2. Grid and Initial Condition ---
x = np.linspace(0, L, nx)
T = np.zeros_like(x) # Initial temperature is zero everywhere
source = np.sin(2 * np.pi * x) # Heat source term

# Set boundary conditions (Dirichlet: T=0 at both ends)
T[0] = 0.0
T[-1] = 0.0

# Store initial condition for plotting
T_initial = T.copy()

# --- 3. Main Simulation Loop ---
# This is the core logic, mixed with the data it operates on.
for n in range(nt):
    # Create a copy to store the old temperature for calculation
    T_old = T.copy()
    
    # Loop over interior points to calculate the second derivative
    for i in range(1, nx - 1):
        # Second-order central difference for the second derivative
        d2T_dx2 = (T_old[i+1] - 2*T_old[i] + T_old[i-1]) / dx**2
        
        # Explicit Euler time step with source term
        T[i] = T_old[i] + dt * (alpha * d2T_dx2 + source[i])

    # Enforce boundary conditions at every time step
    T[0] = 0.0
    T[-1] = 0.0

from heat_plotter import plot_heat_comparison

# --- 4. Output and Visualization ---
t_final = nt * dt
T_exact = (1 / (4 * np.pi**2 * alpha)) * np.sin(2 * np.pi * x) * (1 - np.exp(-4 * np.pi**2 * alpha * t_final))

print(f"Simulation finished after {nt} time steps.")
print(f"Final temperature at the center (x=0.5): {T[nx//2]:.6f}")
print(f"Exact temperature at the center (x=0.5):   {T_exact[nx//2]:.6f}")

# Create a simple Grid-like object for the plotter
class SimGrid:
    def __init__(self, x_coords):
        self.x = x_coords
grid = SimGrid(x)

plot_heat_comparison(
    grid=grid,
    t_final=t_final,
    filename='heat1d_script.png',
    initial_condition=T_initial,
    numerical_solutions={"2nd Order": T},
    exact_solution=T_exact
)

