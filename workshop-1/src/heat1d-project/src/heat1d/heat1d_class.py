import numpy as np
import matplotlib.pyplot as plt

class Grid:
    """Represents the spatial discretization of the 1D domain."""
    def __init__(self, nx, L):
        """
        Initializes the grid.
        
        Args:
            nx (int): Number of grid points.
            L (float): Length of the domain.
        """
        self.nx = nx
        self.L = L
        self.dx = L / (nx - 1)
        self.x = np.linspace(0, L, nx)

class HeatSolver:
    """
    A solver for the 1D heat equation using a specific numerical scheme.
    
    This class encapsulates the state of the simulation (the temperature field)
    and the methods to operate on that state.
    """
    def __init__(self, grid, alpha, dt, source_term=None):
        """
        Initializes the solver.
        
        Args:
            grid (Grid): The grid object defining the spatial domain.
            alpha (float): The thermal diffusivity.
            dt (float): The time step size.
            source_term (np.ndarray, optional): A time-independent source term.
        """
        self.grid = grid
        self.alpha = alpha
        self.dt = dt
        self.T = np.zeros(grid.nx) # The temperature field is now an attribute
        self.source_term = source_term
        if self.source_term is not None and len(self.source_term) != grid.nx:
            raise ValueError("Source term array must match grid size.")

    def set_initial_condition(self, T0):
        """Sets the initial temperature distribution."""
        if len(T0) != self.grid.nx:
            raise ValueError("Initial condition array must match grid size.")
        self.T = T0.copy()

    def step(self):
        """
        Advances the solution by one time step using a second-order accurate
        representation of the Dirichlet boundary conditions.
        """
        # 1. Define the fixed boundary values
        T_left_boundary = 0.0
        T_right_boundary = 0.0

        # 2. Calculate the required ghost cell values to center the BC
        T_ghost_left = 2 * T_left_boundary - self.T[1]
        T_ghost_right = 2 * T_right_boundary - self.T[-2]

        # 3. Create a temporary padded array for the derivative calculation
        T_padded = np.empty(self.grid.nx + 2)
        T_padded[0] = T_ghost_left
        T_padded[-1] = T_ghost_right
        T_padded[1:-1] = self.T

        # 4. Calculate the second derivative for ALL interior points at once
        d2T_dx2_interior = (T_padded[2:] - 2*T_padded[1:-1] + T_padded[:-2]) / self.grid.dx**2

        # 5. Calculate the full right-hand side
        rhs = self.alpha * d2T_dx2_interior[1:-1]
        if self.source_term is not None:
            rhs += self.source_term[1:-1]

        # 6. Update the interior temperature points
        self.T[1:-1] += self.dt * rhs

        # 7. Re-enforce the exact boundary values on the grid points
        self.T[0] = T_left_boundary
        self.T[-1] = T_right_boundary

    def solve(self, nt):
        """
        Runs the full simulation for a given number of time steps.
        
        Args:
            nt (int): The number of time steps to run.
        """
        for _ in range(nt):
            self.step()
        print(f"Simulation finished after {nt} time steps.")


from .heat_plotter import plot_heat_comparison

# --- Main execution block ---
# if __name__ == "__main__":
#     # 1. Setup the simulation parameters
#     nx = 21
#     L = 1.0
#     alpha = 0.01
#     nt = 2000
#     dt = 0.001

#     # 2. Create the objects
#     grid = Grid(nx, L)
#     source = np.sin(2 * np.pi * grid.x)
#     solver = HeatSolver(grid, alpha, dt, source_term=source)

#     # 3. Set the initial and boundary conditions
#     T_initial = np.zeros_like(grid.x) # Initial temperature is zero everywhere
#     T_initial[0] = 0.0
#     T_initial[-1] = 0.0
#     solver.set_initial_condition(T_initial)

#     # 4. Run the simulation by calling a method on the object
#     solver.solve(nt)
    
#     # 5. Output and Visualization
#     t_final = nt * dt
#     T_exact = (1 / (4 * np.pi**2 * alpha)) * np.sin(2 * np.pi * grid.x) * (1 - np.exp(-4 * np.pi**2 * alpha * t_final))
    
#     print(f"Final temperature at the center (x=0.5): {solver.T[nx//2]:.6f}")
#     print(f"Exact temperature at the center (x=0.5):   {T_exact[nx//2]:.6f}")

#     plot_heat_comparison(
#         grid=grid,
#         t_final=t_final,
#         filename='heat1d_class.png',
#         initial_condition=T_initial,
#         numerical_solutions={"2nd Order": solver.T},
#         exact_solution=T_exact
#     )

