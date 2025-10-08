import numpy as np
import matplotlib.pyplot as plt

# A dictionary to define consistent plot styles
STYLES = {
    "Initial": {'color': 'k', 'linestyle': '--', 'label': 'Initial Condition'},
    "Exact":   {'color': 'r', 'linestyle': '-', 'linewidth': 2, 'label': 'Exact Solution'},
    "2nd Order": {'color': 'b', 'marker': 'o', 'markersize': 5, 'linestyle': 'None'},
    "4th Order": {'color': 'g', 'marker': '^', 'markersize': 5, 'linestyle': 'None'}
}

def plot_heat_comparison(
    grid,
    t_final,
    filename,
    initial_condition,
    numerical_solutions,
    exact_solution=None
):
    """
    A parameterized function to create a consistent plot for the 1D heat equation.

    Args:
        grid (Grid): The grid object with the x-coordinates.
        t_final (float): The final simulation time.
        filename (str): The path to save the output plot file.
        initial_condition (np.ndarray): The temperature array at t=0.
        numerical_solutions (dict): A dictionary where keys are labels (e.g., "2nd Order")
                                    and values are the final temperature arrays.
        exact_solution (np.ndarray, optional): The analytical solution. Defaults to None.
    """
    plt.figure(figsize=(12, 7))
    
    # Plot the initial condition
    plt.plot(grid.x, initial_condition, **STYLES["Initial"])
    
    # Plot each numerical solution
    for label, T_data in numerical_solutions.items():
        style = STYLES.get(label, {}) # Get style from dict or empty if not found
        plt.plot(grid.x, T_data, label=f'{label} (t={t_final:.2f})', **style)
        
    # Plot the exact solution if provided
    if exact_solution is not None:
        plt.plot(grid.x, exact_solution, **STYLES["Exact"])
        
    plt.title('1D Heat Equation with Source Term')
    plt.xlabel('Position (x)')
    plt.ylabel('Temperature (T)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
