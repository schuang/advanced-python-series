# heat1d_ext.py
# This version adds a source term to the polymorphic solver, demonstrating
# how the architecture can be extended to handle new physics.

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Grid:
    nx: int
    L: float
    def __post_init__(self):
        self.dx = self.L / (self.nx - 1)
        self.x = np.linspace(0, self.L, self.nx)

class FiniteDifference(ABC):
    @abstractmethod
    def calculate_d2T_dx2(self, T, dx):
        pass

class SecondOrderCentral(FiniteDifference):
    def calculate_d2T_dx2(self, T, dx):
        d2T = np.zeros_like(T)
        d2T[1:-1] = (T[2:] - 2*T[1:-1] + T[0:-2]) / dx**2
        return d2T

class FourthOrderHybrid(FiniteDifference):
    def calculate_d2T_dx2(self, T, dx):
        nx = len(T)
        d2T = np.zeros_like(T)
        
        # --- Interior points (use 4th-order scheme) ---
        for i in range(2, nx - 2):
            d2T[i] = (-T[i+2] + 16*T[i+1] - 30*T[i] + 16*T[i-1] - T[i-2]) / (12 * dx**2)
            
        # --- Boundary-adjacent points (use 2nd-order scheme) ---
        d2T[1] = (T[2] - 2*T[1] + T[0]) / dx**2
        d2T[nx-2] = (T[nx-1] - 2*T[nx-2] + T[nx-3]) / dx**2
        
        return d2T

class HeatSolver:
    def __init__(self, grid, alpha, fd_scheme: FiniteDifference):
        self.grid = grid
        self.alpha = alpha
        self.fd_scheme = fd_scheme
        self.T = np.zeros(grid.nx)

    def set_initial_condition(self, T0):
        self.T = T0.copy()

    def solve(self, nt, dt, source_term=None):
        for _ in range(nt):
            d2T_dx2 = self.fd_scheme.calculate_d2T_dx2(self.T, self.grid.dx)
            
            rhs = self.alpha * d2T_dx2
            if source_term is not None:
                rhs += source_term

            self.T += dt * rhs
            self.T[0], self.T[-1] = 0.0, 0.0
        
        fd_name = self.fd_scheme.__class__.__name__
        print(f"Simulation with Euler / {fd_name} finished.")

from heat_plotter import plot_heat_comparison

if __name__ == "__main__":
    nx = 201
    L = 1.0
    alpha = 0.01
    t_final = 2.0
    dx = L / (nx - 1)
    dt = 0.1 * dx**2 / alpha
    nt = int(t_final / dt)

    grid = Grid(nx=nx, L=L)
    T_initial = np.zeros_like(grid.x) # Initial temperature is zero everywhere
    source = np.sin(2 * np.pi * grid.x)
    
    # --- Solver 1: Pure 2nd-Order ---
    solver1 = HeatSolver(grid, alpha, SecondOrderCentral())
    solver1.set_initial_condition(T_initial.copy())
    solver1.solve(nt=nt, dt=dt, source_term=source)
    
    # --- Solver 2: Hybrid 4th-Order ---
    solver2 = HeatSolver(grid, alpha, FourthOrderHybrid())
    solver2.set_initial_condition(T_initial.copy())
    solver2.solve(nt=nt, dt=dt, source_term=source)

    # --- Analysis ---
    T_exact = (1 / (4 * np.pi**2 * alpha)) * np.sin(2 * np.pi * grid.x) * (1 - np.exp(-4 * np.pi**2 * alpha * t_final))
    
    error1 = np.sqrt(np.sum((solver1.T - T_exact)**2) / nx)
    error2 = np.sqrt(np.sum((solver2.T - T_exact)**2) / nx)

    print(f"\nParameters: nx={nx}, dt={dt:.2e}, t_final={t_final}")
    print("-" * 40)
    print(f"L2 Error (2nd Order Spatial): {error1:.2e}")
    print(f"L2 Error (4th Order Spatial): {error2:.2e}")
    print("-" * 40)
    
    plot_heat_comparison(
        grid=grid,
        t_final=t_final,
        filename='heat1d_ext.png',
        initial_condition=T_initial,
        numerical_solutions={
            "2nd Order": solver1.T,
            "4th Order": solver2.T
        },
        exact_solution=T_exact
    )
