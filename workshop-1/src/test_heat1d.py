# test_heat1d.py
# pytest demo. run with this command:
# $ pytest -v test_heat1d.py

import pytest
import numpy as np

# Import the classes to be tested from the extended solver script
from heat1d_ext import (
    Grid,
    HeatSolver,
    SecondOrderCentral,
    FourthOrderHybrid
)

# --- 1. Pytest Fixtures ---
@pytest.fixture
def default_grid():
    """Provides a default Grid object for tests."""
    return Grid(nx=21, L=1.0)

# --- 2. Unit Tests for Individual Components ---

def test_grid_creation(default_grid):
    """Tests that the Grid object is initialized with the correct properties."""
    assert default_grid.nx == 21
    assert default_grid.L == 1.0
    assert len(default_grid.x) == 21
    assert default_grid.dx == pytest.approx(0.05)

def test_second_order_fd_scheme(default_grid):
    """
    Tests the 2nd-order finite difference scheme with a known, simple case.
    A parabola T = x^2 has a constant second derivative of 2.
    """
    T = default_grid.x**2
    fd_scheme = SecondOrderCentral()
    d2T_dx2 = fd_scheme.calculate_d2T_dx2(T, default_grid.dx)
    
    # Test the center point for accuracy
    center_index = default_grid.nx // 2
    assert d2T_dx2[center_index] == pytest.approx(2.0)

def test_fourth_order_hybrid_scheme(default_grid):
    """
    Tests the 4th-order hybrid scheme. For a parabola T=x^2, the
    second derivative should be exactly 2 everywhere.
    """
    T = default_grid.x**2
    fd_scheme = FourthOrderHybrid()
    d2T_dx2 = fd_scheme.calculate_d2T_dx2(T, default_grid.dx)
    
    # The hybrid scheme should be exact for a parabola over the whole interior
    np.testing.assert_allclose(d2T_dx2[1:-1], 2.0, rtol=1e-10)

# --- 3. Integration and Property Tests for the Solver ---

def test_solver_initialization(default_grid):
    """Tests that the solver correctly stores the initial condition."""
    solver = HeatSolver(default_grid, alpha=0.01, fd_scheme=SecondOrderCentral())
    initial_T = np.sin(np.pi * default_grid.x)
    solver.set_initial_condition(initial_T)
    np.testing.assert_array_equal(solver.T, initial_T)

def test_boundary_conditions_are_enforced(default_grid):
    """
    This is a property test. It verifies that the zero-temperature
    boundary condition is maintained throughout the simulation.
    """
    solver = HeatSolver(default_grid, alpha=0.01, fd_scheme=SecondOrderCentral())
    initial_T = np.sin(np.pi * default_grid.x)
    solver.set_initial_condition(initial_T)
    
    # Run for a few steps
    solver.solve(nt=10, dt=0.001)
    
    # Assert that the boundary values are still zero
    assert solver.T[0] == 0.0
    assert solver.T[-1] == 0.0
