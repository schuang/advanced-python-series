# Solving nonlinear system of equations with PETSc

## Background

- Lithium-ion battery packs power laptops, phones, grid storage, and electric vehicles, making safety and reliability important. 
- A **cell** is the smallest electrochemical unit: anode, cathode, separator, and electrolyte sealed in a can or pouch delivering a few volts. 
- A battery pack groups hundreds to thousands of cells, their electrical connections, thermal pathways, and battery-management electronics into one enclosure.

- Lithium-ion chemistry stores energy by shuttling lithium ions between electrode materials. 
- Malfunctioning cells can trigger exothermic reactions. 
- When those reactions heat the pack faster than thermal systems can remove energy, **thermal runaway** begins: each temperature rise accelerates more reactions, releasing even more heat, potentially leading to venting or fire.

- Predicting the onset requires solving a temperature field coupled with reaction kinetics, involving nonlinear terms that grow exponentially with temperature. Real packs involve thousands of cells and detailed geometries, leading to millions of unknowns and requiring MPI-parallel nonlinear solvers.

## Governing Equations

The heat balance with Arrhenius source (Bratu-type model) is
$$
-\nabla \cdot (k \nabla T) = Q_0 \exp\!\left(\frac{E_a}{R (T + T_\mathrm{ref})}\right)
$$
where \(T\) is temperature rise over ambient, \(k\) is effective thermal conductivity, \(Q_0\) is reaction rate prefactor, \(E_a\) activation energy, and \(R\) gas constant. Boundary conditions include Dirichlet on housing and insulated symmetry planes. Discretization (finite differences or finite elements) leads to a nonlinear residual
$$
F_i(T) = -k \sum_{j \in \mathcal{N}(i)} w_{ij}(T_j - T_i) - Q_0 \exp\!\left(\frac{E_a}{R (T_i + T_\mathrm{ref})}\right).
$$
This system is strongly nonlinear (exponential source) and globally coupled. A Bratu problem is the canonical PDE \(-\Delta u - \lambda e^{u} = 0\), capturing diffusion balanced against exponential heat release, mirroring thermal runaway physics.

## PETSc SNES

- PETSc's **SNES** (Scalable Nonlinear Equations Solvers) provides Newton, line-search, and trust-region methods with flexible Jacobians and preconditioners. 
- DMDA (distributed arrays) manages structured grids across MPI ranks, while DMPlex extends to unstructured packs. Combining SNES with scalable Krylov solvers (KSP) and algebraic multigrid preconditioners (PC) handles more than $10^7$ unknowns, fitting real packs.

## petsc4py Implementation Outline

The script `examples/10_petsc_snes_bratu.py` (Bratu as thermal runaway analogue) demonstrates how to:
1. Create a 2D DMDA grid distributed over MPI ranks.
2. Define the nonlinear residual \(F(T)\) and its Jacobian for the exponential source.
3. Configure SNES (Newton with line search) and solve for the steady-state temperature field.
4. Monitor convergence and gather the solution for analysis or visualization.

The script works in dimensionless variables (Bratu form \( -\Delta u - \lambda e^{u} = 0\)) but the scaffolding matches full thermo-chemical models.

## Key PETSc Calls

```python
from petsc4py import PETSc

dm = PETSc.DMDA().create([nx, ny], dof=1, stencil_width=1)
snes = PETSc.SNES().create()
snes.setDM(dm)
snes.setFunction(residual, None)
snes.setJacobian(jacobian, None)
snes.setFromOptions()
snes.solve(None, dm.createGlobalVec())
```

The residual and Jacobian callbacks leverage DMDA local vectors to access stencil values. Customize runtime with options like `-snes_monitor -ksp_type cg -pc_type gamg`.

## Scaling to Real Systems

Increase DMDA dimensions (3D pack) or switch to DMPlex for CAD-derived meshes. Couple thermal field to electrochemical state equations within SNES or via PETSc's multiphysics interfaces. Deploy on clusters with `mpirun -n 256 python examples/10_petsc_snes_bratu.py` to match realistic resolution.
