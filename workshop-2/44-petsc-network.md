# Stabilizing a power grid with PETSc

## Background

- Modern towns and cities rely on electrical grids. When a feeder line fails or demand spikes, operators must reroute power. 

- Grid engineers solve large sparse systems that model how voltages distribute across the network. 

- As grids grow to millions of nodes (smart meters, EV chargers, rooftop solar), analyzing the network in real time requires MPI-scale solvers.

## Network Model

- We treat the grid as a graph: nodes represent buses or substations, edges represent transmission lines with conductance \(g_{ij}\). 
- The unknowns are the voltage (electrical pressure) at every node, measured in per-unit. 
- Solving the equations tells us how the grid redistributes power after disturbances. 
- Disturbances include feeder outages, lightning strikes, surges in EV charging demand, or generators tripping offline—events that force the network to reroute power safely.

Kirchhoff's current law for each node \(i\) is
$$
\sum_{j \in \mathcal{N}(i)} g_{ij} (v_i - v_j) = p_i,
$$
where \(v\) is node voltage (per-unit) and \(p_i\) is the net injection (generation positive, load negative). This assembles into the graph Laplacian system
$$
L \, v = p,
$$
with $L_{ii} = \sum_j g_{ij}$ and $L_{ij} = -g_{ij}$ for $i \neq j$. We fix one or more slack buses (reference voltages) to make the system well-posed.

## PETSc for Graph Systems

- Graph Laplacians are sparse and often ill-conditioned. 
- PETSc's `KSP` solvers with algebraic multigrid or incomplete factorization preconditioners scale to very large networks. 
- MPI distribution lets you partition the graph and solve voltages for continental-scale grids in near real time. 
- The `petsc4py` interface lets domain engineers script prototypes quickly while delegating heavy computation to PETSc.

## Demo Script

The script `examples/11_petsc_graph_laplacian.py` builds a synthetic city grid (streets × avenues), injects power at generation nodes, and sinks power at neighborhoods. The steps are:
1. Generate the Laplacian \(L\) from the grid graph.
2. Impose a reference voltage at the main substation and a ground at the city perimeter.
3. Solve \(L v = p\) using PETSc's conjugate gradient with GAMG.
4. Report voltages and line currents, showing how power flows around a failed line.

The same scaffolding extends to real utility data (e.g., MATPOWER cases, CIM models).

## Scaling Notes

- Increase grid resolution or import GIS-based line data to reach tens of millions of nodes. 
- Couple with dynamic simulations (faults, renewables) by repeatedly solving updated Laplacians. 
- Command-line options `--nx` and `--ny` describe how many "streets" (east-west lines) and "avenues" (north-south lines) we discretize, offering a convenient way to picture the synthetic city grid layout. 
- For example, a run with `mpirun -n 64 python examples/11_petsc_graph_laplacian.py --nx 200 --ny 200 --load 0.8` would require a HPC cluster.

- Real-world grids are unstructured: lines weave irregularly through geography. 
- Swap the synthetic mesh for your actual line list or GIS/CIM data when assembling \(L\); the PETSc solver workflow stays identical. 
- GIS (geographic information system) layers capture where lines, substations, and loads sit in the city. 
- CIM (Common Information Model) is the utility data standard that stores the same equipment and electrical parameters. 
- Both GIS and CIM data can be fed directly into the Laplacian assembly.
