# Earthquake Ground Motion with PETSc

## Background

- Earthquake early-warning systems and structural engineers need fast forecasts of how seismic waves will shake cities seconds after a fault ruptures. 
- Full-resolution crust models contain millions of cells, making wave propagation simulation on laptops infeasible. 
- Clusters with MPI and scalable solvers are standard. 
- PETSc's distributed data structures and time integrators let us march the wave equation over large domains quickly, even with complex material maps.

## Physics Model

We use a scalar acoustic approximation of seismic waves:
$$
\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u + s(\mathbf{x}, t),
$$
where $u(\mathbf{x}, t)$ is ground displacement, $c$ is wave speed, and $s$ is the source (fault slip). In this 2D acoustic demo, $u$ represents a single out-of-plane (vertical) displacement. Full elastic solvers track three displacement components but follow the same parallel workflow. We rewrite this as leapfrog time stepping:
$$
u^{n+1} = 2u^{n} - u^{n-1} + (c \Delta t)^2 \nabla^2 u^{n} + (\Delta t)^2 s^{n}.
$$
Absorbing boundary layers mimic the Earth soaking up energy so reflections do not pollute the solution.

## PETSc Implementation

The script `examples/12_petsc_wave_propagation.py` uses a PETSc DMDA to distribute a 2D grid (tens of thousands of points) across MPI ranks. Each step:
1. Convert the global field to a local array with halos.
2. Compute a finite-difference Laplacian.
3. Update displacement using the leapfrog scheme plus a tapered source term.
4. Apply damping near the boundaries to absorb outgoing waves.

- PETSc vectors store the wavefield, and MPI handles halo exchanges automatically. 

- Real hazard models use unstructured meshes conforming to true topography and bedrock. 

- Swap the structured DMDA for a DMPlex or imported mesh and the solver flow is unchanged.

## Scaling Highlights

- Increase `--nx` and `--ny` to cover whole metropolitan regions at 5–10 m resolution. 
- Coupled problems (elasticity, attenuation) simply extend the state vectors. 
- A run with `mpirun -n 256 python examples/12_petsc_wave_propagation.py --nx 1024 --ny 1024 --steps 3000` would require a HPC cluster.
