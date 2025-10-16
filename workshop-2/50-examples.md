# Summary of Examples

## Heat Equation: Domain Decomposition
We split the simulation grid across ranks with halo and ghost exchanges. Synchronizing boundary conditions after each timestep is necessary. Scaling considerations include load balance and communication cost. See `05_heat_equation_mpi.py` as a template for diffusion problems.

## Deep Learning: Data Parallelism
We partition the dataset per rank for independent training passes. Synchronizing model updates through collective operations like allreduce is necessary. Handling randomness and reproducibility across processes matters. Reference `06_deep_learning_data_parallel.py` for an end-to-end example.

## Discussion Prompts
Compare domain and data parallel strategies. Explore opportunities for hybrid model and data parallel approaches. Define success metrics such as speedup and accuracy retention.

## PETSc Optimization: MRI T2 Fitting
Calibrate T2 decay parameters from synthetic MRI spin-echo data using PETSc TAO. This illustrates mapping a least-squares objective onto distributed vectors and shows how `petsc4py` exposes TAO quasi-Newton optimizers with minimal boilerplate. Try `mpirun -n 2 python examples/08_petsc_tao_t2_fit.py` and compare recovered vs true parameters.

## PETSc/SLEPc Eigenmodes: Bridge Dynamics
Build stiffness and mass matrices for a rectangular bridge deck with finite differences. Use SLEPc to extract the lowest vibration mode and approximate natural frequency. This demonstrates scalable eigenvalue solves in `petsc4py` and `slepc4py`. Run `mpirun -n 4 python examples/09_petsc_eigen_bridge.py` and inspect the mode profile.

## PETSc SNES: Thermal Runaway (Bratu)
Solve a Bratu-style nonlinear heat equation mimicking battery thermal runaway. Uses SNES with DMDA to distribute the grid and assemble residual and Jacobian in parallel. Try `mpirun -n 4 python examples/10_petsc_snes_bratu.py -nx 128 -ny 64 -lambda 6.0`.

## PETSc Graph Laplacian: Power Network
Model a city-scale power grid as a graph and solve the Laplacian \(L v = p\) for node voltages. This demonstrates PETSc KSP with GAMG for sparse graph systems with Dirichlet constraints. Run `mpirun -n 4 python examples/11_petsc_graph_laplacian.py --nx 80 --ny 40 --load 0.8`.

## PETSc Wave Propagation: Quake Simulation
March the 2D acoustic wave equation to mimic earthquake ground motion across a city basin. Uses DMDA vectors, MPI halo exchange, and explicit leapfrog stepping with absorbing boundaries. Run `mpirun -n 8 python examples/12_petsc_wave_propagation.py --nx 400 --ny 240 --steps 2000`.

## PETSc Data Parallel ML: Logistic Regression
Demonstrates MPI-driven gradient averaging for a logistic classifier across large datasets. Wraps NumPy arrays in PETSc vectors to reuse BLAS updates while sharing gradients via MPI collectives. Run `mpirun -n 16 python examples/13_petsc_data_parallel_gd.py --samples 1000000 --features 128 --epochs 10`.

## PETSc TAO Deep Classifier
Uses TAO with a redundant vector to keep full weights on every rank while averaging gradients via MPI. Shows how to pack and unpack two hidden layers, enabling custom deep-learning research pipelines without heavyweight frameworks. Run `mpirun -n 16 python examples/14_petsc_deep.py --samples 200000 --features 64 --hidden1 256 --hidden2 128`.

## PETSc/MPI Genomics GWAS Scan
Partitions variants across ranks and computes chi-square association statistics for each SNP. Demonstrates how PETSc vectors store distributed scores while MPI reductions find the top hits. Run `mpirun -n 32 python examples/15_petsc_gwas.py --samples 50000 --variants 1000000`.
