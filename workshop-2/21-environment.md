# Computing Environment Setup

## What You Need

- **MPI implementation** (MPICH, Open MPI, Intel MPI, MVAPICH2) with compiler wrappers (`mpicc`, `mpiexec`). I will use MPICH in this workshop.

- **Python 3.9+** with virtual environments (`venv` or `conda`).

- **mpi4py** to access MPI from Python.

- **PETSc + petsc4py** for scalable linear algebra, nonlinear solvers.

- **SLEPc + slepc4py** for eigenvalue problems.

- **TAO** for optimization problems (part of PETSc).

- Recommended extras: BLAS/LAPACK (OpenBLAS, MKL), CMake, Git, compilers (gcc/gfortran or clang/flang).

Keep everything inside a virtual environment (`python -m venv .venv`) so upgrades do not disturb system Python.

## Quick Workflow (All Platforms)

1. Install or load an MPI implementation.
2. Create and activate a Python virtual environment.
3. Upgrade `pip`, then install `mpi4py`.
4. Install PETSc/SLEPc (via package manager or source) and matching `petsc4py`/`slepc4py`.
5. Verify with short MPI and PETSc test programs.

The sections below give concrete commands for each operating system.

---

## Installation

To quickly and conveniently set up a development environment on your laptop computer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install mpi4py mpi4py-fft petsc petsc4py slepc slepc4py
```

For production runs on HPC clusters, a more optimized installation should be considered.

---

## Docker

Use Docker when you want a reproducible environment on any host with container support. You can run Docker on your laptop computer or in the cloud environment. You can convert your Docker "image" to [Singularity/Apptainer](https://apptainer.org/docs/user/main/docker_and_oci.html) for portability to run the HPC cluster.

Install Docker Engine or Docker Desktop first:

- Windows: https://docs.docker.com/desktop/install/windows-install/
- macOS: https://docs.docker.com/desktop/install/mac-install/
- Linux: https://docs.docker.com/engine/install/

```dockerfile
# docker/Dockerfile
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    VENV_PATH=/opt/hpc-venv

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        build-essential gfortran cmake git \
        mpich libmpich-dev \
        libblas-dev liblapack-dev libopenblas-dev \
        petsc-dev slepc-dev && \
    python3 -m venv ${VENV_PATH} && \
    ${VENV_PATH}/bin/pip install --upgrade pip && \
    ${VENV_PATH}/bin/pip install mpi4py petsc4py slepc4py && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="${VENV_PATH}/bin:${PATH}"
CMD ["bash"]
```

Build and launch the image from the repository root:

```bash
docker build -f docker/Dockerfile -t advanced-python-hpc .
docker run --rm -it --name hpc \
    --mount type=bind,source="$PWD",target=/workspace \
    advanced-python-hpc
```

Inside the container, your project appears at `/workspace`; the virtual environment is already active via `PATH`. MPI programs run with `mpiexec` as usual. Adjust `docker run` mount and port flags if you need to expose data directories or Jupyter notebooks.

Note: in the `--mount` flag, `source` points to the host directory (here your current working directory), while `target` is where that directory is mounted inside the container (`/workspace`).

---

## Verification Commands

```bash
$ mpirun -n 2 python tests/test_mpi.py
Hello from rank 0 / 2
Hello from rank 1 / 2
```

```bash
$ mpirun -n 1 python tests/test_slepc.py
PETSc: (3, 24, 0)
SLEPc: 3.24.0
```

```bash
$ mpirun -n 1 python tests/test_tao.py
TAO: lmvm available
```

```bash
$ mpirun -n 4 python tests/test_petsc.py
Rank 0 sum = 8.0
Rank 1 sum = 8.0
Rank 2 sum = 8.0
Rank 3 sum = 8.0
```

