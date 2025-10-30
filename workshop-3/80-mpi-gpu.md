# MPI + GPU

This document explains how to combine MPI (Message Passing Interface) with GPU computing for distributed parallel computing across multiple GPUs and nodes. We cover device mapping strategies and provide practical examples with Numba, CuPy, CUDA Python, and JAX.


## Overview

MPI is the standard for distributed-memory parallel computing, allowing processes on different nodes to communicate. Combining MPI with GPU computing enables scaling beyond a single GPU to entire clusters of GPUs.

**Key Concepts:**

- **MPI Process (Rank)**: Independent process running on a CPU core
- **GPU Device**: Physical GPU that can be accessed by one or more MPI ranks
- **Device Mapping**: Assigning specific GPUs to specific MPI ranks
- **GPU-Aware MPI**: MPI implementations that can directly transfer GPU memory between nodes

**Common Scenarios:**

- Single node, multiple GPUs (1 rank per GPU)
- Multiple nodes, multiple GPUs per node (distributed computing)
- Multi-GPU training in machine learning
- Domain decomposition in scientific simulations


## Why MPI + GPU?

**Scaling Beyond Single GPU:**

- Single GPU memory limited (8-80 GB)
- Problems requiring more compute power
- Data-parallel workloads across multiple GPUs
- Large-scale simulations requiring distributed memory

**Typical Use Cases:**

- Climate modeling across hundreds of GPUs
- Molecular dynamics simulations
- Large-scale machine learning (distributed training)
- Computational fluid dynamics on HPC clusters
- Big data processing pipelines

**Performance Benefits:**

- Near-linear scaling for well-suited problems
- Access to aggregate GPU memory across nodes
- Utilize entire HPC clusters efficiently
- Combine MPI parallelism with GPU parallelism


## MPI Basics for GPU Computing

### Essential MPI Concepts

**MPI Communicator:**

- Group of processes that can communicate
- `MPI.COMM_WORLD`: All processes in the job
- Used for collective operations and point-to-point communication

**MPI Rank:**

- Unique identifier for each process (0, 1, 2, ...)
- Each rank typically controls one GPU
- Rank 0 often designated as "master" for I/O

**MPI Size:**

- Total number of processes in communicator
- Typically equals number of GPUs to use

**Basic MPI Operations:**

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD      # Communicator (all processes)
rank = comm.Get_rank()     # This process's rank (0, 1, 2, ...)
size = comm.Get_size()     # Total number of processes

print(f"Rank {rank} of {size}")
```

**Installation:**

```bash
# Install MPI implementation (choose one)
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev openmpi-bin

# macOS
brew install open-mpi

# Install mpi4py (Python bindings)
pip install mpi4py
```

**Running MPI Programs:**

```bash
# Run with 4 processes (4 GPUs)
mpirun -np 4 python my_gpu_script.py

# On HPC cluster with SLURM
srun -n 4 python my_gpu_script.py
```


## GPU Device Selection Strategies

### Strategy 1: One Rank Per GPU (Most Common)

Assign each MPI rank to a unique GPU on the same node.

**Principle:**

- MPI rank 0 → GPU 0
- MPI rank 1 → GPU 1
- MPI rank 2 → GPU 2
- etc.

**Code Pattern:**

```python
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Method 1: Set CUDA_VISIBLE_DEVICES before importing GPU library
os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)

# Now import GPU library - it sees only one GPU
import cupy as cp
# This rank sees GPU 0 (which is actually physical GPU {rank})
```

**When to Use:**

- Single node with multiple GPUs
- Each rank needs dedicated GPU resources
- Simplest approach for most use cases

### Strategy 2: Manual Device Selection

Explicitly select GPU device in code after querying available devices.

**Code Pattern:**

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Query available GPUs
import cupy as cp
n_gpus = cp.cuda.runtime.getDeviceCount()

# Map rank to GPU (handle more ranks than GPUs)
gpu_id = rank % n_gpus

# Select this rank's GPU
cp.cuda.Device(gpu_id).use()

print(f"Rank {rank} using GPU {gpu_id}")
```

**When to Use:**

- More ranks than GPUs (oversubscription)
- Need explicit control over device selection
- Complex mapping schemes

### Strategy 3: Multi-Node Mapping

For clusters with multiple nodes, each with multiple GPUs.

**Principle:**

- Node 0: Ranks 0-3 → GPUs 0-3
- Node 1: Ranks 4-7 → GPUs 0-3
- Node 2: Ranks 8-11 → GPUs 0-3
- etc.

**Code Pattern:**

```python
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Get node-local rank (0-N on each node)
# This requires MPI implementation support
node_local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', rank))

# Map to GPU on this node
import cupy as cp
n_gpus = cp.cuda.runtime.getDeviceCount()
gpu_id = node_local_rank % n_gpus

cp.cuda.Device(gpu_id).use()
print(f"Global rank {rank}, local rank {node_local_rank}, using GPU {gpu_id}")
```

**Environment Variables for Local Rank:**

- OpenMPI: `OMPI_COMM_WORLD_LOCAL_RANK`
- MPICH: `MPI_LOCALRANKID`
- SLURM: `SLURM_LOCALID`

**When to Use:**

- Multi-node HPC clusters
- Need to map ranks to node-local GPUs


## GPU-Aware MPI

**What is GPU-Aware MPI:**

- MPI implementation can directly transfer GPU memory between nodes
- No explicit CPU staging required
- Uses RDMA (Remote Direct Memory Access) and GPUDirect
- Significant performance improvement for GPU-to-GPU communication

**Benefits:**

- Lower latency for GPU data transfers
- Higher bandwidth utilization
- Simplified code (no manual CPU staging)
- Reduced memory copies

**Requirements:**

- GPU-aware MPI implementation (OpenMPI 4.0+, MVAPICH2-GDR, etc.)
- InfiniBand or RoCE network with GPUDirect support
- CUDA-aware MPI compilation

**Checking GPU-Aware MPI:**

```python
from mpi4py import MPI

# Check if MPI is CUDA-aware
print(f"CUDA-aware MPI: {MPI.CUDA_AWARE}")
```

**Non-GPU-Aware vs GPU-Aware:**

```python
import cupy as cp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# GPU array
data_gpu = cp.random.randn(1000000)

if rank == 0:
    # Non-GPU-aware: Must transfer to CPU first
    data_cpu = cp.asnumpy(data_gpu)
    comm.Send(data_cpu, dest=1)
else:
    # Receive on CPU, transfer to GPU
    data_cpu = np.empty(1000000)
    comm.Recv(data_cpu, source=0)
    data_gpu = cp.asarray(data_cpu)

# GPU-aware: Direct GPU-to-GPU transfer
if rank == 0:
    comm.Send(data_gpu, dest=1)  # Send GPU array directly!
else:
    comm.Recv(data_gpu, source=0)  # Receive directly to GPU!
```


## MPI + CuPy

CuPy works seamlessly with MPI for distributed GPU computing.

### Basic Example: Distributed Array Operations

```python
from mpi4py import MPI
import cupy as cp
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Assign GPU to this rank
cp.cuda.Device(rank).use()

# Each rank creates local data on its GPU
local_size = 1000000 // size
local_data = cp.random.randn(local_size, dtype=cp.float32)

# Perform local computation
local_result = cp.sum(local_data ** 2)

# Gather results to rank 0
if rank == 0:
    results = np.empty(size, dtype=np.float32)
else:
    results = None

# Transfer GPU result to CPU for MPI communication
local_result_cpu = float(local_result.get())
comm.Gather(local_result_cpu, results, root=0)

if rank == 0:
    total = np.sum(results)
    print(f"Total sum of squares: {total}")
```

### Distributed Matrix Multiplication

```python
from mpi4py import MPI
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Select GPU
cp.cuda.Device(rank).use()

# Matrix dimensions
N = 10000
M = N // size  # Each rank gets M rows

# Rank 0 creates and distributes matrix A
if rank == 0:
    A = cp.random.randn(N, N, dtype=cp.float32)
    A_splits = cp.split(A, size, axis=0)
else:
    A_splits = None

# Scatter rows of A to all ranks
local_A = comm.scatter(A_splits, root=0)
if rank != 0:
    local_A = cp.asarray(local_A)  # Convert to CuPy array

# All ranks need full B matrix (replicate)
if rank == 0:
    B = cp.random.randn(N, N, dtype=cp.float32)
    B_cpu = cp.asnumpy(B)
else:
    B_cpu = None

B_cpu = comm.bcast(B_cpu, root=0)
B = cp.asarray(B_cpu)

# Each rank computes its portion: C_local = A_local @ B
local_C = local_A @ B

# Gather results
C_parts = comm.gather(cp.asnumpy(local_C), root=0)

if rank == 0:
    C = cp.asarray(np.vstack(C_parts))
    print(f"Result shape: {C.shape}")
```

### GPU-Aware MPI with CuPy

```python
from mpi4py import MPI
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cp.cuda.Device(rank).use()

# Create GPU data
data = cp.arange(1000000, dtype=cp.float32) * rank

if MPI.CUDA_AWARE:
    # GPU-aware: Direct GPU-to-GPU communication
    if rank == 0:
        # Send GPU array directly
        comm.Send(data, dest=1, tag=11)
    elif rank == 1:
        # Receive directly to GPU
        received = cp.empty_like(data)
        comm.Recv(received, source=0, tag=11)
        print(f"Rank 1 received data from rank 0: {received[:5]}")
else:
    # Non-GPU-aware: Stage through CPU
    if rank == 0:
        comm.Send(cp.asnumpy(data), dest=1, tag=11)
    elif rank == 1:
        data_cpu = np.empty(1000000, dtype=np.float32)
        comm.Recv(data_cpu, source=0, tag=11)
        received = cp.asarray(data_cpu)
```


## MPI + Numba

Numba's CUDA support works well with MPI for custom GPU kernels in distributed settings.

### Basic Example: Distributed Custom Kernel

```python
from mpi4py import MPI
from numba import cuda
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Select GPU for this rank
cuda.select_device(rank)

# Define CUDA kernel
@cuda.jit
def square_kernel(data):
    idx = cuda.grid(1)
    if idx < data.size:
        data[idx] = data[idx] ** 2

# Each rank processes local data
local_size = 1000000 // size
local_data = np.random.randn(local_size).astype(np.float32)

# Transfer to GPU
d_data = cuda.to_device(local_data)

# Launch kernel
threads_per_block = 256
blocks_per_grid = (local_size + threads_per_block - 1) // threads_per_block
square_kernel[blocks_per_grid, threads_per_block](d_data)

# Transfer back to CPU
result = d_data.copy_to_host()

# Compute local sum
local_sum = np.sum(result)

# Reduce across all ranks
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Total sum: {total_sum}")
```

### Multi-GPU Monte Carlo Simulation

```python
from mpi4py import MPI
from numba import cuda
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Select GPU
cuda.select_device(rank)

@cuda.jit
def monte_carlo_pi(n_samples, results):
    """Each thread generates samples and counts hits"""
    idx = cuda.grid(1)

    # Simple RNG (not cryptographically secure)
    seed = idx + rank * 1000000

    count = 0
    for i in range(n_samples):
        # Generate random point
        x = (seed * 1103515245 + 12345) % 2147483648 / 2147483648.0
        seed = (seed * 1103515245 + 12345) % 2147483648
        y = (seed * 1103515245 + 12345) % 2147483648 / 2147483648.0
        seed = (seed * 1103515245 + 12345) % 2147483648

        # Check if inside circle
        if x*x + y*y <= 1.0:
            count += 1

    results[idx] = count

# Each rank does local computation
threads = 1024
samples_per_thread = 10000

d_results = cuda.device_array(threads, dtype=np.int32)
monte_carlo_pi[1, threads](samples_per_thread, d_results)

# Transfer results back
results = d_results.copy_to_host()
local_hits = np.sum(results)
local_samples = threads * samples_per_thread

# Reduce across all ranks
total_hits = comm.reduce(local_hits, op=MPI.SUM, root=0)
total_samples = comm.reduce(local_samples, op=MPI.SUM, root=0)

if rank == 0:
    pi_estimate = 4.0 * total_hits / total_samples
    print(f"Pi estimate: {pi_estimate}")
    print(f"Total samples: {total_samples}")
```


## MPI + JAX

JAX provides excellent support for multi-GPU computing through `pmap` and explicit device management.

### Basic Example: Data Parallel with pmap

```python
from mpi4py import MPI
import jax
import jax.numpy as jnp
from jax import pmap

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# JAX sees all local GPUs
n_local_devices = jax.local_device_count()
print(f"Rank {rank}: {n_local_devices} local devices")

# Define computation
def compute(x):
    return jnp.sum(x ** 2)

# Create data on each device
local_data = jnp.ones((n_local_devices, 1000000))

# pmap automatically distributes across local devices
parallel_compute = pmap(compute)
local_results = parallel_compute(local_data)

# Sum local results
local_sum = jnp.sum(local_results)

# MPI reduce across nodes
total_sum = comm.reduce(float(local_sum), op=MPI.SUM, root=0)

if rank == 0:
    print(f"Total sum: {total_sum}")
```

### Distributed Training Example

```python
from mpi4py import MPI
import jax
import jax.numpy as jnp
from jax import grad, jit, pmap

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Model and loss function
def predict(params, x):
    return params['w'] @ x + params['b']

def loss(params, x, y):
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)

# Gradient function
grad_fn = jit(grad(loss))

# Initialize parameters on each rank
key = jax.random.PRNGKey(rank)
params = {
    'w': jax.random.normal(key, (10, 100)),
    'b': jnp.zeros(10)
}

# Each rank has local training data
x_local = jax.random.normal(key, (1000, 100))
y_local = jax.random.normal(key, (1000, 10))

# Training loop
n_steps = 100
lr = 0.01

for step in range(n_steps):
    # Compute local gradients
    grads = grad_fn(params, x_local, y_local)

    # Average gradients across all ranks
    grads_w = comm.allreduce(grads['w'], op=MPI.SUM) / comm.Get_size()
    grads_b = comm.allreduce(grads['b'], op=MPI.SUM) / comm.Get_size()

    # Update parameters
    params['w'] -= lr * grads_w
    params['b'] -= lr * grads_b

    if rank == 0 and step % 10 == 0:
        current_loss = loss(params, x_local, y_local)
        print(f"Step {step}, Loss: {current_loss:.4f}")
```

### Multi-Node pmap (Advanced)

```python
from mpi4py import MPI
import jax
import jax.numpy as jnp
from jax.experimental import maps

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Configure JAX for multi-host
jax.distributed.initialize(
    coordinator_address=f"localhost:{8476 + rank}",
    num_processes=size,
    process_id=rank
)

# Now pmap can work across nodes
@jax.pmap
def distributed_compute(x):
    return x ** 2 + jnp.sum(x)

# Create data sharded across all devices (including remote)
n_devices = jax.device_count()  # Total across all nodes
data = jnp.ones((n_devices, 1000))

result = distributed_compute(data)
print(f"Rank {rank}: Result shape {result.shape}")
```


## MPI + CUDA Python

CUDA Python provides low-level control for MPI + GPU scenarios.

### Basic Example: Manual Device Management

```python
from mpi4py import MPI
from cuda import cuda
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Initialize CUDA
err, = cuda.cuInit(0)
assert err == cuda.CUresult.CUDA_SUCCESS

# Get device for this rank
err, device = cuda.cuDeviceGet(rank)
assert err == cuda.CUresult.CUDA_SUCCESS

# Create context
err, context = cuda.cuCtxCreate(0, device)
assert err == cuda.CUresult.CUDA_SUCCESS

print(f"Rank {rank} using device {rank}")

# Allocate device memory
n = 1000000
nbytes = n * 4  # float32

err, d_data = cuda.cuMemAlloc(nbytes)
assert err == cuda.CUresult.CUDA_SUCCESS

# Create host data
h_data = np.random.randn(n).astype(np.float32)

# Copy host to device
err, = cuda.cuMemcpyHtoD(d_data, h_data.ctypes.data, nbytes)
assert err == cuda.CUresult.CUDA_SUCCESS

# Process on GPU (simplified - would load and run kernel here)

# Copy device to host
result = np.empty(n, dtype=np.float32)
err, = cuda.cuMemcpyDtoH(result.ctypes.data, d_data, nbytes)
assert err == cuda.CUresult.CUDA_SUCCESS

# Compute local sum
local_sum = np.sum(result)

# Reduce across ranks
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Total sum: {total_sum}")

# Cleanup
err, = cuda.cuMemFree(d_data)
err, = cuda.cuCtxDestroy(context)
```

### GPU-Aware MPI with CUDA Python

```python
from mpi4py import MPI
from cuda import cuda, cudart
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Initialize CUDA
err, = cudart.cudaSetDevice(rank)
assert err == cudart.cudaError_t.cudaSuccess

# Allocate device memory
n = 1000000
err, d_data = cudart.cudaMalloc(n * 4)
assert err == cudart.cudaError_t.cudaSuccess

# Initialize data on device
h_data = (np.arange(n) * rank).astype(np.float32)
err, = cudart.cudaMemcpy(d_data, h_data.ctypes.data, n * 4,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

if MPI.CUDA_AWARE and rank == 0:
    # Send GPU pointer directly
    comm.Send([d_data, MPI.FLOAT], dest=1, tag=11)
elif MPI.CUDA_AWARE and rank == 1:
    # Receive directly to GPU
    err, d_recv = cudart.cudaMalloc(n * 4)
    comm.Recv([d_recv, MPI.FLOAT], source=0, tag=11)

    # Copy back to verify
    result = np.empty(n, dtype=np.float32)
    err, = cudart.cudaMemcpy(result.ctypes.data, d_recv, n * 4,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    print(f"Rank 1 received: {result[:5]}")

    err, = cudart.cudaFree(d_recv)

# Cleanup
err, = cudart.cudaFree(d_data)
```


## Best Practices

### Device Selection

**Always Select Device Early:**

```python
# Bad: Import before device selection
import cupy as cp
cp.cuda.Device(rank).use()  # May be too late

# Good: Select device first
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
import cupy as cp  # Now sees only assigned GPU
```

**Check Device Assignment:**

```python
print(f"Rank {rank} on {socket.gethostname()}, GPU {cp.cuda.runtime.getDevice()}")
comm.Barrier()  # Synchronize before continuing
```

### Communication Patterns

**Minimize Communication:**

```python
# Bad: Communicate every iteration
for i in range(1000):
    local_result = compute()
    global_sum = comm.allreduce(local_result)  # Expensive!

# Good: Batch communications
local_results = []
for i in range(1000):
    local_results.append(compute())
global_sum = comm.allreduce(sum(local_results))  # Once
```

**Use Collective Operations:**

```python
# Bad: Manual gathering
if rank == 0:
    results = [local_result]
    for i in range(1, size):
        results.append(comm.recv(source=i))

# Good: Use MPI collective
results = comm.gather(local_result, root=0)
```

**Overlap Computation and Communication:**

```python
# Use non-blocking communication
req = comm.Isend(data, dest=target)
# Do other computation while sending
other_work()
req.Wait()  # Wait for send to complete
```

### Memory Management

**Reuse GPU Memory:**

```python
# Bad: Allocate every iteration
for i in range(100):
    temp = cp.empty(1000000)  # Allocate
    compute(temp)
    # temp deallocated

# Good: Allocate once
temp = cp.empty(1000000)
for i in range(100):
    compute(temp)  # Reuse allocation
```

**Free Resources:**

```python
# Ensure cleanup on exit
try:
    # MPI + GPU computation
    pass
finally:
    # Free GPU memory
    del large_arrays
    cp.get_default_memory_pool().free_all_blocks()
```

### Error Handling

**Synchronize on Errors:**

```python
try:
    result = compute()
    error = 0
except Exception as e:
    print(f"Rank {rank} error: {e}")
    error = 1

# All ranks must agree on error state
error = comm.allreduce(error, op=MPI.MAX)
if error:
    comm.Abort(1)  # Terminate all ranks
```

### Performance Monitoring

**Time Communication vs Computation:**

```python
import time

# Computation time
t0 = time.time()
local_result = compute_on_gpu()
comp_time = time.time() - t0

# Communication time
t0 = time.time()
global_result = comm.allreduce(local_result)
comm_time = time.time() - t0

if rank == 0:
    print(f"Computation: {comp_time:.4f}s, Communication: {comm_time:.4f}s")
    print(f"Communication overhead: {100*comm_time/(comp_time+comm_time):.1f}%")
```


## Common Patterns

### Pattern 1: Domain Decomposition

Divide problem domain across GPUs, communicate boundaries.

```python
# Each rank owns a slice of the domain
nx_global = 10000
nx_local = nx_global // size

# Include ghost cells for boundaries
data = cp.zeros(nx_local + 2)  # +2 for left/right ghosts

# Exchange boundaries with neighbors
left_neighbor = (rank - 1) % size
right_neighbor = (rank + 1) % size

# Send right boundary to right neighbor, receive into left ghost
comm.Sendrecv(data[-2], dest=right_neighbor,
              recvbuf=data[0], source=left_neighbor)

# Send left boundary to left neighbor, receive into right ghost
comm.Sendrecv(data[1], dest=left_neighbor,
              recvbuf=data[-1], source=right_neighbor)
```

### Pattern 2: Parallel Reduction

Each rank computes partial result, combine with reduction.

```python
# Each rank computes local partial result
local_result = compute_local()

# Reduce to get global result
global_result = comm.allreduce(local_result, op=MPI.SUM)

# All ranks now have global_result
```

### Pattern 3: Scatter-Compute-Gather

Distribute data, compute, collect results.

```python
if rank == 0:
    # Create global data
    global_data = cp.random.randn(size * 1000000)
    splits = cp.split(global_data, size)
else:
    splits = None

# Scatter to all ranks
local_data = comm.scatter(splits, root=0)
if rank != 0:
    local_data = cp.asarray(local_data)

# Each rank computes
local_result = compute(local_data)

# Gather results
results = comm.gather(cp.asnumpy(local_result), root=0)

if rank == 0:
    final = cp.concatenate([cp.asarray(r) for r in results])
```




## Summary

**Key Takeaways:**

1. **Device Selection**: Always map MPI ranks to GPUs explicitly
2. **Communication**: Minimize and batch MPI communication
3. **GPU-Aware MPI**: Use when available for direct GPU-to-GPU transfers
4. **Framework Choice**:
   - CuPy: Easiest for array-based computations
   - Numba: Best for custom kernels
   - JAX: Excellent for ML with `pmap`
   - CUDA Python: Maximum control, most complex

5. **Scaling**: MPI + GPU enables scaling to hundreds/thousands of GPUs



