import importlib.util
import os
import numpy as np
import pytest
from petsc4py import PETSc

# Load the example module by file path (examples is not a python package here)
HERE = os.path.dirname(os.path.dirname(__file__))
module_path = os.path.join(HERE, "examples", "08_petsc_tao_t2_fit.py")
spec = importlib.util.spec_from_file_location("tao_example", module_path)
tao_example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tao_example)


def make_seq_vec(arr: np.ndarray) -> PETSc.Vec:
    v = PETSc.Vec().createSeq(len(arr))
    v.setArray(arr.copy())
    return v


def test_objective_gradient_at_true_params():
    # Synthetic data reproduced from the example
    times_ms = np.array([10, 20, 40, 60, 80, 120, 160, 200, 240, 280], dtype=np.float64)
    true_amp = 1.0
    true_t2 = 85.0
    # use noiseless signal for unit test so true parameters give zero gradient
    noise = np.zeros(times_ms.size, dtype=np.float64)
    signal = true_amp * np.exp(-times_ms / true_t2) + noise

    obj = tao_example.T2Objective(times_ms, signal)

    # create distributed-semantics vectors but run single-process with SEQ Vecs
    x = PETSc.Vec().createSeq(2)
    x.setValues([0, 1], [true_amp, true_t2])
    x.assemble()

    g = PETSc.Vec().createSeq(2)
    g.set(0.0)

    f = obj(None, x, g)

    # objective should be small (near zero) and gradient near zero at the true params
    assert f >= 0.0
    assert f < 1e-2

    grad = g.getArray()
    # Expect gradient components near zero (machine tolerance)
    assert np.allclose(grad, 0.0, atol=1e-8)
