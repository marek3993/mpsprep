"""Regression tests for problems found in the independent audit of commit a406644 (19. 10. 2026).

    python tests/test_regressions.py        or        python -m pytest tests
"""
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from mpsprep import prepare, probabilities

TOL = 1e-8


def raises(exc, f, *a, **k):
    try:
        f(*a, **k)
    except exc:
        return True
    raise AssertionError(f"{f.__name__}{a} did not raise {exc.__name__}")


def test_reset_rejected():
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.reset(0)
    raises(ValueError, probabilities, qc, [0, 1])          # was: one random branch, e.g. [1, 0, 0, 0]


def test_mid_circuit_measurement_rejected():
    qc = QuantumCircuit(2, 1)
    qc.h(0); qc.measure(0, 0); qc.cx(0, 1)
    raises(ValueError, probabilities, qc, [1])


def test_final_measurements_still_accepted():
    qc = QuantumCircuit(3, 3)
    qc.h(0); qc.cx(0, 2); qc.measure([0, 1, 2], [0, 1, 2])
    ref = Statevector(qc.remove_final_measurements(inplace=False)).probabilities([2])
    assert np.max(np.abs(probabilities(qc, [2]) - ref)) < TOL


def test_bad_qubit_indices_rejected():
    qc = QuantumCircuit(3)
    qc.h(0); qc.cx(0, 2)
    raises(ValueError, probabilities, qc, [-1])             # was: silently [1, 0] instead of [0.5, 0.5]
    raises(ValueError, probabilities, qc, [3])
    raises(ValueError, probabilities, qc, [0, 0])
    raises(ValueError, probabilities, qc, [])
    raises(TypeError, probabilities, qc, [1.0])
    assert np.max(np.abs(probabilities(qc, [2]) - [0.5, 0.5])) < TOL


def test_swap_metric_counts_original_swaps():
    qc = QuantumCircuit(4)
    qc.h(range(4))
    for _ in range(5):
        qc.swap(0, 1)
    qc.cx(0, 3); qc.rzz(0.3, 1, 2)
    p = prepare(qc, None)
    d = p.info["swaps_detail"]
    assert d["original"] == 5
    assert p.info["swaps"]["rcm+look"] == d["original"] + d["rcm+look_inserted"]
    assert p.circuit.count_ops().get("swap", 0) == p.info["swaps"][p.method]   # reported = actually run
    ref = Statevector(qc).probabilities()
    assert np.max(np.abs(probabilities(qc) - ref)) < TOL


def test_tiny_angles():
    qc = QuantumCircuit(3)
    qc.h(0); qc.ry(1e-6, 0); qc.cx(0, 1); qc.rzz(1e-7, 1, 2); qc.ry(3e-6, 2)
    for q in ([0], [0, 1, 2]):
        ref = Statevector(qc).probabilities(q)
        assert np.max(np.abs(probabilities(qc, q) - ref)) < 1e-12, q


def test_benchmark_worker_error_does_not_hang():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import benchmark
    qc = QuantumCircuit(2)
    qc.h(0)
    t = time.perf_counter()
    status, msg = benchmark.timed(qc, [5], timeout=60)          # qubit 5 does not exist -> worker error
    assert status == "error", (status, msg)
    assert time.perf_counter() - t < 60


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
