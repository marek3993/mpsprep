"""mpsprep must give exactly the Statevector probabilities (tolerance 1e-8).

    python -m pytest tests        or        python tests/test_correctness.py
"""
import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector

from mpsprep import prepare, probabilities

TOL = 1e-8


def qaoa(n, edges, p, rng):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(p):
        g, b = rng.uniform(0.2, 1.2), rng.uniform(0.2, 1.2)
        for u, v in edges:
            qc.rzz(2 * g, u, v)
        qc.rx(2 * b, range(n))
    return qc


def grid_edges(r, c, perm):
    e = []
    for i in range(r):
        for j in range(c):
            q = i * c + j
            if j + 1 < c:
                e.append((perm[q], perm[q + 1]))
            if i + 1 < r:
                e.append((perm[q], perm[q + c]))
    return e


def cases():
    rng = random.Random(7)
    for seed in range(12):  # random circuits with many gate types, incl. 3-qubit gates
        n = 6 + seed % 5
        yield f"random{seed}", random_circuit(n, depth=6, max_operands=3, seed=seed), rng.sample(range(n), 3)
    for seed in range(4):
        perm = list(range(12)); random.Random(seed).shuffle(perm)
        yield f"grid_shuffled{seed}", qaoa(12, grid_edges(3, 4, perm), 2, rng), [0, 1, 2, 3]
    qc = QuantumCircuit(10)  # far-apart gates
    for k in range(10):
        qc.ry(0.3 + k, k)
    for a, b in [(0, 9), (2, 7), (9, 1), (4, 0), (3, 8)]:
        qc.cx(a, b)
        qc.cp(0.7, b, a)
    yield "long_range", qc, [9, 0, 5]


def check_one(name, qc, qubits):
    worst = 0.0
    for q in (qubits, None):
        ref = Statevector(qc).probabilities(q)
        got = probabilities(qc, q)
        worst = max(worst, float(np.max(np.abs(ref - got))))
    assert worst < TOL, (name, worst)
    return worst


def test_all():
    for name, qc, qubits in cases():
        check_one(name, qc, qubits)


def test_nearest_neighbour_only():
    for name, qc, qubits in cases():
        p = prepare(qc, qubits)
        for ins in p.circuit.data:
            qs = [p.circuit.find_bit(q).index for q in ins.qubits]
            assert len(qs) == 1 or abs(qs[0] - qs[1]) == 1, (name, ins.operation.name, qs)


if __name__ == "__main__":
    w = 0.0
    for name, qc, qubits in cases():
        w = max(w, check_one(name, qc, qubits))
    test_nearest_neighbour_only()
    print(f"all cases ok, worst |p - p_statevector| = {w:.1e}")
