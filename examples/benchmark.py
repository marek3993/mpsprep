"""Compare plain Aer MPS with mpsprep on a QAOA circuit on a 2D grid with shuffled qubit labels.

    python examples/benchmark.py [n_rows n_cols seed timeout_s]
Plain Aer runs in a separate process and is killed after timeout_s (default 300).
"""
import multiprocessing as mp
import random
import sys
import time

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from mpsprep import prepare

OPTS = dict(method="matrix_product_state", max_parallel_threads=1, mps_omp_threads=1)


def qaoa_grid_shuffled(r, c, seed, p=2):
    rng = random.Random(seed)
    n = r * c
    perm = list(range(n)); rng.shuffle(perm)
    edges = [(perm[i * c + j], perm[i * c + j + 1]) for i in range(r) for j in range(c - 1)]
    edges += [(perm[i * c + j], perm[(i + 1) * c + j]) for i in range(r - 1) for j in range(c)]
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(p):
        g, b = rng.uniform(0.2, 1.2), rng.uniform(0.2, 1.2)
        for u, v in edges:
            qc.rzz(2 * g, u, v)
        qc.rx(2 * b, range(n))
    return qc


def run(qc, qubits, out):
    c = qc.copy()
    c.save_probabilities(qubits, label="p")
    t = time.perf_counter()
    AerSimulator(**OPTS).run(c).result()
    out.put(time.perf_counter() - t)


def timed(qc, qubits, timeout):
    q = mp.Queue()
    pr = mp.Process(target=run, args=(qc, qubits, q))
    pr.start(); pr.join(timeout)
    if pr.is_alive():
        pr.kill(); return None
    return q.get()


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]] + [4, 8, 1, 300][len(sys.argv) - 1:]
    r, c, seed, tmo = args[:4]
    qc = qaoa_grid_shuffled(r, c, seed)
    qubits = [0, 1, 2, 3]
    t = timed(qc, qubits, tmo)
    print(f"plain Aer MPS ({qc.num_qubits} qubits): " + (f"{t:.3f} s" if t is not None else f"> {tmo} s"))
    for cone in (False, True):
        p = prepare(qc, qubits, seed=seed, light_cone_reduction=cone)
        t = timed(p.circuit, p.positions, tmo)
        print(f"mpsprep, light cone={cone}: prepare {p.info['prepare_s']:.3f} s, run "
              + (f"{t:.3f} s" if t is not None else f"> {tmo} s")
              + f"  ({p.circuit.num_qubits} qubits, {p.method}, swaps {p.info['swaps']})")
