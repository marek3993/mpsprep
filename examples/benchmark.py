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
    try:
        c = qc.copy()
        c.save_probabilities(qubits, label="p")
        t = time.perf_counter()
        res = AerSimulator(**OPTS).run(c).result()
        dt = time.perf_counter() - t
        out.put(("ok", dt) if res.success else ("error", str(res.status)))
    except Exception as e:  # report the failure instead of leaving the parent waiting
        out.put(("error", f"{type(e).__name__}: {e}"))


def timed(qc, qubits, timeout):
    """Returns ("ok", seconds) | ("timeout", None) | ("error", message)."""
    import queue
    q = mp.Queue()
    pr = mp.Process(target=run, args=(qc, qubits, q))
    pr.start(); pr.join(timeout)
    if pr.is_alive():
        pr.kill(); pr.join()
        return "timeout", None
    try:
        return q.get(timeout=5)
    except queue.Empty:
        return "error", f"worker exited with code {pr.exitcode} without a result"
    finally:
        q.close()


def show(r, tmo):
    status, v = r
    return f"{v:.3f} s" if status == "ok" else (f"> {tmo} s" if status == "timeout" else f"error ({v})")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]] + [4, 8, 1, 300][len(sys.argv) - 1:]
    r, c, seed, tmo = args[:4]
    qc = qaoa_grid_shuffled(r, c, seed)
    qubits = [0, 1, 2, 3]
    print(f"plain Aer MPS ({qc.num_qubits} qubits): " + show(timed(qc, qubits, tmo), tmo))
    for cone in (False, True):
        p = prepare(qc, qubits, seed=seed, light_cone_reduction=cone)
        print(f"mpsprep, light cone={cone}: prepare {p.info['prepare_s']:.3f} s, run "
              + show(timed(p.circuit, p.positions, tmo), tmo)
              + f"  ({p.circuit.num_qubits} qubits, {p.method}, swaps {p.info['swaps']})")
