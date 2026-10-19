"""mpsprep - prepare Qiskit circuits for the Qiskit Aer matrix-product-state (MPS) simulator.

Two exact (result-preserving) steps before simulation:
  1. light cone: when only some qubits are measured, drop the gates that cannot affect them
     (exact rule, see `light_cone`) and drop the qubits that end up idle;
  2. routing onto a line: Aer MPS applies 2-qubit gates only to neighbouring sites and moves
     qubits without lookahead. We route the circuit ourselves and pick the cheaper of
       - reverse Cuthill-McKee ordering + lookahead routing ("rcm+look"),
       - Qiskit SABRE on a line coupling map ("sabre"),
     by the number of inserted swaps ("guard").

The prepared circuit has only nearest-neighbour 2-qubit gates, so Aer inserts no swaps itself.

    from mpsprep import probabilities
    p = probabilities(qc, qubits=[0, 1, 2, 3])          # same as Statevector(qc).probabilities([0,1,2,3])
"""
from dataclasses import dataclass, field
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

__all__ = ["prepare", "probabilities", "Prepared", "light_cone", "route_lookahead", "rcm_order"]
__version__ = "0.1.0"


@dataclass
class Prepared:
    circuit: "QuantumCircuit"   # physical circuit on a line, nearest-neighbour 2-qubit gates only
    positions: list             # positions[i] = site of the i-th requested qubit in `circuit`
    method: str                 # "sabre" or "rcm+look"
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers
def _ops(qc):
    """(operation, qubit indices) for every instruction; rejects what we cannot route exactly."""
    out = []
    for ins in qc.data:
        op = ins.operation
        if op.name == "barrier":
            continue
        if op.name == "measure" or ins.clbits or getattr(op, "condition", None) is not None:
            raise ValueError("mid-circuit measurements / classical control are not supported; "
                             "pass the circuit without final measurements and use `qubits=`")
        qs = tuple(qc.find_bit(q).index for q in ins.qubits)
        if len(qs) > 2:
            raise ValueError(f"gate {op.name} acts on {len(qs)} qubits; decompose first")
        out.append((op, qs))
    return out


_BASIS = ["cp", "csx", "cx", "cy", "cz", "ecr", "h", "id", "p", "r", "rx", "rxx", "ry", "ryy", "rz",
          "rzx", "rzz", "s", "sdg", "swap", "sx", "sxdg", "t", "tdg", "u", "u1", "u2", "u3", "x", "y", "z"]
# 1- and 2-qubit gates that Aer 0.17 MPS applies natively; everything else is decomposed into them


def _validate(qc):
    """Only unitary gates (and barriers) are supported: reject resets, mid-circuit measurements,
    classical control, delays and other non-unitary instructions instead of returning a wrong answer."""
    from qiskit.circuit import Gate
    for ins in qc.data:
        op = ins.operation
        if op.name == "barrier":
            continue
        if not isinstance(op, Gate) or ins.clbits or getattr(op, "condition", None) is not None:
            raise ValueError(f"unsupported instruction '{op.name}': mpsprep handles unitary circuits only "
                             "(no reset, mid-circuit measurement, classical control or delay); "
                             "final measurements are removed automatically")


def _check_qubits(qubits, n):
    """Requested qubits must be distinct integers 0 <= q < n (negative indices are rejected)."""
    import numbers
    qs = list(qubits)
    if not qs:
        raise ValueError("qubits must not be empty (use None for all qubits)")
    for q in qs:
        if isinstance(q, bool) or not isinstance(q, numbers.Integral):
            raise TypeError(f"qubit index {q!r} is not an integer")
        if not 0 <= q < n:
            raise ValueError(f"qubit index {q} out of range 0..{n - 1} (negative indices are not accepted)")
    if len(set(qs)) != len(qs):
        raise ValueError(f"qubit indices must be distinct, got {qs}")
    return [int(q) for q in qs]


def _unroll(qc):
    from qiskit import transpile
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import RemoveFinalMeasurements
    qc = PassManager([RemoveFinalMeasurements()]).run(qc)
    _validate(qc)
    return transpile(qc, basis_gates=_BASIS, optimization_level=0)


def _build(ops, n, global_phase=0.0):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n, global_phase=global_phase)
    for op, qs in ops:
        qc.append(op, qs)
    return qc


_DIAGONAL = {"id", "z", "s", "sdg", "t", "tdg", "p", "u1", "rz", "cz", "cp", "rzz"}
# gates diagonal in the computational basis (exactly, for every angle)


def light_cone(qc, qubits):
    """Exact backward light cone of measuring `qubits` in the Z basis; idle qubits are dropped.

    Walk the circuit backwards with an active set A (initially `qubits`) and a flag per active qubit:
    "only diagonal gates follow on this qubit". A diagonal gate whose active qubits are all flagged is
    dropped (it only multiplies basis states by phases that the Z-basis marginal cannot see); a gate
    touching no active qubit is dropped; every other gate is kept and its qubits become active
    (flagged only if the kept gate is diagonal). No numerical commutation tolerance is used: Qiskit's
    LightCone pass treats e.g. RY(1e-6) as commuting with Z and drops it (error ~5e-7 in probabilities).

    Returns (reduced circuit on k qubits, new index of each requested qubit, kept original indices)."""
    data = [(ins.operation, tuple(qc.find_bit(q).index for q in ins.qubits)) for ins in qc.data
            if ins.operation.name != "barrier"]
    active = {q: True for q in qubits}          # qubit -> flag
    kept = []
    for op, qs in reversed(data):
        touched = [q for q in qs if q in active]
        if not touched:
            continue
        diag = op.name in _DIAGONAL
        if diag and all(active[q] for q in touched):
            continue
        kept.append((op, qs))
        for q in qs:
            active[q] = diag and active.get(q, True)
    kept.reverse()
    keep = sorted(set(active) | set(qubits))
    new = {q: i for i, q in enumerate(keep)}
    return (_build([(op, tuple(new[q] for q in qs)) for op, qs in kept], len(keep), qc.global_phase),
            [new[q] for q in qubits], keep)


def rcm_order(ops, n):
    """order[site] = qubit; reverse Cuthill-McKee on the 2-qubit interaction graph."""
    rows, cols = [], []
    for _, qs in ops:
        if len(qs) == 2:
            rows += [qs[0], qs[1]]; cols += [qs[1], qs[0]]
    W = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return [int(x) for x in reverse_cuthill_mckee(W, symmetric_mode=True)]


def route_lookahead(ops, n, order, window=20, decay=0.8):
    """Route onto a line starting from `order` (order[site] = qubit). For each distant gate, choose
    the meeting point that minimises swaps now + discounted distance of the next `window` gates.
    Returns (physical ops with explicit swaps, final order, number of INSERTED swaps)."""
    from qiskit.circuit.library import SwapGate
    order = list(order)
    loc = [0] * n
    for s, q in enumerate(order):
        loc[q] = s
    twoq = [i for i, (_, qs) in enumerate(ops) if len(qs) == 2]
    rank = {i: k for k, i in enumerate(twoq)}
    out, nswap = [], 0
    sw = SwapGate()

    def swap(s):
        nonlocal nswap
        a, b = order[s], order[s + 1]
        order[s], order[s + 1] = b, a
        loc[a], loc[b] = s + 1, s
        out.append((sw, (s, s + 1)))
        nswap += 1

    def move(q, t):
        while loc[q] < t:
            swap(loc[q])
        while loc[q] > t:
            swap(loc[q] - 1)

    for i, (op, qs) in enumerate(ops):
        if len(qs) == 1:
            out.append((op, (loc[qs[0]],)))
            continue
        x, y = qs
        sx, sy = loc[x], loc[y]
        if abs(sx - sy) > 1:
            lo_q, hi_q = (x, y) if sx < sy else (y, x)
            lo, hi = min(sx, sy), max(sx, sy)
            future = [ops[j][1] for j in twoq[rank[i] + 1: rank[i] + 1 + window]]
            best = None
            for meet in range(lo, hi):
                l2 = list(loc)
                for s in range(lo + 1, meet + 1):
                    l2[order[s]] -= 1
                for s in range(meet + 1, hi):
                    l2[order[s]] += 1
                l2[lo_q], l2[hi_q] = meet, meet + 1
                score, w = float(hi - lo - 1), 1.0
                for u, v in future:
                    w *= decay
                    score += w * (abs(l2[u] - l2[v]) - 1)
                if best is None or score < best[0] - 1e-12:
                    best = (score, meet)
            move(lo_q, best[1])
            move(hi_q, best[1] + 1)
        out.append((op, (loc[x], loc[y])))
    return out, order, nswap


def _sabre(qc, seed):
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    t = transpile(qc, coupling_map=CouplingMap.from_line(qc.num_qubits), layout_method="sabre",
                  routing_method="sabre", optimization_level=1, seed_transpiler=seed,
                  basis_gates=_BASIS)
    return t, list(t.layout.final_index_layout()), t.count_ops().get("swap", 0)


# ---------------------------------------------------------------- public API
def prepare(qc, qubits=None, seed=0, light_cone_reduction=True):
    """Prepare `qc` (no measurements) for Aer MPS.

    qubits: qubits whose joint distribution you want (enables the light cone); None = all qubits.
    Returns Prepared(circuit, positions, method, info): measure/save `positions` of `circuit`."""
    t0 = time.perf_counter()
    qc = _unroll(qc)
    n0 = qc.num_qubits
    want = list(range(n0)) if qubits is None else _check_qubits(qubits, n0)
    info = dict(original_qubits=n0)
    if qubits is not None and light_cone_reduction:
        qc, idx, keep = light_cone(qc, want)
        info.update(cone_qubits=qc.num_qubits, kept_original_qubits=keep)
    else:
        idx = want
    n = qc.num_qubits
    if n <= 2:
        info["prepare_s"] = time.perf_counter() - t0
        return Prepared(qc, idx, "none", info)
    ops = _ops(qc)
    original = sum(1 for op, _ in ops if op.name == "swap")
    phys_r, final_r, inserted_r = route_lookahead(ops, n, rcm_order(ops, n))
    qc_s, final_s, sw_s = _sabre(qc, seed)
    sw_r = original + inserted_r
    # compare the same quantity: total swap gates in the circuit each candidate would run
    info.update(swaps={"rcm+look": sw_r, "sabre": sw_s},
                swaps_detail={"original": original, "rcm+look_inserted": inserted_r, "sabre_total": sw_s})
    if sw_r <= sw_s:
        site = [0] * n
        for s, q in enumerate(final_r):
            site[q] = s
        out, method = _build(phys_r, n, qc.global_phase), "rcm+look"
    else:
        site, out, method = final_s, qc_s, "sabre"
    info["prepare_s"] = time.perf_counter() - t0
    return Prepared(out, [site[i] for i in idx], method, info)


def probabilities(qc, qubits=None, simulator=None, seed=0, **sim_options):
    """Joint probabilities of `qubits` (Qiskit little-endian order: qubits[0] is the lowest bit),
    computed with Aer MPS after `prepare`. Equals Statevector(qc).probabilities(qubits)."""
    from qiskit_aer import AerSimulator
    p = prepare(qc, qubits, seed)
    c = p.circuit.copy()
    c.save_probabilities(p.positions, label="p")
    sim = simulator or AerSimulator(method="matrix_product_state", **sim_options)
    res = sim.run(c, shots=1).result()
    if not res.success:
        raise RuntimeError(res.status)
    return np.asarray(res.data(0)["p"])
