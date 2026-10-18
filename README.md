# mpsprep

Prepares Qiskit circuits for the Qiskit Aer **matrix product state (MPS)** simulator.
The output distribution stays exactly the same; the run can be much faster.

```python
from mpsprep import probabilities
p = probabilities(qc, qubits=[0, 1, 2, 3])   # == Statevector(qc).probabilities([0, 1, 2, 3])
```

or, to keep control of the simulator:

```python
from mpsprep import prepare
prep = prepare(qc, qubits=[0, 1, 2, 3])
c = prep.circuit.copy()
c.save_probabilities(prep.positions)          # or c.measure(...) on prep.positions
AerSimulator(method="matrix_product_state").run(c)
```

## What it does
1. **Light cone** (only when `qubits=` is given). Qiskit's `LightCone` pass removes the gates that cannot affect the
   requested qubits. Qubits that are left idle are then removed.
2. **Routing onto a line.** Aer MPS applies 2-qubit gates only to neighbouring sites. For a distant gate it
   moves one qubit next to the other without lookahead and does not move it back (see `MPS::apply_2_qubit_gate`;
   the tutorial says otherwise, [Qiskit/qiskit-aer#2482](https://github.com/Qiskit/qiskit-aer/issues/2482)).
   mpsprep routes the circuit itself and keeps the variant with fewer swaps:
   - reverse Cuthill–McKee ordering + lookahead routing, or
   - Qiskit SABRE on a line coupling map.

Neither technique is new. Light cones are standard for QAOA, and bandwidth or Fiedler orderings have been used in DMRG
for decades. This package only puts them together for Aer MPS and measures the effect.

## Measured (Windows notebook, 1 CPU thread, Qiskit 2.5.2, Aer 0.17.2, joint distribution of 4 qubits)
| circuit (32 qubits) | plain Aer MPS | SABRE on a line | mpsprep without light cone | mpsprep |
|---|---:|---:|---:|---:|
| QAOA p=2, 4×8 grid | > 300 s | 6.3–7.0 s | 5.1–5.4 s | 0.0013–0.0014 s |
| QAOA p=2, 4×8 grid, shuffled labels | > 300 s | 40–202 s | 17–202 s | 0.016–0.040 s |
| QAOA p=2, random 3-regular graph | > 300 s | > 300 s | > 300 s | 0.006–0.028 s |
| random pairs, depth 4 | > 300 s | > 300 s | > 300 s | 0.012–1.8 s |

Preparation takes 0.02–0.08 s. Each row covers 3 random instances. All results agree with each other and with the statevector
(at 24–28 qubits) to within 1e-8. Data and scripts: the parent BendaLabs project.

`examples/benchmark.py` (a different QAOA instance, 4×8 shuffled grid, seed 1): plain Aer 157 s; mpsprep without
light cone **260 s (slower)**; mpsprep 0.004 s.

## Limits (please read)
- The big gains come from the light cone, so you need to ask for **a few qubits** of a **shallow** circuit.
  For deep circuits (e.g. QAOA p ≥ 3–4, random depth ≥ 6) the cone covers almost everything, and no exact method we tried
  finished within 5 minutes at 32 qubits.
- Without the light cone, the routing is only a modest improvement over SABRE (about 1.1×). It can be slower than plain Aer.
  Choosing by swap count is cheap, but it does not predict the run time well.
- Unitary circuits only: no mid-circuit measurements, resets or classical control.
- Timing on a laptop varies up to 3× with its thermal and power state. Compare methods back to back.

## Tests
```
python tests/test_correctness.py   # or: python -m pytest tests
# 17 circuits incl. random ones with 3-qubit gates, vs Statevector, tol 1e-8
```

Built by Marek Benda with the help of an AI assistant (Claude).

License: Apache-2.0
