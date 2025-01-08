import numpy as np
from math import gcd
from fractions import Fraction

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# Authenticate with IBM Quantum Service
service = QiskitRuntimeService(
    channel="ibm_quantum",
    token='YOUR_IBM_QUANTUM_TOKEN'
)

# Parameters
N = 77  # Number to factor
a = 10  # Chosen coprime (10 works for 77)
N_COUNT = 8  # Number of counting qubits

# Controlled modular exponentiation
def c_amodN(a, power, N):
    qc = QuantumCircuit(7)  # 7 qubits for 77
    for _ in range(power):
        qc.swap(1, 2)
        qc.swap(0, 1)
        if a == 10:
            for q in range(7):
                qc.x(q)
    gate = qc.to_gate()
    gate.name = f"{a}^{power} mod {N}"
    return gate.control(1)

# Inverse Quantum Fourier Transform
def qft_dagger(n):
    qc = QuantumCircuit(n)
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / 2**(j - m), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc

# Quantum Circuit
qc = QuantumCircuit(N_COUNT + 7, N_COUNT)

for i in range(N_COUNT):
    qc.h(i)
qc.x(N_COUNT)

for i in range(N_COUNT):
    qc.append(c_amodN(a, 2**i, N), [i] + list(range(N_COUNT, N_COUNT + 7)))

qc.append(qft_dagger(N_COUNT), range(N_COUNT))
qc.measure(range(N_COUNT), range(N_COUNT))

# Run on backend
backend = service.least_busy(simulator=False)
print("Using backend:", backend.name)
transpiled_circ = transpile(qc, backend=backend)
sampler = Sampler()
job = sampler.run([transpiled_circ], shots=4096)
result = job.result()
pub_result = result[0]
counts_dict = pub_result.data.c.get_counts()
shots_used = sum(counts_dict.values())
counts_prob = {bitstring: c / shots_used for bitstring, c in counts_dict.items()}

# Analyze Results
phases = [(int(bitstring, 2) / (2**N_COUNT), prob) for bitstring, prob in counts_prob.items()]
phases.sort(key=lambda x: x[1], reverse=True)
best_phase, best_prob = phases[0]

frac = Fraction(best_phase).limit_denominator(N)
r = frac.denominator
guess1 = gcd(a**(r//2) - 1, N)
guess2 = gcd(a**(r//2) + 1, N)

print("\nPotential factors from guess:")
print(f"  gcd({a}^({r//2}) - 1, {N}) = {guess1}")
print(f"  gcd({a}^({r//2}) + 1, {N}) = {guess2}")
