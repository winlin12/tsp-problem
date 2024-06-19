import cirq
import numpy as np

# Define the matrix to be decomposed
matrix = np.array([[0, 1j/np.sqrt(2), 0, 1j/np.sqrt(2)],
                   [1j/np.sqrt(2), 0, 1j/np.sqrt(2), 0],
                   [0, 1j/np.sqrt(2), 0, -1j/np.sqrt(2)],
                   [1j/np.sqrt(2), 0, -1j/np.sqrt(2), 0]])

matrix = np.random.randint(2, size=(4, 4))
matrix = matrix + matrix.T             

# Set the number of qubits needed for the circuit
n = matrix.shape[0]
qreg = [cirq.LineQubit(i) for i in range(n)]
u, s, vh = np.linalg.svd(matrix)
print(u)
print(s)
print(vh)
# Create the quantum circuit
circuit = cirq.Circuit()
qid_shape=(2,)*n
# Apply a Hadamard gate to the first qubit
circuit.append(cirq.H(qreg[0]))

# Prepare the input state using a series of controlled unitary gates
for i in range(n):
    gate = cirq.MatrixGate(matrix)
    controlled_gate = cirq.ControlledGate(gate)
    circuit.append(controlled_gate(qreg[i],qreg[(i+1)%n],qreg[(i+2)%n]))

# Apply a series of controlled phase gates based on the state of the last qubit
for i in range(n-1):
    circuit.append(cirq.CNOT(qreg[n-1], qreg[i]))
    for j in range(2**(n-i-2)):
        circuit.append(cirq.rz(np.pi/2**(n-i-1)).on(qreg[i]).controlled_by(qreg[n-1]))
    circuit.append(cirq.CNOT(qreg[n-1], qreg[i]))

# Measure the first qubit to obtain the singular values
circuit.append(cirq.measure(*qreg, key='svd'))

simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)
print(result.histogram(key='svd'))