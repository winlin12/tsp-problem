import numpy as np
import cirq
import networkx as nx
import matplotlib.pyplot as plt

def normal_pagerank(H,convergence_threshold):
    # Set the damping factor
    alpha = 0.85

    # Define the maximum number of iterations
    max_iter = 1000

    # Initialize the PageRank vector as a vector of ones
    n = H.shape[0]
    x = np.ones(H.shape[0]) / H.shape[0]  
    # Iterate the PageRank algorithm
    v = np.ones(n) / n
    for i in range(max_iter):
        x_new = alpha * H.dot(x) + (1 - alpha) * v

        norm = np.linalg.norm(abs(x_new - x))
        #print("Norm: %s, Xnew: %s"%(norm,x_new))

        #x = x_new
        #print("Norm: %s, Xnew: %s"%(norm,x_new))
        if (norm < convergence_threshold):
            print(f'Normal Converged after {i+1} iterations')
            break

        x = x_new

    # Normalize the final PageRank vector so that its elements sum to 1
    pagerank = x / x.sum()
    return pagerank
# Hyperlink matrix representing the links between web pages
H = np.array([[0, 1, 0, 0],
              [1, 0, 1, 0],
              [1, 0, 0, 1],
              [1, 1, 0, 0]])
H = np.array([[0, 1, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1, 0]])
print(np.linalg.det(H))
# Compute the transition probability matrix
P = H / H.sum(axis=0, keepdims=True)
print(P)
#P = H
# Set the damping factor
alpha = 0.85

# Set the number of qubits
num_qubits = int(np.ceil(np.log2(P.shape[0])))
#print(num_qubits)
# Define the maximum number of iterations
max_iter = 1

# Initialize the PageRank vector as a vector of ones
x = np.ones(P.shape[0]) / P.shape[0]
print("X shape = %s"%x.shape)
#print(num_qubits)
convergence_threshold = 0.001
# Iterate the QSVD-based PageRank algorithm
for iterP in range(max_iter):
    # Define the quantum circuit
    circuit = cirq.Circuit()

    # Define the qubits
    qubits = cirq.LineQubit.range(num_qubits)

    # Define the initial state as the current PageRank vector
    #state = np.sqrt(x)
    state = x
    for i,s in enumerate(state):
        if np.isnan(s):
            state[i] = 0
    

    for i, s in enumerate(state):
        circuit.append(cirq.H(qubits[i%num_qubits]))
        circuit.append(cirq.X(qubits[i%num_qubits])**s)

    # Apply controlled-U gates to perform QSVD
    for i in range(num_qubits-1):
        angle = 2*np.arcsin(1/(2**(i+1)))
        for j in range(i+1):
            circuit.append(cirq.CZ(qubits[j], qubits[i+1])**(2*angle))

    # Measure the qubits
    #for i in range(num_qubits):
    circuit.append(cirq.measure(*qubits, key='result'))

    # Simulate the circuit and get the results
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1024)

    # Compute the singular values and vectors from the results
    counts = result.histogram(key='result')
    singular_values = np.sqrt([v/sum(counts.values()) for v in counts.values()])
    singular_vectors = np.ones((num_qubits,singular_values.shape[0]), dtype=complex)
    #print("SV SHAPE: %s"%singular_values.shape)
    #print(singular_values)
    #print("SVect SHAPE: %s"%singular_vectors.shape[1])
    for i, s in enumerate(singular_values):
        for j in range(num_qubits):
            #phase = (-1)**sum([int(bitstring[k]) for k in range(i+1)])
            if counts[i] == 0:
                singular_vectors[j][i] = 0
            else:
                singular_vectors[j][i] = np.sqrt(s)/ np.sqrt(counts[i])
            #singular_vectors[j][i] = np.sqrt(s)*singular_vectors[j][i]/ np.sqrt(counts[i])

    # Compute the new PageRank vector as the sum of the singular vectors weighted by the singular values
    #print(np.sum(singular_vectors,axis=0))
    #sprint(np.sum(singular_vectors,axis=0)*singular_values)
    #x_new = alpha * sum(singular_vectors[i] * singular_values[i] for i in range(num_qubits)) + (1 - alpha) * np.ones(P.shape[0]) / P.shape[0]
    SV = np.sum(singular_vectors,axis=0)*singular_values
    if len(SV) > len(x):
        SV = SV[:x.shape[0]]
    elif len(SV) < len(x):
        #print(len(SV))
        SV = np.pad(SV,(0,len(x)-len(SV)))
    #x_new = alpha*np.matmul(P,SV)  + (1 - alpha) * np.ones(P.shape[0]) / P.shape[0]
    x_new = np.matmul(P,SV)
    # Check for convergence
    norm = np.linalg.norm(x_new - x)
    print("Norm: %s, Xnew: %s"%(norm,x_new/x_new.sum()))

    if norm < convergence_threshold:
        print(f'Quantum Converged after {iterP +1} iterations')
        break
    x = x_new
    

    

# Normalize the final PageRank vector so that its elements sum to 1
pagerank = x / x.sum()

print(np.round(np.real(pagerank),3))
PRC = normal_pagerank(P,convergence_threshold=1e-8)

print(np.round(PRC,3))


'''G = nx.from_numpy_matrix(H)
pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes
nx.draw_networkx_edges(G, pos, width=1.0, alpha=1)
labels = {}
labels[0] = "1"
labels[1] = "2"
labels[2] = "3"
labels[3] = "4"
labels[4] = "5"
labels[5] = "6"
nx.draw_networkx_labels(G, pos, labels, font_size=22, font_color="Black")

nx.draw_circular(G2)


plt.axis('equal')
plt.show()'''
