import numpy as np
import cirq
import networkx as nx
import matplotlib.pyplot as plt

def create_tsp_quantum_circuit(distance_matrix):
    n = len(distance_matrix)
    qubits = cirq.GridQubit.rect(1, n)
    circuit = cirq.Circuit()

    # Apply Hadamard gates to create superposition
    for qubit in qubits:
        circuit.append(cirq.H(qubit))
    
    # Apply controlled-Z gates to encode the distances
    for i in range(n):
        for j in range(i + 1, n):
            weight = distance_matrix[i][j]
            if weight > 0:
                circuit.append(cirq.Z(qubits[i]) ** weight)
                circuit.append(cirq.Z(qubits[j]) ** weight)
                circuit.append(cirq.CZ(qubits[i], qubits[j]) ** weight)
    
    # Measure the qubits
    circuit.append(cirq.measure(*qubits, key='result'))
    
    return circuit, qubits

def run_quantum_tsp(circuit, qubits):
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=100)
    return result

def decode_tsp_result(result, n):
    # Get the most frequent measurement result
    counts = result.histogram(key='result')
    most_common_result = counts.most_common(1)[0][0]
    tour = [int(bit) for bit in format(most_common_result, f'0{n}b')]
    return tour

def visualize_tsp(tour, distance_matrix, total_distance):
    G = nx.Graph()
    pos = {}
    labels = {}

    for idx, node in enumerate(tour[:-1]):  # Exclude the last repeated node for visualization
        pos[node] = (np.cos(2 * np.pi * idx / (len(tour) - 1)), np.sin(2 * np.pi * idx / (len(tour) - 1)))
        labels[node] = str(node + 1)
        
    edges = [(tour[i], tour[i + 1]) for i in range(len(tour) - 1)]
    
    G.add_edges_from(edges)
    edge_labels = {(tour[i], tour[i + 1]): distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)}
    
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=500, font_size=15)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title(f'TSP Solution with Edge Weights\nTotal Distance: {total_distance}')
    plt.show()

# Calculate total distance of the tour
def calculate_total_distance(tour, distance_matrix):
    total_distance = sum(distance_matrix[tour[i-1]][tour[i]] for i in range(1, len(tour)))
    return total_distance

# Example distance matrix
distance_matrix = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

# Create the quantum circuit for TSP
circuit, qubits = create_tsp_quantum_circuit(distance_matrix)
print("Quantum Circuit:\n", circuit)

# Run the quantum TSP
result = run_quantum_tsp(circuit, qubits)
print("Quantum TSP Result:\n", result)

# Decode the result to get the tour
n = len(distance_matrix)
tour = decode_tsp_result(result, n)

# Ensure the tour is valid (visits every node exactly once)
if len(set(tour)) == n:
    total_distance = calculate_total_distance(tour, distance_matrix)
    print("Tour:", tour)
    print("Total distance:", total_distance)
    visualize_tsp(tour, distance_matrix, total_distance)
else:
    print("Invalid tour:", tour)
