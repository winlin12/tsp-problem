import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dimod

# Function to create QUBO for TSP
def create_tsp_qubo(distance_matrix):
    n = len(distance_matrix)
    Q = {}

    # Objective: Minimize the total distance
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(n):
                    for l in range(n):
                        if k != l:
                            Q[((i, k), (j, l))] = distance_matrix[i][j]

    # Constraint: Each city is visited exactly once
    A = n * n * np.max(distance_matrix)
    for i in range(n):
        for k in range(n):
            Q[((i, k), (i, k))] = -A
            for l in range(k + 1, n):
                Q[((i, k), (i, l))] = 2 * A

    # Constraint: Each position in the tour is occupied by exactly one city
    for k in range(n):
        for i in range(n):
            Q[((i, k), (i, k))] = -A
            for j in range(i + 1, n):
                Q[((i, k), (j, k))] = 2 * A

    return Q

# Solve TSP using Simulated Annealing
def solve_tsp_sa(distance_matrix):
    n = len(distance_matrix)
    Q = create_tsp_qubo(distance_matrix)

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler = dimod.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=100)

    best_solution = response.first.sample
    best_tour = [-1] * n
    for (i, k), value in best_solution.items():
        if value == 1:
            best_tour[k] = i

    # Ensure the tour starts at the same city and forms a cycle
    start_city = best_tour[0]
    best_tour.append(start_city)

    return best_tour

# Visualize the TSP solution
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

# Solve TSP using Simulated Annealing
tour = solve_tsp_sa(distance_matrix)

# Ensure the tour is valid (visits every node exactly once)
if len(set(tour)) == len(distance_matrix):
    total_distance = calculate_total_distance(tour, distance_matrix)
    print("Tour:", tour)
    print("Total distance:", total_distance)
    visualize_tsp(tour, distance_matrix, total_distance)
else:
    print("Invalid tour:", tour)
