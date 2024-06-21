import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import dimod
from dwave.system import LeapHybridSampler
from itertools import permutations, combinations

# Generate a random complete graph with NetworkX
def generate_random_complete_graph(num_nodes, max_weight=100):
    G = nx.complete_graph(num_nodes)
    for (u, v, w) in G.edges(data=True):
        w['weight'] = np.random.randint(1, max_weight)
    distance_matrix = nx.to_numpy_array(G, weight='weight')
    return G, distance_matrix

# Function to create QUBO for TSP
def create_tsp_qubo(distance_matrix):
    n = len(distance_matrix)
    Q = {}

    # Large constant to ensure constraints are enforced
    A = n * np.max(distance_matrix)

    # Objective: Minimize the total distance
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(n):
                    Q[(i*n + k, j*n + (k+1) % n)] = distance_matrix[i][j]

    # Constraint: Each city is visited exactly once
    for i in range(n):
        for k in range(n):
            Q[(i*n + k, i*n + k)] = -A
            for l in range(k+1, n):
                Q[(i*n + k, i*n + l)] = 2 * A

    # Constraint: Each position in the tour is occupied by exactly one city
    for k in range(n):
        for i in range(n):
            Q[(i*n + k, i*n + k)] = -A
            for j in range(i+1, n):
                Q[(i*n + k, j*n + k)] = 2 * A

    return Q

# Solve TSP using D-Wave's hybrid solver
def solve_tsp_dwave(Q, num_nodes):
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler = LeapHybridSampler(token='DEV-533f2f3ab78a490efb2dd3dff0a2256f682b6b2c')
    response = sampler.sample(bqm, time_limit=5)
    
    best_solution = response.first.sample
    best_tour = [-1] * num_nodes
    n = num_nodes
    for i in range(n):
        for k in range(n):
            if best_solution[i*n + k] == 1:
                best_tour[k] = i
    
    # Ensure the tour starts at the same city and forms a cycle
    start_city = best_tour[0]
    best_tour.append(start_city)
    
    # Calculate the total distance of the tour
    total_distance = 0
    for i in range(len(best_tour) - 1):
        total_distance += distance_matrix[best_tour[i]][best_tour[i + 1]]
    
    return best_tour, total_distance

# Dynamic Programming (Held-Karp) TSP solver
def tsp_dynamic_programming(distance_matrix):
    n = len(distance_matrix)
    all_sets = 1 << n
    dp = [[float('inf')] * n for _ in range(all_sets)]
    dp[1][0] = 0

    for mask in range(1, all_sets):
        for u in range(n):
            if mask & (1 << u):
                for v in range(n):
                    if mask & (1 << v) and u != v:
                        dp[mask][u] = min(dp[mask][u], dp[mask ^ (1 << u)][v] + distance_matrix[v][u])

    min_distance = min(dp[all_sets - 1][u] + distance_matrix[u][0] for u in range(1, n))
    
    # Reconstruct the path
    mask = all_sets - 1
    last = 0
    tour = [0]
    for _ in range(n - 1):
        index = -1
        for j in range(1, n):
            if mask & (1 << j):
                if index == -1 or dp[mask][index] + distance_matrix[index][last] > dp[mask][j] + distance_matrix[j][last]:
                    index = j
        tour.append(index)
        mask ^= (1 << index)
        last = index
    tour.append(0)  # Complete the cycle
    
    return tour, min_distance

# Simulated Annealing TSP solver
def total_distance(tour, dist_matrix):
    return sum(dist_matrix[tour[i-1]][tour[i]] for i in range(len(tour)))

def swap_two_opt(tour):
    i, j = sorted(random.sample(range(len(tour)), 2))
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour

def simulated_annealing(dist_matrix, initial_temp, cooling_rate):
    n = len(dist_matrix)
    tour = list(range(n))
    random.shuffle(tour)
    best_tour = tour[:]
    current_temp = initial_temp

    while current_temp > 1:
        new_tour = swap_two_opt(tour)
        current_distance = total_distance(tour, dist_matrix)
        new_distance = total_distance(new_tour, dist_matrix)
        if new_distance < current_distance or random.random() < np.exp((current_distance - new_distance) / current_temp):
            tour = new_tour
            if new_distance < total_distance(best_tour, dist_matrix):
                best_tour = new_tour

        current_temp *= cooling_rate

    return best_tour, total_distance(best_tour, dist_matrix)

# Function to visualize the TSP solution
def visualize_tsp(tour, distance_matrix, total_distance):
    G = nx.Graph()
    pos = {}
    labels = {}

    for idx, node in enumerate(tour):  # Exclude the last repeated node for visualization
        pos[node] = (np.cos(2 * np.pi * idx / len(tour)), np.sin(2 * np.pi * idx / len(tour)))
        labels[node] = str(node + 1)
        
    edges = [(tour[i], tour[i + 1]) for i in range(len(tour) - 1)]
    edges.append((tour[-1], tour[0]))
    
    G.add_edges_from(edges)
    edge_labels = {(tour[i], tour[i + 1]): distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)}
    edge_labels[(tour[-1], tour[0])] = distance_matrix[tour[-1]][tour[0]]
    
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=500, font_size=15)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title(f'TSP Solution with Edge Weights\nTotal Distance: {total_distance}')
#    plt.show()

if __name__ == "__main__":
    num_nodes = 15  # Number of cities
    initial_temp = 10000
    cooling_rate = 0.995

    # Generate a random complete graph and distance matrix
    G, distance_matrix = generate_random_complete_graph(num_nodes)

    print("Random Distance Matrix:")
    print(distance_matrix)

    # Solve TSP using dynamic programming
    tour_dp, total_distance_dp = tsp_dynamic_programming(distance_matrix)
    print("DP Tour:", tour_dp)
    print("DP Total Distance:", total_distance_dp)
    visualize_tsp(tour_dp, distance_matrix, total_distance_dp)

    # Create QUBO for the distance matrix
    Q = create_tsp_qubo(distance_matrix)

    # Solve TSP using D-Wave's hybrid solver
    tour_dwave, total_distance_dwave = solve_tsp_dwave(Q, num_nodes)
    print("D-Wave Tour:", tour_dwave)
    print("D-Wave Total Distance:", total_distance_dwave)
    visualize_tsp(tour_dwave, distance_matrix, total_distance_dwave)

    # Solve TSP using simulated annealing
    best_tour_sa, best_distance_sa = simulated_annealing(distance_matrix, initial_temp, cooling_rate)
    print("SA Tour:", best_tour_sa)
    print("SA Total Distance:", best_distance_sa)
    visualize_tsp(best_tour_sa, distance_matrix, best_distance_sa)
