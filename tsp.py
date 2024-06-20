import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def tsp(distance_matrix):
    G = nx.complete_graph(len(distance_matrix))
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            G[i][j]['weight'] = distance_matrix[i][j]

    tour = nx.approximation.traveling_salesman_problem(G, cycle=True)
    total_distance = sum(distance_matrix[tour[i-1]][tour[i]] for i in range(len(tour)))
    
    return tour, total_distance

# Function to calculate the total distance of a permutation
def calculate_total_distance(permutation, distance_matrix):
    total_distance = 0
    n = len(permutation)
    for i in range(n):
        total_distance += distance_matrix[permutation[i-1]][permutation[i]]
    return total_distance

# Brute-force TSP solver
def tsp_brute_force(distance_matrix):
    n = len(distance_matrix)
    all_permutations = permutations(range(n))
    min_distance = float('inf')
    best_route = None
    
    for perm in all_permutations:
        current_distance = calculate_total_distance(perm, distance_matrix)
        if current_distance < min_distance:
            min_distance = current_distance
            best_route = perm
    
    return best_route, min_distance

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
    plt.show()

# Generate a random complete graph with NetworkX
def generate_random_complete_graph(num_nodes, max_weight=100):
    G = nx.complete_graph(num_nodes)
    for (u, v, w) in G.edges(data=True):
        w['weight'] = np.random.randint(1, max_weight)
    distance_matrix = nx.to_numpy_array(G, weight='weight')
    return G, distance_matrix

# Generate a random test case
num_nodes = 5  # Number of cities
G, distance_matrix = generate_random_complete_graph(num_nodes)

print("Random Distance Matrix:")
print(distance_matrix)

# Solve TSP using brute-force
tour_brute_force, total_distance_brute_force = tsp_brute_force(distance_matrix)

# Solve TSP using dynamic programming
tour_dp, total_distance_dp = tsp_dynamic_programming(distance_matrix)

print("Brute-Force Tour:", tour_brute_force)
print("Brute-Force Total Distance:", total_distance_brute_force)
print("DP Tour:", tour_dp)
print("DP Total Distance:", total_distance_dp)
visualize_tsp(tour_brute_force, distance_matrix, total_distance_brute_force)
visualize_tsp(tour_dp, distance_matrix, total_distance_dp)

# Solve TSP and visualize the solution
tour, total_distance = tsp(distance_matrix)
print("Tour:", tour)
print("Total distance:", total_distance)
visualize_tsp(tour, distance_matrix, total_distance)
