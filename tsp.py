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

def visualize_original_graph(distance_matrix):
    G = nx.complete_graph(len(distance_matrix))
    pos = {}
    labels = {}

    for idx in range(len(distance_matrix)):
        pos[idx] = (np.cos(2 * np.pi * idx / len(distance_matrix)), np.sin(2 * np.pi * idx / len(distance_matrix)))
        labels[idx] = str(idx + 1)
    
    edge_labels = {(i, j): distance_matrix[i][j] for i in range(len(distance_matrix)) for j in range(i + 1, len(distance_matrix))}
    
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=500, font_size=15)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title('Original Graph with Edge Weights')
    plt.show()

def visualize_tsp(tour, distance_matrix):
    G = nx.Graph()
    pos = {}
    labels = {}

    for idx, node in enumerate(tour):
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
    plt.title('TSP Solution with Edge Weights')
    plt.show()

# Example distance matrix
distance_matrix = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

# Visualize original graph
visualize_original_graph(distance_matrix)

# Solve TSP and visualize the solution
tour, total_distance = tsp(distance_matrix)
print("Tour:", tour)
print("Total distance:", total_distance)
visualize_tsp(tour, distance_matrix)
