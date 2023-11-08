import matplotlib.pyplot as plt
import networkx as nx
import random

from agent import Car

ALGORITHMS = ['kruskal', 'prim', 'boruvka']

def generate_route(N):
    nodes = []
    pos = []
    for x in range(N):
        for y in range(N):
            pos.append((x, y))
            nodes.append((x, y))

    edges = []
    for x in range(N):
        for y in range(N):
            current_node = (x, y)
            if x < N - 1:
                neighbor_node = (x + 1, y)
                if neighbor_node in nodes:
                    edges.append((current_node, neighbor_node))
            if y < N - 1:
                neighbor_node = (x, y + 1)
                if neighbor_node in nodes:
                    edges.append((current_node, neighbor_node))
    pos = {k: v for k, v in zip(nodes,pos)}
    return pos, edges

def remove_edges(G, edges_to_remove):
    if edges_to_remove >= len(G.edges()):
        print("The number of edges to remove is greater "
              "than or equal to the number of edges in the graph")
        return G
    edges = list(G.edges)

    minimum_edges = nx.minimum_spanning_tree(G, algorithm=random.choice(ALGORITHMS)).edges

    non_minimum_edges = [e for e in edges if e not in minimum_edges]

    if edges_to_remove < len(minimum_edges):
        try:
            remove = random.sample(non_minimum_edges, edges_to_remove)
            G.remove_edges_from(remove)
        except(ValueError):
            print("Too many edges to remove")
    return G

if __name__ == '__main__':
    N = 5
    pos, edges = generate_route(N)
    G = nx.Graph()

    G.add_nodes_from(pos.keys())
    G.add_edges_from(edges,color='gray')
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    nx.draw(G,pos,with_labels=True, edge_color=edge_colors)
    plt.show()
    nx.draw(remove_edges(G,2),pos,with_labels=True, edge_color=edge_colors)
    plt.show()

    start_node = (0, 0)
    finish_node = (N - 1, N - 1)

    agent = Car(G, start_node, finish_node)
    agent.navigate()

    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    nx.draw(remove_edges(G, 2), pos, with_labels=True, edge_color=edge_colors)
    plt.show()

