import matplotlib.pyplot as plt
import networkx as nx
import random

from agent import Car

NODE_SIZE = 650
def generate_route(N):
    """
    Generates route NxN.
    :param N:
    :return:
    """
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

def remove_edges(G: nx.Graph, edges_to_remove: int):
    """
    Remove edges from route.
    :param G: nx.Graph
    :param edges_to_remove: int
    :return: nx.Graph
    """
    if edges_to_remove >= len(G.edges()):
        print("The number of edges to remove is greater "
              "than or equal to the number of edges in the graph")
        return G
    edges = list(G.edges)

    minimum_edges = nx.minimum_spanning_tree(G,weight="weight").edges

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
    # G.add_edges_from(edges,color='gray')
    for i in edges:
        G.add_edge(i[0], i[1],color="grey",weight=random.random())
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    # First graph generation
    nx.draw(G,pos,with_labels=True,
            edge_color=edge_colors,
            node_size=NODE_SIZE)
    plt.show()
    # Removing edges
    nx.draw(remove_edges(G,15),pos,
            with_labels=True,
            edge_color=edge_colors,
            node_size=NODE_SIZE)
    plt.show()

    start_node = (0, 0)
    finish_node = (4, 4)
    # Agent
    agent = Car(G, start_node, finish_node)
    agent.navigate()

    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    node_colors = {node: '#1F78B4' for node in G.nodes}
    node_colors[start_node] = "green"
    node_colors[finish_node] = "red"
    # Drawing graph with path
    nx.draw(G, pos, with_labels=True,
            edge_color=edge_colors,
            node_size=NODE_SIZE,
            node_color = [node_colors[i] for i in G.nodes])
    nx.draw_networkx_edges(G, pos,
                           node_size=NODE_SIZE,
                           arrows=True,
                           edgelist=agent.direction_edges_list,
                           edge_color='red',
                           arrowstyle='->',
                           arrowsize=15)
    plt.show()
    print(f"Full car path: {agent.full_path.__str__().replace('), (',') -> (',)} ")
    print(agent.knowledge_base)
