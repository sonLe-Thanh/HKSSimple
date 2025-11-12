import random 
from graph import Graph, Vertex, Edge


def generate_complete_graph(n, weight_min = 0.0, weight_max = 1.0):
    # Vertices
    vertices = [Vertex(i, 0, val=0) for i in range(n)]

    # Edge
    edges_list = []
    for i in range(n):
        for j in range(i + 1, n):
            weight = random.uniform(weight_min, weight_max)
            edges_list.append(Edge(i, j, weight))

    g = Graph()
    g.vertices_list = vertices
    g.edges_list = edges_list
    return g