import numpy as np

def symmetrize_matrix(A):
    """
    Symmetrize a sparse matrix A by filling the missing values

    Input:
        A : np.array: a matrix

    Output:
        (A + A.T) / 2
    """
    return (A + A.T) / 2


def popoluate_adjacency_matrix(A, vertices_weight):
    """
    Given a unweighted adjacency matrix, and a list of vertices weight, return the weighted adjacency matrix
    The weight are the absolute differences between the vertices of an edge.

    Input:
        A: np.array: an unweighted adjacency matrix
        vertices_weight: np.array: list of scalars of each vertices

    Output:
        W: np.array: weighted adjacency matrix
    """

    # Absolute differences
    abs_diff = np.abs(vertices_weight[:, None] - vertices_weight[None, :])

    # Apply the adjacency mask
    W = A * abs_diff

    return W