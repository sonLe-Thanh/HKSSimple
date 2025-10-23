import numpy as np

from graph import Graph, create_compability_matrix, compute_matching_grb, interpret_matching
from graphFromIvw import create_adj_matrix_from_ivw
from graphHKS import *

graph1 = Graph()
graph2 = Graph()


ivw_file1 = "IvwData/Case1/Timestep1.txt"
ivw_file2 = "IvwData/Case1/Timestep2.txt"

use_edge_length = True

t_hks = np.logspace(1, 15, 100)
is_normalized_Laplacian = True
is_normalize_hks = False
is_normalize_vectors = False

use_physical_costraint = True
# Read file

graph1.read_ivw_info(ivw_file1)
graph1.build_adj_mat(use_edge_length)
graph1.compute_graph_laplacian(is_normalized_Laplacian)
graph1.compute_eigen_laplacian()
graph1.compute_graph_hks(t_hks, is_normalize_hks, is_normalize_vectors)



graph2.read_ivw_info(ivw_file2)
graph2.build_adj_mat(use_edge_length)
graph2.compute_graph_laplacian(is_normalized_Laplacian)
graph2.compute_eigen_laplacian()
graph2.compute_graph_hks(t_hks, is_normalize_hks, is_normalize_vectors)

comp_mat = create_compability_matrix(graph1, graph2, t_hks, is_normalize_hks, is_normalize_vectors, use_physical_costraint)
mat_opt = compute_matching_grb(graph1, graph2, comp_mat)
print(mat_opt)
interpret_matching(mat_opt)