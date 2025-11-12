import numpy as np

from graph import Graph
from graphMatching import create_compability_matrix, compute_matching_grb, interpret_matching
from createGraph import generate_complete_graph

# graph1 = Graph()
# graph2 = Graph()


# ivw_file1 = "IvwData/Case1/Timestep1.txt"
# ivw_file2 = "IvwData/Case1/Timestep2.txt"

# use_edge_length = True

# t_hks = np.linspace(1, 4, 100)
# is_normalized_Laplacian = True
# is_normalize_hks = False
# is_normalize_vectors = False

# use_physical_costraint = True
# # Read file

# graph1.read_ivw_info(ivw_file1)
# graph1.build_adj_mat(use_edge_length)
# graph1.compute_graph_laplacian(is_normalized_Laplacian)
# graph1.compute_eigen_laplacian()
# graph1.compute_graph_hks(t_hks, is_normalize_hks, is_normalize_vectors)



# graph2.read_ivw_info(ivw_file2)
# graph2.build_adj_mat(use_edge_length)
# graph2.compute_graph_laplacian(is_normalized_Laplacian)
# graph2.compute_eigen_laplacian()
# graph2.compute_graph_hks(t_hks, is_normalize_hks, is_normalize_vectors)

# comp_mat = create_compability_matrix(graph1, graph2, t_hks, is_normalize_hks, is_normalize_vectors, use_physical_costraint)
# mat_opt = compute_matching_grb(graph1, graph2, comp_mat)

# interpret_matching(mat_opt)


num_sample = 10
c_h = 1

g = generate_complete_graph(100)

g.build_adj_mat(use_edge_length=False)
g.compute_graph_laplacian(is_normalized=False)
g.compute_eigen_laplacian()
# print(max(g.eigenvals))
t_max = 5
g.compute_graph_adalfs(t_max, num_sample, c_h)