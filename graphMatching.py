import numpy as np 
from graphHKS import compute_graph_hks


# Create the compability matrix

def create_compability_matrix(adj_mat1, adj_mat2, 
                              eigenvals1, eigenvec1,
                              eigenvals2, eigenvec2,
                              t_list_hks,
                              is_normalized_kernel_hks,
                              is_normalized_vector_hks
                              ):
    

    # From the adj_mats, build the compability matrix K

    num_vertices1 = adj_mat1.shape[0]
    num_vertices2 = adj_mat2.shape[0]

    comp_mat = np.zeros((num_vertices1 * num_vertices2, num_vertices1, num_vertices2))

    # Start filling this matrix
    for idx_i_g1, idx_j_g1 in zip(range(num_vertices1), range(num_vertices1)):
        for idx_a_g2, idx_b_g2 in zip(range(num_vertices2), range(num_vertices2)):
            
            # We calculate K ia jb

            # First order condition 

            if idx_i_g1 == idx_j_g1 and idx_a_g2 == idx_b_g2:
                # We also want to access the node from their index 
                # HKS distance
                hks1 = compute_graph_hks(eigenvals1, eigenvec1, t_list_hks, is_normalized_kernel_hks, is_normalized_vector_hks)
                hks2 = compute_graph_hks(eigenvals2, eigenvec2, t_list_hks, is_normalized_vector_hks, is_normalized_vector_hks)
