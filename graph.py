# Simple graph data structure that holds every information that we want to calculate

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cityblock, euclidean, minkowski

import cvxpy as cp

class Vertex:
    def __init__(self, idx, node_type, val):
        self.idx = idx
        self.node_type = node_type
        self.val = val

        self.pos = np.zeros(3)

        # Extra info
        self.ivw_idx = -1


class Edge:
    def __init__(self, start_node_idx, end_node_idx, weight=1, length=1):
        self.start_node_idx = start_node_idx
        self.end_node_idx = end_node_idx

        self.weight = weight

        # Extra info
        self.length = length
        self.num_points = 0


class Graph:
    def __init__(self, vertices_list : list[Vertex], edges_list : list[Edge]):
        self.vertices_list = vertices_list
        self.edges_list = edges_list


        # Keep extra info
        self.minima_idx = []
        self.maxima_idx = []
        self.saddle_idx = []

    def __init__(self):
        self.vertices_list = []
        self.edges_list = []

        # Keep extra info
        self.minima_idx = []
        self.maxima_idx = []
        self.saddle_idx = []

    def get_node(self, node_idx):
        return self.vertices_list[node_idx]
    

    def build_adj_mat(self):
        # Build adjacency matrix
        self.adj_mat = np.zeros((len(self.vertices_list), len(self.vertices_list)))

        for edge in self.edges_list:
            self.adj_mat[edge.start_node_idx, edge.end_node_idx] = self.adj_mat[edge.end_node_idx, edge.start_node_idx] = edge.weight


    def read_ivw_info(self, in_file):
        self.vertices_list = []
        self.edges_list = []


        idx_graph_counter = 0

        all_idx_wpc = {}
        with open(in_file, "r") as file:
            for line in file:
                parts = line.split(",")

                # First one is the number of points in the separatrices
                num_points = int(parts[0])
                # After that is the length of the separatrices
                sepa_length = float(parts[1])

                # After that, we have the first node info
                first_node_info = parts[2].split(" ")
                # Second node info 
                second_node_info = parts[3].split(" ")

                # We also collect the position 
                first_node_pos = parts[4].split(" ")
                second_node_pos = parts[5].split(" ")

                # Check if the first node is not in the graph index
                first_node_ivw_idx, first_node_val, first_node_type = int(first_node_info[0]), float(first_node_info[1]), int(first_node_info[2])
                second_node_ivw_idx, second_node_val, second_node_type = int(second_node_info[0]), float(second_node_info[1]), int(second_node_info[2])

                # If the nodes have node appeared yet
                if first_node_ivw_idx not in all_idx_wpc:
                    # This node is new
                    first_node = Vertex(idx_graph_counter, first_node_type, first_node_val)
                    
                    first_node.ivw_idx = first_node_ivw_idx
                    first_node.pos = np.array([float(first_node_pos[0]), float(first_node_pos[1]), float(first_node_pos[2])])
                    # Add that to the graph
                    self.vertices_list.append(first_node)

                    # Update the index list
                    all_idx_wpc[first_node.ivw_idx] = idx_graph_counter

                    match first_node.node_type:
                        case 0:
                            self.minima_idx.append(first_node.idx)
                        case 1:
                            self.saddle_idx.append(first_node.idx)
                        case 2:
                            self.maxima_idx.append(first_node.idx)
                        case _:
                            print("Not supported")
                    idx_graph_counter += 1
                
                # Do the same thing for the seond node
                if second_node_ivw_idx not in all_idx_wpc:
                    # This node is new
                    second_node = Vertex(idx_graph_counter, second_node_type, second_node_val)
                    second_node.ivw_idx = second_node_ivw_idx
                    second_node.pos = np.array([float(second_node_pos[0]), float(second_node_pos[1]), float(second_node_pos[2])])
                    # Add that to the graph
                    self.vertices_list.append(second_node)

                    # Update the index list 
                    all_idx_wpc[second_node.ivw_idx] = idx_graph_counter

                    match second_node.node_type:
                        case 0:
                            self.minima_idx.append(first_node.idx)
                        case 1:
                            self.saddle_idx.append(first_node.idx)
                        case 2:
                            self.maxima_idx.append(first_node.idx)
                        case _:
                            print("Not supported")
                    idx_graph_counter += 1
                

                # We want to build the edge as well 
                # Get the two starting nodes
                # We use the dict, at this point, it is made sure that the indices exist
                start_node_idx = all_idx_wpc[first_node_ivw_idx]
                end_node_idx = all_idx_wpc[second_node_ivw_idx]

                # Get the actual nodes 
                start_node = self.get_node(start_node_idx)
                end_node = self.get_node(end_node_idx)

                # Create a new edge
                new_edge = Edge(start_node_idx, end_node_idx, np.abs(start_node.val - end_node.val))
                # Set the length for the edge
                new_edge.length = sepa_length
                new_edge.num_points = num_points

                # Add that to the list 
                self.edges_list.append(new_edge)



    
    # ======================== Graph signatures ===============================

    def compute_graph_laplacian(self, is_normalized = False):
        """
        Compute a Laplacian matrix given an adjacency matrix of a graph G=(V,E)

        Input:
            self : to get - adjacency matrix of size n x n
            is_normalized: bool - decide whether to normalize this matrix or not

        Output:
            L: np.array - Laplacian matrix of G of size n x n
        """
        # Check if adjacency matrix is non-zero
        if self.adj_mat is None or np.all(self.adj_mat, 0):
            self.build_adj_mat()

        # Compute the matrix D 
        # Summing over rows
        diag_entries = self.adj_mat.sum(axis=1)

        if not is_normalized:
            D = np.diag(diag_entries)
            self.laplacian = D - self.adj_mat
        else:
            # We normalize
            # The formula is L = I - D^{-1/2} A D^{-1/2}
            # We cannot always expect that D is invertible
            # This will happen if any of the entries of D is zeros

            # Exclude all zeros entries from the list and perform normally
            non_zeros_entries = diag_entries != 0
            inverse_entries = np.zeros_like(diag_entries)
            inverse_entries[non_zeros_entries] = 1. / np.sqrt(diag_entries[non_zeros_entries])
            
            D_sqrt_inv = np.diag(inverse_entries)
            identity_mat = np.eye(self.adj_mat.shape[0])
            self.laplacian = identity_mat - D_sqrt_inv @ self.adj_mat @ D_sqrt_inv


    def compute_eigen_laplacian(self, num_eigen = 100):
        """
        Compute the eigenvalues and eigen vectors of the Laplacian matrix of the Graph G(V,E)

        Input:
            self : to get the Laplacian matrix
            num_eigen: number of eigenvalues we want to calculate

        Output:
            eigenvalues, eigenvectors: eigen values and eigen vectors of L, the Laplacian Matrix associating to G
        """

        if self.laplacian is None:
            # Use normalized Laplacian on default
            self.compute_graph_laplacian(True)

        num_eigen_calc = min(self.adj_mat.shape[0], num_eigen)
        # The i-th value containing the eigen vals and eigenvecs of i-th vertices, resp.
        self.eigenvals, self.eigenvecs = eigsh(self.laplacian, num_eigen_calc, which="SM", sigma=0.0)


    def compute_graph_hks(self, t_list, is_normalized_kernel = False, is_normalized_vectors = False):
        """
        Compute the heat kernel signature, given eigenvalues, eigenvectors, and the scaling parameter t
        The discrete HKS is given as 
            HKS(t,x) = \Sigma_{i=1}^k e^{-\lambda_i t} v_i^2,
            where \lambda_i and v_i are the corresponding eigenvalues and eigenvectors of the Laplacian
        

        Input:
            self : to get the eigenvalues and the eigenvectors of the Laplacian
            t_list: np.array - scaling variables
            is_normalized_kernel: do we want to normalized the HKS with the heat trace
            is_normalized_vectors: do we want to normalized the eigenvectors

        """

        if self.eigenvals is None or self.eigenvecs is None:
            self.compute_eigen_laplacian()

        num_vertices = len(self.vertices_list)
        self.hks = np.zeros((num_vertices, len(t_list)))

        for idx, t in enumerate(t_list):
            exp_term = np.exp(-t * self.eigenvals)

            if is_normalized_vectors:
                eigenvectors_norm = self.eigenvecs / np.linalg.norm(self.eigenvecs, axis=0, keepdims=True)
                self.hks[:, idx] = (eigenvectors_norm ** 2) @ exp_term
            else:
                self.hks[:, idx] = (self.eigenvecs ** 2) @ exp_term

            if is_normalized_kernel:
                heat_trace = np.sum(exp_term)
                self.hks[:, idx] /= heat_trace


    def compute_graph_wks(self, t_list, sigma, is_normalized_kernel = False, is_normalized_vectors = False):
        """
        Compute the wave kernel signature, given eigenvalues, eigenvectors, and the scaling parameter t
        The discrete HKS is given as 
            HKS(t,x) = \Sigma_{i=1}^k e^{-(t - log \lamba_k)^2 / (2 \sigma^2)} v_i^2,
            where \lambda_i and v_i are the corresponding eigenvalues and eigenvectors of the Laplacian
        

        Input:
            self: to get eigenvalues and eigenvectors of the Laplacian
            t_list: np.array - scaling variables
            sigma: value: the value sigma for wave kernel signature
            is_normalized_value: do we want to normalized the WKS with the 'wave trace'
            is_normalized_vectors: do we want to normalized the eigenvectors

        """
        if self.eigenvals is None or self.eigenvecs is None:
            self.compute_eigen_laplacian()

        num_vertices = len(self.vertices_list)
        self.wks = np.zeros((num_vertices, len(t_list)))

        for idx, t in enumerate(t_list):
            exp_term = np.exp(-(t  - np.log(self.eigenvals)) ** 2 / (2 * sigma ** 2))

            if is_normalized_vectors:
                eigenvectors_norm = self.eigenvecs / np.linalg.norm(self.eigenvecs, axis=0, keepdims=True)
                self.wks[:, idx] = (eigenvectors_norm ** 2) @ exp_term
            else:    
                self.wks[:, idx] = (self.eigenvecs ** 2) @ exp_term

            if is_normalized_kernel:
                heat_trace = np.sum(exp_term)
                self.wks[:, idx] /= heat_trace



    def compute_graph_heat_kernel(self, t_list, idx_v1, idx_v2, is_normalized_kernel = False, is_normalized_vectors = False):
        """
        Compute the heat kernel between the 2 vectrices, given the ts, indices of the two vectices


        Input:
            self: to get eigenvalues and eigenvectors of the Laplacian
            t_list: np.array - scaling variables
            idx_v1: idx of the starting vertex
            idx_v2: idx of the ending vertex
            is_normalized_kernel: do we want to normalize the kernel with heat trace
            is_normalized_vectors: do we want to normalize the eigenvectors

        Output:
            heat_kernel: np.array, i-th element corresponding to the heat at time t_i
        """

        # Calculate phi_k(i) phi_k(j)
        eigen_vec_prod = self.eigenvecs[idx_v1, :] * self.eigenvecs[idx_v2, :]
        # Calculate the expterms
        # exp(-\lambda_k t)
        exp_terms = np.exp(-self.eigenvals[:, None] * t_list[None, :])

        if is_normalized_kernel:
            exp_terms /= exp_terms.sum(axis=0, keepdims=True)

        # Return the heat value, of size (t,)
        return eigen_vec_prod @ exp_terms




def create_compability_matrix(graph1 : Graph, graph2 : Graph, 
                              t_list, is_normalized_kernel_hks, is_normalized_vector_hks,
                              use_physical_constraint = True
                              ):

    # Given two graphs as above, compute the compability matrix
    num_ver1 = len(graph1.vertices_list)
    num_ver2 = len(graph2.vertices_list)

    comp_mat = 1000 * np.ones((num_ver1 * num_ver2, num_ver1 * num_ver2))

    # Compute the hks of these matrix
    hks1 = graph1.compute_graph_hks(t_list, is_normalized_kernel_hks, is_normalized_vector_hks)
    hks2 = graph2.compute_graph_hks(t_list, is_normalized_kernel_hks, is_normalized_vector_hks)

    # Start filling this matrix
    for idx_i_g1, idx_j_g1 in zip(range(num_ver1), range(num_ver1)):
        for idx_a_g2, idx_b_g2 in zip(range(num_ver2), range(num_ver2)):
            
            # First order condition
            # Only calcultate if they are of the same types
            #  
            if idx_i_g1 == idx_j_g1 and idx_a_g2 == idx_b_g2 and graph1.get_node(idx_i_g1).node_type == graph2.get_node(idx_a_g2).node_type:
        
                # We get the hks at these points
                hks_i = hks1[idx_i_g1]
                hks_a = hks2[idx_a_g2]

                # HKS distance
                diff_hks = minkowski(hks_i, hks_a, 1)
                comp_mat[idx_i_g1 * idx_a_g2, idx_j_g1 * idx_b_g2] = diff_hks

                if use_physical_constraint:
                    # We want to apply the physical constraint as well
                    point1 = graph1.get_node(idx_i_g1)
                    point2 = graph2.get_node(idx_a_g2)

                    # Compute the euclidean distance between 2 points
                    e_dist = euclidean(point1.pos, point2.pos)
                    comp_mat[idx_i_g1 * idx_a_g2, idx_j_g1 * idx_b_g2] += e_dist

            # Second order condition 
            if idx_i_g1 != idx_j_g1 and idx_a_g2 != idx_b_g2:
                # Differences between the heat kernel

                heat_edge_g1 = graph1.compute_graph_heat_kernel(t_list, idx_i_g1, idx_j_g1, is_normalized_kernel_hks, is_normalized_vector_hks)
                heat_edge_g2 = graph2.compute_graph_heat_kernel(t_list, idx_a_g2, idx_b_g2, is_normalized_kernel_hks, is_normalized_vector_hks)

                diff_heat = minkowski(heat_edge_g1, heat_edge_g2)
                comp_mat[idx_i_g1 * idx_a_g2, idx_j_g1 * idx_b_g2] = diff_heat
    return comp_mat


def compute_matching(graph1 : Graph, graph2 : Graph, compatibility_mat):
    # Given the compatibility matrix, compute the optimal matching of nodes

    num_ver1 = len(graph1.vertices_list)
    num_ver2 = len(graph2.vertices_list)

    x = cp.Variable(num_ver1 * num_ver2, boolean=True)
    object_func = cp.Minimize(cp.quad_form(x, compatibility_mat))

    # Constraints
    # Build one to one mapping constraints
    A1 = np.zeros((num_ver1, num_ver1 * num_ver2))
    for i in range(num_ver1):
        for j in range(num_ver2):
            k = i * num_ver2 + j
            A1[i, k] = 1

    b1 = np.ones(num_ver1)

    # Each node in G2 match exactly to one in G1
    A2 = np.zeros((num_ver2, num_ver1 * num_ver2))
    for j in range(num_ver2):
        for i in range(num_ver1):
            k = i * num_ver2 + j
            A2[j, k] = 1
    b2 = np.ones(num_ver2)

    # Combine both constraints
    A = np.vstack((A1, A2))
    b = np.vstack((b1, b2))
    constraints = [A @ x <= b]

    # Solve
    problem = cp.Problem(objective=object_func, constraints=constraints)
    problem.solve(solver=cp.GUROBI, verbose=True)

    # Get the solution
    x_opt = np.round(x.value).astype(int)
    # Reshape to get the mapping
    map_opt = x_opt.reshape((num_ver1, num_ver2))

    return map_opt


def interpret_matching(mapping):
    matching = {node_g1 : np.argmax(mapping[node_g1]) for node_g1 in range(mapping.shape[0])}
    for start_node in matching:
        print(f"{start_node} of graph 1 to {matching[start_node]} of graph 2")