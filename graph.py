# Simple graph data structure that holds every information that we want to calculate

import numpy as np
from scipy.sparse.linalg import eigsh

import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt


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
    

    def build_adj_mat(self, use_edge_length = False):
        # Build adjacency matrix
        self.adj_mat = np.zeros((len(self.vertices_list), len(self.vertices_list)))

        for edge in self.edges_list:
            self.adj_mat[edge.start_node_idx, edge.end_node_idx] = self.adj_mat[edge.end_node_idx, edge.start_node_idx] = edge.weight if not use_edge_length else edge.weight * edge.length


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
                            self.minima_idx.append(second_node.idx)
                        case 1:
                            self.saddle_idx.append(second_node.idx)
                        case 2:
                            self.maxima_idx.append(second_node.idx)
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
        # if self.adj_mat is None or self.adj_mat.all():
        #     self.build_adj_mat()

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


    def compute_eigen_laplacian(self, num_eigen = 100, is_normalized=True):
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
            self.compute_graph_laplacian(is_normalized)

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


    def compute_graph_adalfs(self, t_max, num_sample=50, c_h=10):
        # Implementation of adaLFS 
        # Given a list of eigenvalues of a Laplacian of a graph
        # We need to optimize the signature h(|t-\lambda_i|) such that 
        # It minimize some gap

        # Given the time variable t_max, we only need to define h on [0, t_max]
        # Assume that we have h evenly sampled from [0, t_max]
        # To get h = [h_0,...,h_N]^\top
        # \delta h = t_max/N
        # h' = [(h_1 - h_0)/\delta h,..., (h_N - h_{N-1})/\delta h]^\top.
        # h'' = [(h_2 - h_0)/\delta h,...,(h_N - h_{N-2})/\delta h]^\top.

        # Let h_j be the shifted version

        # Problem to minimize 
        # min 
        #           \mu \sum_{t=0}^N \max(\hh_{1, t},\ldots,\hh_{k,t}) + \sum_{t=0}^N \max(\hh_{1,t}',\ldots,\hh_{k,t}')
        # s.t.      \hh_i \geq 0,\\
        #           \hh_i' \leq 0,\\
        #           -c_h \leq \hh_i'' \leq 0,\\
        #           h_0 = 1,\\
        #           h_N = 0.
        # \mu = 2 / (avg \lambda_i)

        if self.eigenvals is None or self.eigenvecs is None:
            self.compute_eigen_laplacian(True)

        # Calculate the mu value
        mu = 2 / np.mean(np.diff(self.eigenvals))
        # Sample step
        delta_h = t_max / num_sample

        # Compute eig_shift
        eig_shift = [int(round(val/delta_h)) for val in self.eigenvals]

        # Number of lambdas
        k = len(self.eigenvals)
        # Define GRB model
        model = gp.Model("adaLFS")
        model.Params.OutputFlag = 1

        # Variables
        h = model.addVars(num_sample + 1, lb=0.0, name="h") # Sample h
        h_p = model.addVars(num_sample, lb=-GRB.INFINITY, name="h_prime") # h'
        h_pp = model.addVars(num_sample - 1, lb=-GRB.INFINITY, name="h_prime_prime") # h''

        # Explicit shifted variables
        h_shifted       = model.addVars(k,  num_sample + 1, lb=0.0, name="h_shifted") # Shifted h
        h_p_shifted     = model.addVars(k,  num_sample, lb=-GRB.INFINITY, name="h_prime_shifted") # Shifted h'
        h_pp_shifted    = model.addVars(k,  num_sample - 1, lb=-GRB.INFINITY, name="h_prime_prime_shifted") # Shifted h''

        # Element-wise max variables
        max_h = model.addVars(num_sample + 1, lb=0.0, name="max_h") # max_h
        max_h_p = model.addVars(num_sample, lb=-GRB.INFINITY, name="max_h_prime") # max h'


        # Constraints
        # Boundary condition
        model.addConstr(h[0] == 1.0, name="h_0_fixed")
        model.addConstr(h[num_sample] == 0.0, name="h_N_fixed")

        # First derivative
        for i in range(num_sample):
            model.addConstr(h_p[i] == (h[i + 1] - h[i]) / delta_h, name=f"h_prime_def_{i}") # First derivative definition
            # model.addConstr(h_p[i] <= 0.0, name=f"h_prime_sign_{i}") # h decreasing constraint

        # Second derivative
        for i in range(num_sample - 1):
            model.addConstr(h_pp[i] == (h[i + 2] - h[i])/ (delta_h ** 2), name=f"h_prime_prime_def_{i}") # Second derivative defintion
            # model.addConstr(h_pp[i] >= -c_h, name=f"h_prime_prime_lb_{i}")  # Smooth constraint
            # model.addConstr(h_pp[i] <= 0, name=f"h_prime_prime_ub_{i}") # Smooth constraint

        # Shifted constraints for every h
        for idx, shift in enumerate(eig_shift):
            for t in range(num_sample + 1):
                h_shift_idx = t - shift
                if 0 <= h_shift_idx <= num_sample:
                    model.addConstr(h_shifted[idx, t] == h[h_shift_idx])
                else:
                    model.addConstr(h_shifted[idx, t] == 0)
                
                model.addConstr(h_shifted[idx, t] >= 0)

            # Shifted for every h'
            for t in range(num_sample):
                h_shift_idx = t - shift
                if 0 <= h_shift_idx <= num_sample - 1:
                    model.addConstr(h_p_shifted[idx, t] == h_p[h_shift_idx])
                else:
                    model.addConstr(h_p_shifted[idx, t] == 0)

                model.addConstr(h_p_shifted[idx, t] <= 0)
            
            # Shifted for every h''
            for t in range(num_sample - 1):
                h_shift_idx = t - shift
                if 0 <= h_shift_idx <= num_sample - 2:
                    model.addConstr(h_pp_shifted[idx, t] == h_pp[h_shift_idx])
                else:
                    model.addConstr(h_pp_shifted[idx, t] == 0)

                model.addConstr(h_pp_shifted[idx, t] >= -c_h)
                model.addConstr(h_pp_shifted[idx, t] <= c_h)

        # Maxima auxilary constrains
        for t in range(num_sample + 1):
            for idx in range(k):
                model.addConstr(max_h[t] >= h_shifted[idx, t])
        
        for t in range(num_sample):
            for idx in range(k):
                model.addConstr(max_h_p[t] >= h_p_shifted[idx, t])

        # # Element-wise shift for max
        # # h-shifted
        # for t in range(num_sample + 1):
        #     for idx, shift in enumerate(eig_shift):
        #         h_shift_idx = abs(t - shift)
        #         if 0 <= h_shift_idx <= num_sample:
        #             model.addConstr(max_h[t] >= h[h_shift_idx], name=f"max_h_def_{t}_{idx}") # Defintion of max shifted
        #         else:
        #             model.addConstr(max_h[t] == 0, name=f"max_h_def_{t}_{idx}") # 0 -padding
        # # h'-shifted
        # for t in range(num_sample):
        #     for idx, shift in enumerate(eig_shift):
        #         h_prime_shift_idx = abs(t - shift)
        #         if 0 <= h_prime_shift_idx <= num_sample - 1:
        #             model.addConstr(max_h_p[t] >= h_p[h_prime_shift_idx], name=f"max_h_prime_def_{t}_{idx}") # Definition of max shifted
        #         else:
        #             model.addConstr(max_h_p[t] == 0, name=f"max_h_prime_def_{t}_{idx}") # 0-padding
        
        # Objective function
        model.setObjective(mu * gp.quicksum(max_h[t] for t in range(num_sample + 1)) + gp.quicksum(max_h_p[t] for t in range(num_sample)), GRB.MINIMIZE)

        # Optimized
        model.optimize()

        # Check the result
        if model.Status == GRB.OPTIMAL:
            print("Optimal objective found", model.ObjVal)
            h_val = np.array([h[i].X for i in range(num_sample + 1)])
            x = np.linspace(0, t_max, num_sample + 1)
            plt.figure(figsize=(6,4))
            plt.plot(x, h_val,label='Optimized h(t)')
            plt.xlabel('t (input of h)')
            plt.ylabel('h(t)')
            plt.title('Optimized Spectral Kernel h(t)')
            plt.grid(True)
            plt.legend()
            plt.show()
            # for v in model.getVars():
            #     print(f"{v.VarName}: LB={v.LB}, UB={v.UB}, Type={v.VType}, Value={v.X}")

            # for c in model.getConstrs():
            #     print(f"{c.ConstrName}: {c.Sense} RHS={c.RHS}")
        else:
            print("Model is infeasible. Computing IIS to diagnose conflicting constraints...")
            model.computeIIS()
            # write IIS to file and print a short summary
            model.write("model_iis.ilp")
            print("IIS written to model_iis.ilp. The following constraints are in the IIS:")
            for c in model.getConstrs():
                if c.IISConstr:
                    print("  ", c.ConstrName)
            print("You can inspect model_iis.ilp in a text editor to see the minimal infeasible subsystem.")
        # else:
        #     print("Optimization ended with status:", model.status)