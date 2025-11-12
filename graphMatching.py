import numpy as np
from graph import Graph
from scipy.spatial.distance import cityblock, euclidean, minkowski

import cvxpy as cp

import gurobipy as gp
from gurobipy import GRB

def create_compability_matrix(graph1 : Graph, graph2 : Graph, 
                              t_list, is_normalized_kernel_hks, is_normalized_vector_hks,
                              use_physical_constraint = True
                              ):

    # Given two graphs as above, compute the compability matrix
    num_ver1 = len(graph1.vertices_list)
    num_ver2 = len(graph2.vertices_list)

    comp_mat = 10000000000 * np.ones((num_ver1 * num_ver2, num_ver1 * num_ver2))

    # Compute the hks of these matrix
    hks1 = graph1.hks
    hks2 = graph2.hks

    # Start filling this matrix
    for idx_i_g1 in range(num_ver1):
        for idx_j_g1 in range(num_ver1):
            for idx_a_g2 in range(num_ver2):
                for idx_b_g2 in range(num_ver2):

                    # Convert to the correct index
                    idx_ia = idx_a_g2 * num_ver1 + idx_i_g1
                    idx_jb = idx_b_g2 * num_ver2 + idx_j_g1
            
                    # First order condition
                    # Only calcultate if they are of the same types
                    #  
                    if idx_i_g1 == idx_j_g1 and idx_a_g2 == idx_b_g2 and graph1.get_node(idx_i_g1).node_type == graph2.get_node(idx_a_g2).node_type:
                
                        # We get the hks at these points
                        hks_i = hks1[idx_i_g1]
                        hks_a = hks2[idx_a_g2]
                        # HKS distance
                        diff_hks = minkowski(hks_i, hks_a, 1)
                        comp_mat[idx_ia, idx_jb] = diff_hks

                        if use_physical_constraint:
                            # We want to apply the physical constraint as well
                            point1 = graph1.get_node(idx_i_g1)
                            point2 = graph2.get_node(idx_a_g2)

                            # Compute the euclidean distance between 2 points
                            e_dist = euclidean(point1.pos, point2.pos)
                            comp_mat[idx_ia, idx_jb] += e_dist

                    # Second order condition 
                    if idx_i_g1 != idx_j_g1 and idx_a_g2 != idx_b_g2:
                        # Differences between the heat kernel

                        heat_edge_g1 = graph1.compute_graph_heat_kernel(t_list, idx_i_g1, idx_j_g1, is_normalized_kernel_hks, is_normalized_vector_hks)
                        heat_edge_g2 = graph2.compute_graph_heat_kernel(t_list, idx_a_g2, idx_b_g2, is_normalized_kernel_hks, is_normalized_vector_hks)

                        diff_heat = minkowski(heat_edge_g1, heat_edge_g2)
                        comp_mat[idx_ia, idx_jb] = diff_heat
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
    b = np.hstack((b1, b2))
    constraints = [A @ x <= b]

    # Solve
    problem = cp.Problem(objective=object_func, constraints=constraints)
    problem.solve(solver=cp.GUROBI, qcp=True, verbose=True)

    # Get the solution
    x_opt = np.round(x.value).astype(int)
    # Reshape to get the mapping
    map_opt = x_opt.reshape((num_ver1, num_ver2))

    return map_opt


def interpret_matching(mapping):
    matching = {node_g1 : np.argmax(mapping[node_g1]) for node_g1 in range(mapping.shape[0])}
    for start_node in matching:
        print(f"Node {start_node} of graph 1 to Node {matching[start_node]} of graph 2")


def compute_matching_grb(graph1: Graph, graph2: Graph, compatibility_mat):
     # Given the compatibility matrix, compute the optimal matching of nodes

    num_ver1 = len(graph1.vertices_list)
    num_ver2 = len(graph2.vertices_list)

    model = gp.Model("MSC Matching")

    # Var
    x = model.addMVar(shape=num_ver1 * num_ver2, vtype=GRB.BINARY, name="x")

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

    A = np.vstack((A1, A2))
    b = np.ones(A.shape[0])

    # Add constraints
    model.addMConstr(A, x, GRB.EQUAL, b)

    object_funct = x.T @ compatibility_mat @ x
    model.setObjective(object_funct, GRB.MINIMIZE)

    # Optimize
    model.optimize()

    # Solution
    x_opt = x.X
    # Reshape
    map_opt = x_opt.reshape((num_ver1, num_ver2))
    return map_opt