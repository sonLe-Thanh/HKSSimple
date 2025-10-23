import numpy as np 

class IvwCriticalPoints:
    def __init__(self, idx_wpc, val, node_type):
        self.idx_wpc = idx_wpc
        self.val = val
        self.node_type = node_type
        # Maybe save the graph idx?
        self.idx_graph = -1
        self.pos = np.zeros(3)

    def __str__(self):
        return f"Node {self.idx_wpc} of type {self.node_type}, val {self.val}, idx in graph {self.idx_graph}, pos ({self.pos[0]}, {self.pos[1]}, {self.pos[2]})"

# Read the data from the file
def read_graph_info(in_file):
    results = []

    all_idx_wpc = {}
    minima_idx = []
    saddle_idx = []
    maxima_idx = []

    all_nodes = []

    idx_graph_counter = 0
    with open(in_file, "r") as file:
        for line in file:
            parts = line.split(",")
            # The first one is the lenght of the separatrices
            sepa_length = int(parts[0])
            
            # After that we have the node info
            first_node_info = parts[2].split(" ")
            first_node = IvwCriticalPoints(int(first_node_info[0]), float(first_node_info[1]), int(first_node_info[2]))
            # Set the correct graph index
            if first_node.idx_wpc not in all_idx_wpc:
                # This node is new
                all_idx_wpc[first_node.idx_wpc] = idx_graph_counter
                match first_node.node_type:
                    case 0:
                        minima_idx.append(idx_graph_counter)
                    case 1:
                        saddle_idx.append(idx_graph_counter)
                    case 2:
                        maxima_idx.append(idx_graph_counter)
                    case _:
                        # Do nothing
                        print("Not supporting atm")

                idx_graph_counter += 1
                all_nodes.append(first_node)
            # Update the idx graph
            first_node.idx_graph = all_idx_wpc[first_node.idx_wpc]

            # Update the positions
            first_node_pos = parts[4].split(" ")
            first_node.pos = np.array([float(first_node_pos[0]), float(first_node_pos[1]), float(first_node_pos[2])])

            second_node_info = parts[3].split(" ")
            second_node = IvwCriticalPoints(int(second_node_info[0]), float(second_node_info[1]), int(second_node_info[2]))
            # Set the correct graph index
            if second_node.idx_wpc not in all_idx_wpc:
                # This node is new
                all_idx_wpc[second_node.idx_wpc] = idx_graph_counter
                match second_node.node_type:
                    case 0:
                        minima_idx.append(idx_graph_counter)
                    case 1:
                        saddle_idx.append(idx_graph_counter)
                    case 2:
                        maxima_idx.append(idx_graph_counter)
                    case _:
                        # Do nothing
                        print("Not supporting atm")
                idx_graph_counter += 1
                all_nodes.append(second_node)
            # Update the idx graph
            second_node.idx_graph = all_idx_wpc[second_node.idx_wpc]

            # Update the positions
            second_node_pos = parts[5].split(" ")
            second_node.pos = np.array([float(second_node_pos[0]), float(second_node_pos[1]), float(second_node_pos[2])])

            # Add to the result
            results.append([sepa_length, first_node, second_node])


    # Also keep track of the number of nodes in the graph
    return results, idx_graph_counter, minima_idx, saddle_idx, maxima_idx

# Create an adjacency matrix from the list
def create_adj_matrix_from_ivw(in_file, use_sepa_length=False):
    edges, num_nodes, minima_idx, saddle_idx, maxima_idx = read_graph_info(in_file)

    # Create a zero matrix
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # Loop for every read separatrices
    for sepa in edges:
        # Edge length
        sepa_length = sepa[0]
        # Node info
        first_node_info = sepa[1]
        second_node_info = sepa[2]
        # Node weight
        sepa_weight = abs(first_node_info.val - second_node_info.val)

        # Mark the adj_matrix
        adj_matrix[first_node_info.idx_graph, second_node_info.idx_graph] = adj_matrix[second_node_info.idx_graph, first_node_info.idx_graph] = sepa_weight * sepa_length if use_sepa_length else sepa_weight

    return adj_matrix, minima_idx, saddle_idx, maxima_idx