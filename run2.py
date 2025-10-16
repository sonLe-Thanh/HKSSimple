from graphHKS import *
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, cosine

from utils import symmetrize_matrix, popoluate_adjacency_matrix
from mpl_toolkits.mplot3d import Axes3D  # needed for older Matplotlib versions


# We would want to know the effect of the connectivity of a node

def create_graph(n, weight=1):
    adj_mat = np.zeros((n, n))
    adj_mat[0, :] = weight # Row
    adj_mat[:, 0] = weight # Column 
    adj_mat[0, 0] = 0 # Not on the diagonal

    return adj_mat


save_path = "ConnectivityEffect/"

possible_weights = np.linspace(1, 5, 5)

possible_connectivity = np.arange(3, 51, 1)

t_hks = np.linspace(0.01, 10, 1000)

is_normalized_Laplacian = True
is_normalize_hks = True

for weight in possible_weights:
    all_hks_main = []
    all_hks_sub = []
    for deg in possible_connectivity:
        adj_mat = create_graph(deg, weight)
        
        # Compute the eigen values
        num_eigen = int(deg) + 1 
        eigen_val, eigen_vec, laplacian = compute_eigen_laplacian(adj_mat, num_eigen, is_normalized_Laplacian)
        # Compute the HKS
        hks = compute_graph_hks(eigen_val, eigen_vec, t_hks, is_normalize_hks)

        # Only care about the main node and one sub node
        all_hks_main.append(hks[0])
        all_hks_sub.append(hks[1])

    # Draw the values
    all_hks_main = np.array(all_hks_main[::-1])
    all_hks_sub = np.array(all_hks_sub[::-1])

    vmin = min(all_hks_main.min(), all_hks_sub.min())
    vmax = max(all_hks_main.max(), all_hks_sub.max())

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    datasets = [all_hks_main, all_hks_sub]
    titles = ["Effects of Connectivity - Main Node",
            "Effects of Connectivity - Sub Node"]
    
    for ax, data, title in zip(axes, datasets, titles):
        im = ax.imshow(
            data,
            cmap="plasma",
            interpolation="nearest",
            extent=[t_hks.min(), t_hks.max(),
                    possible_connectivity.min(), possible_connectivity.max() ],
            aspect="equal",
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(title)
        ax.set_xlabel("t")

    # Set common y-label
    axes[0].set_ylabel("Main Node Degree")

    # Add a single colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label="Graph HKS Value")

    # plt.tight_layout()
    plt.savefig(save_path+f"HKS_{weight}.pdf", format="pdf")