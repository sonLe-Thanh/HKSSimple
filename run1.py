from graphHKS import *
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, cosine

from utils import symmetrize_matrix, popoluate_adjacency_matrix
from mpl_toolkits.mplot3d import Axes3D  # needed for older Matplotlib versions
# We would want to know the effect of the weights of the graph to the Graph HKS


# We construct a simple graph with 2 vertices

possible_weights = np.linspace(0, 20, 1000)

# Parameters for graph HKS
is_normalized_Laplacian = True
is_normalize_hks = True
num_eigen = 3

t_hks = np.linspace(0.01, 10, 1000)



other_weights = [0, 1, 2, 5]

save_path = "WeightEffects/"
for other_weight in other_weights:
    all_hks = []
    all_hks_0 = []
    all_hks_1 = []
    all_hks_2 = []
    for weight in possible_weights:
        adjacent_matrix = np.array([[0, weight, 0],
                                    [weight, 0, other_weight],
                                    [0, other_weight, 0]])
        
        
        # Compute the eigen values 
        eigen_val, eigen_vec, laplacian = compute_eigen_laplacian(adjacent_matrix, num_eigen, is_normalized_Laplacian)
        # Compute the HKS
        hks = compute_graph_hks(eigen_val, eigen_vec, t_hks, is_normalize_hks)
        all_hks_0.append(hks[0])
        all_hks_1.append(hks[1])
        all_hks_2.append(hks[2])

        # print(np.linalg.norm(hks[0] - hks[2], ord=2))


    # Draw the values


    all_hks_0 = np.array(all_hks_0)
    all_hks_1 = np.array(all_hks_1)
    all_hks_2 = np.array(all_hks_2)

    # print(np.linalg.norm(all_hks_0 - all_hks_2, ord=2))

    # rows, cols = all_hks_0.shape
    # X, Y = np.meshgrid(range(cols), range(rows))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, all_hks_1)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('all_hks_0')

    # plt.show()


    vmin = min(all_hks_0.min(), all_hks_1.min(), all_hks_2.min())
    vmax = max(all_hks_0.max(), all_hks_1.max(), all_hks_2.max())

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    datasets = [all_hks_0, all_hks_1, all_hks_2]
    titles = ["Effects of Weight - Node 0",
            "Effects of Weight - Node 1",
            "Effects of Weight - Node 2"]

    for ax, data, title in zip(axes, datasets, titles):
        im = ax.imshow(
            data,
            cmap="plasma",
            interpolation="nearest",
            extent=[t_hks.min(), t_hks.max(),
                    possible_weights.min(), possible_weights.max()],
            aspect="equal",
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(title)
        ax.set_xlabel("t")

    # Set common y-label
    axes[0].set_ylabel("weight")

    # Add a single colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label="Graph HKS Value")

    # plt.tight_layout()
    plt.savefig(save_path+f"otherWeight_{other_weight}.pdf", format="pdf")
    # plt.show()
