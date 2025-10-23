from graphHKS import *
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, cosine

from utils import symmetrize_matrix, popoluate_adjacency_matrix
from mpl_toolkits.mplot3d import Axes3D  # needed for older Matplotlib vers
from graphFromIvw import create_adj_matrix_from_ivw


file1 = "IvwData/Case1/Timestep1.txt"
file2 = "IvwData/Case1/Timestep2.txt"

# t_hks = np.linspace(0.01, 80, 1000)
t_hks = np.logspace(1, 10, 100)

is_normalized_Laplacian = True
is_normalize_hks = True
is_normalize_vectors = False

use_sepa_length = False
sigma = 0.7

# Get the adjacency matrix
adj_mat_t1, minima_idx1, saddle_idx1, maxima_idx1 = create_adj_matrix_from_ivw(file1, use_sepa_length)
adj_mat_t2, minima_idx2, saddle_idx2, maxima_idx2 = create_adj_matrix_from_ivw(file2, use_sepa_length)

num_eigen = adj_mat_t1.shape[0] + 1
# Compute the eigen values
eigen_val1, eigen_vec1, laplacian1 = compute_eigen_laplacian(adj_mat_t1, num_eigen, is_normalized_Laplacian)
eigen_val2, eigen_vec2, laplacian2 = compute_eigen_laplacian(adj_mat_t2, num_eigen, is_normalized_Laplacian)

# print(eigen_vec1.shape)
# print(eigen_val1.shape)

# Compute the hks
hks1 = compute_graph_hks(eigen_val1, eigen_vec1, t_hks, is_normalize_hks, is_normalize_vectors)
hks2 = compute_graph_hks(eigen_val2, eigen_vec2, t_hks, is_normalize_hks, is_normalize_vectors)
# 
# Compute the wks
# hks1 = compute_graph_wks(eigen_val1, eigen_vec1, t_hks, sigma, is_normalize_hks, is_normalize_vectors)
# hks2 = compute_graph_wks(eigen_val2, eigen_vec2, t_hks, sigma, is_normalize_hks, is_normalize_vectors)

# minima_idx1.pop(2)
# minima_idx2.pop(2)


# We want to only compare the critical points with the same type
hks1_min    = hks1[minima_idx1]
hks1_saddle = hks1[saddle_idx1]
hks1_max    = hks1[maxima_idx1]


hks2_min    = hks2[minima_idx2]
hks2_saddle = hks2[saddle_idx2]
hks2_max    = hks2[maxima_idx2]

# print(hks1_min.shape)

# Compute the differences between the points
dist_min_e    = cdist(hks1_min,       hks2_min,       metric="cityblock")
dist_max_e    = cdist(hks1_max,       hks2_max,       metric="cityblock")
dist_saddle_e = cdist(hks1_saddle,    hks2_saddle,    metric="cityblock")


dist_min_cosine    = cdist(hks1_min,       hks2_min,       metric="cosine")
dist_max_cosine    = cdist(hks1_max,       hks2_max,       metric="cosine")
dist_saddle_cosine = cdist(hks1_saddle,    hks2_saddle,    metric="cosine")

# Plot heatmap
val_min = min(hks1.min(), hks2.min())
val_max = max(hks1.max(), hks2.max())


plt.figure(figsize=(8, 6))
plt.imshow(dist_min_e, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Euclidean distance of HKS - Min')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Pairwise Distances Heatmap')
# plt.xticks(range(11))
# plt.yticks(range(11))
plt.show()


plt.figure(figsize=(8, 6))
plt.imshow(dist_saddle_e, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Euclidean distance of HKS - Saddles')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Pairwise Distances Heatmap')
# plt.xticks(range(11))
# plt.yticks(range(11))
plt.show()



plt.figure(figsize=(8, 6))
plt.imshow(dist_max_e, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Euclidean distance of HKS - Max')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Pairwise Distances Heatmap')
# plt.xticks(range(11))
# plt.yticks(range(11))
plt.show()


# plt.figure(figsize=(8, 6))
# plt.imshow(dist_min_cosine, cmap='viridis', interpolation='nearest')
# plt.colorbar(label='Cosine distance of HKS - Min')
# plt.xlabel('Graph 2')
# plt.ylabel('Graph 1')
# plt.title('Pairwise Distances Heatmap')
# # plt.xticks(range(11))
# # plt.yticks(range(11))
# plt.show()


# plt.figure(figsize=(8, 6))
# plt.imshow(dist_saddle_cosine, cmap='viridis', interpolation='nearest')
# plt.colorbar(label='Cosine distance of HKS - Saddles')
# plt.xlabel('Graph 2')
# plt.ylabel('Graph 1')
# plt.title('Pairwise Distances Heatmap')
# # plt.xticks(range(11))
# # plt.yticks(range(11))
# plt.show()



# plt.figure(figsize=(8, 6))
# plt.imshow(dist_max_cosine, cmap='viridis', interpolation='nearest')
# plt.colorbar(label='Consine distance of HKS - Max')
# plt.xlabel('Graph 2')
# plt.ylabel('Graph 1')
# plt.title('Pairwise Distances Heatmap')
# # plt.xticks(range(11))
# # plt.yticks(range(11))
# plt.show()



# We may want a proper optimization scheme, the normal distance only showed some sights