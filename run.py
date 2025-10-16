from graphHKS import *
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, cosine

from utils import symmetrize_matrix, popoluate_adjacency_matrix
# Note to me: HKS can be interesting, but I should also considering how to incoorporate the scalar values at the vertices
# We may want to look at the heat kernel only.



# We may also want to have that extra bits of scalar at the end when finding the coorespondence

# Now create an example of simple MS cell

# Assume that we have the unweighted adjacent matrix, and the scalar values at the vertices

unweighted_adjacency = np.zeros((11, 11))

unweighted_adjacency[0, 1]  = 1
unweighted_adjacency[1, 2]  = 1
unweighted_adjacency[3, 4]  = 1
unweighted_adjacency[4, 5]  = 1
unweighted_adjacency[5, 6]  = 1
unweighted_adjacency[6 ,7]  = 1
unweighted_adjacency[8, 9]  = 1
unweighted_adjacency[9, 10] = 1

unweighted_adjacency[0, 3]  = 1
unweighted_adjacency[1, 4]  = 1
unweighted_adjacency[2, 5]  = 1
unweighted_adjacency[2, 7]  = 1
unweighted_adjacency[3, 8]  = 1
unweighted_adjacency[4, 5]  = 1
unweighted_adjacency[5, 8]  = 1
unweighted_adjacency[6, 9]  = 1
unweighted_adjacency[7, 10] = 1


unweighted_adjacency = symmetrize_matrix(unweighted_adjacency)


# Now define the first set of the vertices weight
# vertices_weight_1 = -np.array([0, 3.2, 1, 3.3, 6, 3, 5, 3.4, 2, 3.5, 2.5])
# vertices_weight_2 = -np.array([0, 3.2, 1, 3.3, 7, 3, 5, 3.4, 2, 3.5, 2.5])

# Symmetric weights
# vertices_weight_1 = -np.array([0, 3.2, 0, 3.2, 6, 3, 5, 3.2, 0, 3.2, 0])
# vertices_weight_2 = -np.array([0, 3.2, 0, 3.2, 7, 3, 5, 3.2, 0, 3.2, 0])

vertices_weight_1 = np.array([0, 3.2, 1, 3.3, 6, 3, 5, 3.4, 2, 3.5, 2.5])
vertices_weight_2 = np.array([0, 3.2, 1, 3.3, 4, 3, 5, 3.4, 2, 3.5, 2.5])

# Build the weighted graph
# weighted_adjacency = popoluate_adjacency_matrix(unweighted_adjacency, vertices_weight_1)
# weighted_adjacency2 = popoluate_adjacency_matrix(unweighted_adjacency, vertices_weight_2)

# Test with unweighted graph
weighted_adjacency = popoluate_adjacency_matrix(unweighted_adjacency, vertices_weight_1)
weighted_adjacency2 = popoluate_adjacency_matrix(unweighted_adjacency, vertices_weight_2)

# Parameters for graph HKS
is_normalized_Laplacian = True
is_normalize_hks = True
num_eigen = 11


# Compute the eigen of the Laplacian
eigen_val, eigen_vec, laplacian1 = compute_eigen_laplacian(weighted_adjacency, num_eigen, is_normalized_Laplacian)
eigen_val2, eigen_vec2, laplacian2 = compute_eigen_laplacian(weighted_adjacency2, num_eigen, is_normalized_Laplacian)

t_min = 1.0 / eigen_val.max()
t_max = 1.0 / eigen_val[eigen_val > 1e-12].min()

# t_hks =  np.logspace(np.log10(t_max), np.log10(t_min), num=100)
t_hks = np.linspace(0.01, 20, 1000)


# Compute the HKS
hks1 = compute_graph_hks(eigen_val, eigen_vec, t_hks, is_normalize_hks)


# Compute the HKS
hks2 = compute_graph_hks(eigen_val2, eigen_vec2, t_hks, is_normalize_hks)



# Let see the diffused graph Laplacian

diffused_hks1 = compute_diffused_graph_hks(eigen_val, eigen_vec, vertices_weight_1, t_hks, is_normalize_hks)
diffused_hks2 = compute_diffused_graph_hks(eigen_val2, eigen_vec2, vertices_weight_2, t_hks, is_normalize_hks)


plt.figure(figsize=(10, 10))

# Get the default color cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for i in range(hks1.shape[0]):
    color = colors[i % len(colors)]
    plt.plot(t_hks, hks1[i], label=f"Vertex {i}", color=color)
    plt.plot(t_hks, hks2[i], label=f"Vertex {i} Graph 2", linestyle="--", color=color)

plt.xlabel("t value")
plt.ylabel("Diffused HKS Value")

plt.title("Diffused HKS on Graph")
plt.legend()
plt.show()


dist = np.linalg.norm(hks1[:, np.newaxis, :] - hks2[np.newaxis, :, :], axis=2)



minima = [0, 2, 8, 10]
minima_gr1 = [0, 2]
minima_gr2 = [8, 10]
saddles = [1, 3, 5, 7, 9]
saddles_gr1 = [1, 3, 5]
saddles_gr2 = [7, 9]
maxima = [4, 6]

hks1_min    = hks1[minima, :]
hks1_max    = hks1[maxima, :]
hks1_saddle = hks1[saddles, :]


hks1_min_gr1    = hks1[minima_gr1, :]
hks1_min_gr2    = hks1[minima_gr2, :]
hks1_saddle_gr1 = hks1[saddles_gr1, :]
hks1_saddle_gr2 = hks1[saddles_gr2, :]

hks2_min    = hks2[minima, :]
hks2_max    = hks2[maxima, :]
hks2_saddle = hks2[saddles, :]


hks2_min_gr1    = hks2[minima_gr1, :]
hks2_min_gr2    = hks2[minima_gr2, :]
hks2_saddle_gr1 = hks2[saddles_gr1, :]
hks2_saddle_gr2 = hks2[saddles_gr2, :]

dist_min    = np.linalg.norm(hks1_min[:, np.newaxis, :]     - hks2_min[np.newaxis, :, :],       axis=2)
dist_max    = np.linalg.norm(hks1_max[:, np.newaxis, :]     - hks2_max[np.newaxis, :, :],       axis=2)
dist_saddle = np.linalg.norm(hks1_saddle[:, np.newaxis, :]  - hks2_saddle[np.newaxis, :, :],    axis=2)


cosine_max      = cdist(hks1_max,       hks2_max,       metric="cosine")
cosine_min      = cdist(hks1_min,       hks2_min,       metric="cosine")
cosine_saddle   = cdist(hks1_saddle,    hks2_saddle,    metric="cosine")

# Eucledean
dist_min_gr1    = np.linalg.norm(hks1_min_gr1[:, np.newaxis, :]     - hks2_min_gr1[np.newaxis, :, :],       axis=2)
dist_min_gr2    = np.linalg.norm(hks1_min_gr2[:, np.newaxis, :]     - hks2_min_gr2[np.newaxis, :, :],       axis=2)
dist_saddle_gr1 = np.linalg.norm(hks1_saddle_gr1[:, np.newaxis, :]  - hks2_saddle_gr1[np.newaxis, :, :],    axis=2)
dist_saddle_gr2 = np.linalg.norm(hks1_saddle_gr2[:, np.newaxis, :]  - hks2_saddle_gr2[np.newaxis, :, :],    axis=2)

# Cosine
cosine_min_gr1      = cdist(hks1_min_gr1,       hks2_min_gr1,       metric="cosine")
cosine_min_gr2      = cdist(hks1_min_gr2,       hks2_min_gr2,       metric="cosine")
cosine_saddle_gr1   = cdist(hks1_saddle_gr1,    hks2_saddle_gr1,    metric="cosine")
cosine_saddle_gr2   = cdist(hks1_saddle_gr2,    hks2_saddle_gr2,    metric="cosine")

save_path = "SimpleCase1_DecreaseMax/"

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(dist, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Euclidean distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Pairwise Distances Heatmap')
plt.xticks(range(11))
plt.yticks(range(11))
plt.savefig(save_path+"Euclidean_All.pdf", format="pdf")
plt.show()


plt.figure(figsize=(8, 6))
plt.imshow(dist_min, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Euclidean distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Pairwise Distances Heatmap - Min')
plt.xticks(range(4))
plt.yticks(range(4))
plt.savefig(save_path+"Euclidean_Min.pdf", format="pdf")
plt.show()


plt.figure(figsize=(8, 6))
plt.imshow(dist_max, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Euclidean distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Pairwise Distances Heatmap - Max')
plt.xticks(range(2))
plt.yticks(range(2))
plt.savefig(save_path+"Euclidean_Max.pdf", format="pdf")
plt.show()



plt.figure(figsize=(8, 6))
plt.imshow(dist_saddle, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Euclidean distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Pairwise Distances Heatmap - Saddle')
plt.xticks(range(5))
plt.yticks(range(5))
plt.savefig(save_path+"Euclidean_Saddle.pdf", format="pdf")
plt.show()


plt.figure(figsize=(8, 6))
plt.imshow(cosine_min, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Cosine distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Cosine Distances Heatmap - Min')
plt.xticks(range(4))
plt.yticks(range(4))
plt.savefig(save_path+"Cosine_Min.pdf", format="pdf")
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(cosine_saddle, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Cosine distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Cosine Distances Heatmap - Saddle')
plt.xticks(range(4))
plt.yticks(range(4))
plt.savefig(save_path+"Cosine_Saddle.pdf", format="pdf")
plt.show()


plt.figure(figsize=(8, 6))
plt.imshow(cosine_max, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Cosine distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Cosine Distances Heatmap - Max')
plt.xticks(range(2))
plt.yticks(range(2))
plt.savefig(save_path+"Cosine_Max.pdf", format="pdf")
plt.show()

# Min group
plt.figure(figsize=(8, 6))
plt.imshow(dist_min_gr1, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Euclidean distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Pairwise Distances Heatmap - Min - Group1')
plt.xticks(range(len(minima_gr1)))
plt.yticks(range(len(minima_gr1)))
plt.savefig(save_path+"Euclidean_Min_gr1.pdf", format="pdf")
plt.show()


plt.figure(figsize=(8, 6))
plt.imshow(dist_min_gr2, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Euclidean distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Pairwise Distances Heatmap - Min - Group2')
plt.xticks(range(len(minima_gr2)))
plt.yticks(range(len(minima_gr2)))
plt.savefig(save_path+"Euclidean_Min_gr2.pdf", format="pdf")
plt.show()


plt.figure(figsize=(8, 6))
plt.imshow(cosine_min_gr1, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Cosine distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Cosine Distances Heatmap - Min - Group1')
plt.xticks(range(len(minima_gr1)))
plt.yticks(range(len(minima_gr1)))
plt.savefig(save_path+"Cosine_Min_gr1.pdf", format="pdf")
plt.show()


plt.figure(figsize=(8, 6))
plt.imshow(cosine_min_gr2, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Cosine distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Cosine Distances Heatmap - Min - Group2')
plt.xticks(range(len(minima_gr2)))
plt.yticks(range(len(minima_gr2)))
plt.savefig(save_path+"Cosine_Min_gr2.pdf", format="pdf")
plt.show()


# Saddles group

plt.figure(figsize=(8, 6))
plt.imshow(dist_saddle_gr1, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Euclidean distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Pairwise Distances Heatmap - Saddles - Group1')
plt.xticks(range(len(saddles_gr1)))
plt.yticks(range(len(saddles_gr1)))
plt.savefig(save_path+"Euclidean_Saddles_gr1.pdf", format="pdf")
plt.show()


plt.figure(figsize=(8, 6))
plt.imshow(dist_saddle_gr2, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Euclidean distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Pairwise Distances Heatmap - Saddles - Group2')
plt.xticks(range(len(saddles_gr2)))
plt.yticks(range(len(saddles_gr2)))
plt.savefig(save_path+"Euclidean_Saddles_gr2.pdf", format="pdf")
plt.show()


plt.figure(figsize=(8, 6))
plt.imshow(cosine_saddle_gr1, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Cosine distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Cosine Distances Heatmap - Saddles - Group1')
plt.xticks(range(len(saddles_gr1)))
plt.yticks(range(len(saddles_gr1)))
plt.savefig(save_path+"Cosine_Saddles_gr1.pdf", format="pdf")
plt.show()


plt.figure(figsize=(8, 6))
plt.imshow(cosine_saddle_gr2, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Cosine distance of HKS')
plt.xlabel('Graph 2')
plt.ylabel('Graph 1')
plt.title('Cosine Distances Heatmap - Saddles - Group2')
plt.xticks(range(len(minima_gr2)))
plt.yticks(range(len(minima_gr2)))
plt.savefig(save_path+"Cosine_Saddles_gr2.pdf", format="pdf")
plt.show()
# The GraphHKS seems to work fine ????