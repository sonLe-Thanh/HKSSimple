import numpy as np
from scipy.sparse.linalg import eigsh

def compute_laplacian_matrix(A, is_nomalized = False):
    """
    Compute a Laplacian matrix given an adjacency matrix of a graph G=(V,E)

    Input:
        A : np.array - adjacency matrix of size n x n
        is_normalized: bool - decide whether to normalize this matrix or not

    Output:
        L: np.array - Laplacian matrix of G of size n x n
    """
    # Compute the matrix D 
    # Summing over rows
    diag_entries = A.sum(axis=1)

    if not is_nomalized:
        D = np.diag(diag_entries)
        return D - A
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
        identity_mat = np.eye(A.shape[0])
        return identity_mat - D_sqrt_inv @ A @ D_sqrt_inv


def compute_eigen_laplacian(A, num_eigen, is_nomalized = False):
    """
    Compute the eigenvalues and eigen vectors of the Laplacian matrix of the Graph G(V,E)

    Input:
        A : np.array - adjacency matrix of size n x n
        num_eigen: number of eigenvalues we want to calculate
        is_normalized: bool - decide whether to normalize this matrix or not

    Output:
        eigenvalues, eigenvectors: eigen values and eigen vectors of L, the Laplacian Matrix associating to G
    """
    laplacian = compute_laplacian_matrix(A, is_nomalized)
    num_eigen_calc = min(A.shape[0], num_eigen)
    (eigenvals, eigenvectors) = eigsh(laplacian, num_eigen_calc, which="SM", sigma=0.)

    return eigenvals, eigenvectors, laplacian


def compute_graph_hks(eigenvals, eigenvectors, t_list, is_normalized_kernel = False, is_normalized_vectors = False):
    """
    Compute the heat kernel signature, given eigenvalues, eigenvectors, and the scaling parameter t
    The discrete HKS is given as 
        HKS(t,x) = \Sigma_{i=1}^k e^{-\lambda_i t} v_i^2,
        where \lambda_i and v_i are the corresponding eigenvalues and eigenvectors of the Laplacian
    

    Input:
        eigenvals : np.array - eigenvalues of the Laplacian
        eigenvectors : np.array - corresponding eigenvectors of the Laplacian
        t_list: np.array - scaling variables
        is_normalized_kernel: do we want to normalized the HKS with the heat trace
        is_normalized_vectors: do we want to normalized the eigenvectors

    Output:
        hks: np.array
    """

    num_vertices = eigenvectors.shape[0]
    hks = np.zeros((num_vertices, len(t_list)))

    for idx, t in enumerate(t_list):
        exp_term = np.exp(-t * eigenvals)
        if is_normalized_vectors:
            eigenvectors_norm = eigenvectors / np.linalg.norm(eigenvectors, axis=0, keepdims=True)
            hks[:, idx] = (eigenvectors_norm ** 2) @ exp_term
        else:
            hks[:, idx] = (eigenvectors ** 2) @ exp_term

        if is_normalized_kernel:
            heat_trace = np.sum(exp_term)
            hks[:, idx] /= heat_trace
    # Each column correcspond to the hks at that i-th vertex
    return hks


def compute_graph_wks(eigenvals, eigenvectors, t_list, sigma, is_normalized_kernel = False, is_normalized_vectors = False):
    """
    Compute the wave kernel signature, given eigenvalues, eigenvectors, and the scaling parameter t
    The discrete HKS is given as 
        HKS(t,x) = \Sigma_{i=1}^k e^{-(t - log \lamba_k)^2 / (2 \sigma^2)} v_i^2,
        where \lambda_i and v_i are the corresponding eigenvalues and eigenvectors of the Laplacian
    

    Input:
        eigenvals : np.array - eigenvalues of the Laplacian
        eigenvectors : np.array - corresponding eigenvectors of the Laplacian
        t_list: np.array - scaling variables
        sigma: value: the value sigma for wave kernel signature
        is_normalized_value: do we want to normalized the WKS with the 'wave trace'
        is_normalized_vectors: do we want to normalized the eigenvectors

    Output:
        hks: np.array
    """


    num_vertices = eigenvectors.shape[0]
    wks = np.zeros((num_vertices, len(t_list)))

    for idx, t in enumerate(t_list):
        exp_term = np.exp(-(t  - np.log(eigenvals)) ** 2 / (2 * sigma ** 2))

        if is_normalized_vectors:
            eigenvectors_norm = eigenvectors / np.linalg.norm(eigenvectors, axis=0, keepdims=True)
            wks[:, idx] = (eigenvectors_norm ** 2) @ exp_term
        else:    
            wks[:, idx] = (eigenvectors ** 2) @ exp_term

        if is_normalized_kernel:
            heat_trace = np.sum(exp_term)
            wks[:, idx] /= heat_trace

    return wks


def compute_diffused_graph_hks(eigenvals, eigenvectors, scalar_function, t_list, is_normlized = False):
    """
    Compute the diffused heat kernel signature, given eigenvalues, eigenvectors, scalar_function, and the scaling parameter t
    The discrete HKS is given as 
        HKS(t,x, f) = \Sigma_{i=1}^k e^{-\lambda_i t} v_i^2 f,
        where \lambda_i and v_i are the corresponding eigenvalues and eigenvectors of the Laplacian
    

    Input:
        eigenvals : np.array - eigenvalues of the Laplacian
        eigenvectors : np.array - corresponding eigenvectors of the Laplacian
        scalar_function: np.array - scalar function at the vertices of the graph
        t_list: np.array - scaling variables
        is_normalized: do we want to normalized the HKS with the heat trace

    Output:
        dhks: np.array
    """

    num_vertices = eigenvectors.shape[0]
    dhks = np.zeros((num_vertices, len(t_list)))

    for idx, t in enumerate(t_list):
        exp_term = np.exp(-t * eigenvals)
        dhks[:, idx] = ((eigenvectors ** 2) @ exp_term) * scalar_function 

        if is_normlized:
            heat_trace = np.sum(exp_term)
            dhks[:, idx] /= heat_trace

    return dhks



def compute_graph_heat_kernel(laplacian, t_list):
    """
    Compute the heat kernel, given the Laplacian, and the scaling parameter t
    The graph heat kernel is given as 
        H_t = e^{-Laplacian t},
    

    Input:
        laplacian: np.array: Laplacian matrix of G(V, E)
        t_list: np.array - scaling variables

    Output:
        heat_kernel: np.array: 3-d array for the heat kernel 
    """

    # Laplacian is always a positive semi-definite matrix
    num_vertices = laplacian.shape[0]
    heat_kernel = np.zeros((num_vertices, num_vertices, len(t_list)))

    for idx, t in enumerate(t_list):
        exp_term = np.exp(-t * laplacian)
        heat_kernel[:, :, idx] = exp_term

    return heat_kernel

def compute_diffused_graph_heat_kernel(laplacian, scalar_value, t_list):
    """
    Compute the heat kernel, given the Laplacian, and the scaling parameter t
    The graph heat kernel is given as 
        H_tf = e^{-Laplacian t} f,
    

    Input:
        laplacian: np.array: Laplacian matrix of G(V, E)
        scalar_value: np.array: scalar values at the vertices of the graph
        t_list: np.array - scaling variables

    Output:
        diffused_heat_kernel: np.array: 3-d array for the heat kernel 
    """

    # Laplacian is always a positive semi-definite matrix
    num_vertices = laplacian.shape[0]
    heat_kernel = np.zeros((num_vertices, num_vertices, len(t_list)))

    for idx, t in enumerate(t_list):
        exp_term = np.exp(-t * laplacian) * scalar_value
        heat_kernel[:, :, idx] = exp_term

    return heat_kernel