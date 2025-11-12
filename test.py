# save as optimize_lfs_filter.py
import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import eigh
import gurobipy as gp
from gurobipy import GRB

def compute_laplacian_eigenvalues(A, k=None):
    """
    A : adjacency matrix (numpy array)
    k : number of eigenvalues to return (None => all)
    returns sorted eigenvalues (ascending)
    """
    # graph Laplacian L = D - A
    L = csgraph.laplacian(A, normed=False)
    if k is None or k >= L.shape[0]:
        vals, _ = eigh(L)
    else:
        # compute smallest k eigenvalues if desired (scipy sparse eigen solvers could be used for big graphs)
        vals, _ = eigh(L)  # for simplicity; replace with sparse solvers for large graphs
        vals = vals[:k]
    return np.sort(vals)

def build_shift_index_map(eigvals, t_grid):
    """
    For each eigenvalue lambda_j and each t_i, compute the index k such that
    h_shift_j[i] = h[k] approximates h(|t_i - lambda_j|)
    If the computed index is outside [0, N], we treat it as "no contribution" (i.e., we won't add constraints for it).
    Returns a list of lists: shifts[j] is array of length len(t_grid) containing target indices (ints) or -1 if out-of-range.
    """
    dt = t_grid[1] - t_grid[0]
    N = len(t_grid) - 1
    shifts = []
    for lam in eigvals:
        idxs = np.full(len(t_grid), -1, dtype=int)
        for i, t in enumerate(t_grid):
            val = abs(t - lam)
            k = int(round(val / dt))
            if 0 <= k <= N:
                idxs[i] = k
            else:
                idxs[i] = -1
        shifts.append(idxs)
    return shifts

def optimize_filter(A,
                    T=None,
                    N=100,
                    mu=None,
                    ch=10.0,
                    use_top_eigs=None,
                    verbose=1):
    """
    Solve the discretized convex optimization for h as in Hu et al. (2018).
    Inputs:
      - A : adjacency matrix (n x n)
      - T : maximum t to sample on [0, T]. If None, set to max_eig + margin.
      - N : number of intervals (so samples = N+1)
      - mu : the parameter mu = 2/delta (if None -> computed by using average eigen-gap)
      - ch : curvature bound c_h (|h''| <= ch and h'' <= 0)
      - use_top_eigs : if int, use only first use_top_eigs eigenvalues; else all
      - verbose : Gurobi output level
    Returns:
      h (numpy array length N+1), metadata dict
    """
    n = A.shape[0]
    eigvals = compute_laplacian_eigenvalues(A, k=use_top_eigs)
    # grid
    if T is None:
        Tmax = float(np.max(eigvals)) + 0.5 * (np.max(eigvals) - np.min(eigvals) + 1e-6)
    else:
        Tmax = T
    t_grid = np.linspace(0.0, Tmax, N+1)
    dt = t_grid[1] - t_grid[0]

    # compute eigen-gap delta (average gap)
    gaps = np.diff(eigvals)
    avg_gap = float(np.mean(gaps)) if len(gaps) > 0 else 1.0
    if mu is None:
        mu = 2.0 / (avg_gap if avg_gap > 1e-8 else 1e-8)

    # build index map for shifts
    shifts = build_shift_index_map(eigvals, t_grid)
    num_eigs = len(eigvals)

    # Start Gurobi model
    m = gp.Model("adaLFS_filter")
    m.setParam('OutputFlag', int(bool(verbose)))

    # Variables: h_0..h_N (continuous)
    h = m.addVars(N+1, lb=0.0, ub=GRB.INFINITY, name="h")

    # boundary conditions:
    m.addConstr(h[0] == 1.0, name="h0_eq_1")
    m.addConstr(h[N] == 0.0, name="hN_eq_0")

    # finite-difference variables (not strictly necessary but convenient)
    # hprime_i for i=0..N-1 representing derivative approx
    hprime = m.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="hprime")
    for i in range(N):
        # h'_i = (h_{i+1} - h_i)/dt
        m.addConstr(hprime[i] == (h[i+1] - h[i]) / dt, name=f"hprime_def_{i}")
        # monotonic nonincreasing: h'_i <= 0
        m.addConstr(hprime[i] <= 0.0, name=f"hprime_le_0_{i}")

    # hsec_i for i=0..N-2
    hsec = m.addVars(max(0, N-1), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="hsec")
    for i in range(max(0, N-1)):
        # h''_i = (h_{i+2} - 2 h_{i+1} + h_i) / dt^2
        m.addConstr(hsec[i] == (h[i+2] - 2.0*h[i+1] + h[i]) / (dt*dt), name=f"hsec_def_{i}")
        # bound: -ch <= hsec <= 0
        m.addConstr(hsec[i] <= 0.0, name=f"hsec_le_0_{i}")
        m.addConstr(hsec[i] >= -ch, name=f"hsec_ge_negch_{i}")

    # For each sample index i, introduce aux var m_i >= h_shift_j[i] for all j
    mvars = m.addVars(N+1, lb=0.0, ub=GRB.INFINITY, name="mmax_hshift")
    # For derivatives, we need shifted h' as well. Build mapping for h'shift indices: because h' defined on 0..N-1 t positions (we will use same t_i grid,
    # but h' shifts are defined for i=0..N-1). We'll create mprime vars of length N.
    mprime_vars = m.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="mmax_hpshift")

    # Add constraints: for each eigenvalue j and each t_i, if shift maps to index k then:
    # m_i >= h[k]
    for j in range(num_eigs):
        idxs = shifts[j]
        for i, k in enumerate(idxs):
            if k >= 0:
                m.addConstr(mvars[i] >= h[k], name=f"m_ge_h_j{j}_i{i}_k{k}")

    # For shifted derivatives: we need h' sampled at the same grid positions 0..N-1.
    # For each eigenvalue j and each derivative grid index i (i = 0..N-1), we map |t_i - lambda| to derivative index k' in [0..N-1]
    # We'll compute on-the-fly:
    for j, lam in enumerate(eigvals):
        for i in range(N):
            t = t_grid[i]
            val = abs(t - lam)
            kp = int(round(val / dt))
            if 0 <= kp <= N-1:
                m.addConstr(mprime_vars[i] >= hprime[kp], name=f"mp_ge_hprime_j{j}_i{i}_k{kp}")

    # Objective: mu * sum_i mvars[i] + sum_i mprime_vars[i]
    objective = mu * gp.quicksum(mvars[i] for i in range(N+1)) + gp.quicksum(mprime_vars[i] for i in range(N))
    m.setObjective(objective, GRB.MINIMIZE)

    # Optimize
    m.optimize()

    # Extract solution
    h_sol = np.array([h[i].X for i in range(N+1)])
    metadata = {
        't_grid': t_grid,
        'dt': dt,
        'eigvals': eigvals,
        'mu': mu,
        'ch': ch,
        'gurobi_status': m.Status
    }
    return h_sol, metadata

from graph import Graph, create_compability_matrix, compute_matching_grb, interpret_matching

if __name__ == "__main__":
    # small example
    # np.random.seed(1)
    n = 100
    # random symmetric weighted adjacency
    M = np.random.rand(n, n)
    A = (M + M.T) / 2.0
    # zero diagonal
    np.fill_diagonal(A, 0.0)

    ivw_file1 = "IvwData/Case1/Timestep1.txt"
    ivw_file2 = "IvwData/Case1/Timestep2.txt"

    use_edge_length = True
    graph1 = Graph()

    t_hks = np.logspace(1, 15, 100)
    is_normalized_Laplacian = True
    is_normalize_hks = False
    is_normalize_vectors = False

    use_physical_costraint = True
    # Read file

    graph1.read_ivw_info(ivw_file2)
    graph1.build_adj_mat(use_edge_length)

    h, meta = optimize_filter(A, T=50, N=80, mu=None, ch=20.0, verbose=1)
    print("t_grid[0:5]:", meta['mu'])
    print("first 10 samples of h:", h[:])