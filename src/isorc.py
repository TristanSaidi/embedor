import networkx as nx
import numpy as np
import numpy as np
from src.orcml import *
from src.graph_utils import *

def isorc_annotate(G, lda=0.01, delta=0.8, verbose=False):
    """
    Annotate the graph with the orcml method for isorc embedding.
    Parameters
    ----------
    G : networkx.Graph
        The graph to prune.
    eps : float
        The epsilon parameter for the proximity graph.
    lda : float
        The lambda parameter for pruning (see paper).
    delta : float, optional
        The delta (confidence) parameter for pruning (see paper).
    Returns
    -------
    G_ann : networkx.Graph
        The annotated graph.
    """    

    # compute ORC
    G = compute_orc(G)["G"]
    # construct the candidate set C, and filtered graph G'
    C = []
    G_prime = nx.Graph()
    threshold = -1 + 2*(2-2*delta)
    candidate_edge_indices = []
    for idx, (i, j, d) in enumerate(G.edges(data=True)):
        # add entry for shortcut flag and G_prime distance
        d['shortcut'] = 0
        d['G_prime_dist'] = d['weight'] # initialize to euclidean distance

        if d['ricciCurvature'] < threshold:
            C.append((i,j))
            candidate_edge_indices.append(idx)
        else:
            G_prime.add_edge(i, j, weight=d["weight"])
            G_prime[i][j]['ricciCurvature'] = d['ricciCurvature']
            G_prime[i][j]['effective_eps'] = d['effective_eps']
            G_prime[i][j]['G_prime_dist'] = d['G_prime_dist']
    if verbose:
        print(f"Number of candidate edges: {len(C)}, Number of edges in G': {len(G.edges())}")

    G_ann = G.copy()
    non_shortcut_edges = list(range(len(G.edges()))) # start from all edges and remove as we go
    shortcut_edges = []
    for num, (i, j) in enumerate(C):
        # get epsilon for the edge
        effective_eps = G[i][j]['effective_eps']

        # check distance d_G'(x,y) for all x,y in C
        threshold = ((1-lda)*np.pi**2)/(2*np.sqrt(24*lda)) * effective_eps

        if i not in G_prime.nodes() or j not in G_prime.nodes():
            continue
        try:
            d_G_prime = nx.shortest_path_length(G_prime, source=i, target=j, weight="weight") # use euclidean distance (weight)
        except nx.NetworkXNoPath:
            d_G_prime = np.inf

        # adjust G_ann
        G_ann[i][j]['G_prime_dist'] = d_G_prime
        if d_G_prime > threshold:
            # adjust edge lists
            non_shortcut_edges.remove(candidate_edge_indices[num])
            shortcut_edges.append(candidate_edge_indices[num])
            # adjust G_ann
            G_ann[i][j]['shortcut'] = 1
            if verbose:
                shortcut_str = f"Shortcut Edge Detected: edge {num}\n d_G'(x,y)/effective_eps: {d_G_prime/effective_eps}\n Threshold/effective_eps: {threshold/effective_eps}\n\n"
                print(shortcut_str)
    
    return_dict = {
        "G_ann": G_ann,
        "non_shortcut_edges": non_shortcut_edges,
        "shortcut_edges": shortcut_edges
    }
    return return_dict

def isorc_summary(G):
    """
    Summarize the isorc annotations of the graph.
    Parameters
    ----------
    G : networkx.Graph
        The graph to summarize. Should be annotated with isorc.
    Returns
    -------
    sc_ind : np.ndarray
        The shortcut indicator matrix.
    D_G_prime : np.ndarray
        The G_prime distance matrix.
    """
    # create two |V| x |V| matrices
    n = len(G.nodes)
    sc_ind = np.zeros((n, n))
    D_G_prime = np.zeros((n, n))

    # iterate through edges of G
    for u, v in G.edges:
        # get the isorc annotation of the edge: shortcut
        sc = G[u][v]['shortcut']
        sc_ind[u, v] = sc
        sc_ind[v, u] = sc
        # get the isorc annotation of the edge: distance
        d = G[u][v]['G_prime_dist']
        if d == np.inf:
            d = 10 * G[u][v]['weight']
        D_G_prime[u, v] = d
        D_G_prime[v, u] = d
    return sc_ind, D_G_prime

def equilibrium_matrix(sc_ind, D_G_prime, A, dist_scale=1.5):
    """
    Compute the spring equilibrium matrix for the isorc embedding.
    Parameters
    ----------
    sc_ind : np.ndarray
        The shortcut indicator matrix.
    D_G_prime : np.ndarray
        The G_prime distance matrix.
    A : np.ndarray
        The adjacency matrix of the graph.
    dist_scale : float, optional
        The distance scaling factor.
    Returns
    -------
    E : np.ndarray
        The equilibrium matrix.
    """
    # create the equilibrium matrix as element-wise maximum of A and sc_ind * D_G_prime
    E = np.maximum(A, sc_ind * D_G_prime * dist_scale)
    return E

def compute_forces_parallel(E, G, A, X, sc_ind, k=50, k_sc=0.1):
    """ 
    Compute the forces on the nodes of the graph using a vectorized implementation.
    Parameters
    ----------
    E : np.ndarray
        The equilibrium matrix.
    G : networkx.Graph
        The graph.
    A : np.ndarray
        The adjacency matrix.
    X : np.ndarray
        The node positions.
    sc_ind : np.ndarray
        The shortcut indicator matrix.
    k : float, optional
        The spring constant for good edges.
    k_sc : float, optional
        The spring constant for shortcuts.
    Returns
    -------
    F : np.ndarray
        The forces on the nodes.
    total_energy : float
        The total potential energy of the system
    """
    # X.T is D x N
    # Broadcast X.T to N x D x N
    X_1 = np.array([X.T for _ in range(len(G.nodes))])
    # create batched D
    D_batched = X_1 - X_1.transpose(2, 1, 0) # N x D x N
    # E is N x N, A is N x N. Take difference to create displacement matrix
    D_eq = A - E # N x N
    # create mask for D_eq
    mask = np.where(A > 0, 1, 0) # N x N
    # combine mask and D_eq with element-wise multiplication
    D_eq_masked = D_eq * mask # N x N
    # create spring constant matrix
    # k for good edges, k_sc for shortcuts
    # sc_ind is N x N. 
    K = np.where(sc_ind == 1, k_sc, k) # N x N
    # combine D_eq_masked and K with element-wise multiplication
    D_eq_masked_K = D_eq_masked * K # N x N
    # compute forces with np.matmul to get N x D x 1
    F = np.matmul(D_batched, D_eq_masked_K[:, :, np.newaxis]) # N x D x 1
    # flatten F to get N x D
    F = np.squeeze(F, axis=2) # N x D
    # compute total energy
    total_energy = 0.5 * np.sum(D_eq_masked * D_eq_masked * K)
    return F, total_energy

def compute_forces_local(E, G, A, X, i, sc_ind, k, k_sc):
    """ 
    Compute the forces on a single node.
    Parameters
    ----------
    E : np.ndarray
        The equilibrium matrix.
    G : networkx.Graph
        The graph.
    A : np.ndarray
        The adjacency matrix.
    X : np.ndarray
        The node positions.
    i : int
        The node index to compute forces for.
    sc_ind : np.ndarray
        The shortcut indicator matrix.
    k : float, optional
        The spring constant for good edges.
    k_sc : float, optional
        The spring constant for shortcuts.
    Returns
    -------
    F : np.ndarray
        The forces on the nodes.
    total_energy : float
        The total potential energy of the system
    """
    # compute spring constant vector. k for good edges, k_sc for shortcuts
    k_i = np.where(sc_ind[i] == 1, k_sc, k)
    # get adjacency of node i
    A_i = A[i]
    # get indicator of adjacency of node i
    ind_i = np.where(A_i > 0, 1, 0)
    # get the equilibrium vector of the neighborhood of node i
    E_i = E[i]
    # get displacement vector from equilibrium of node i to others
    D_eq_i = A_i - E_i
    # get the pos of node i
    v_i = X[i]
    # broadcast node i to the size of the |V|. Transpose to get a column vector
    V_i = np.tile(v_i, (len(G.nodes), 1)).T
    # compute displacement vector from node i to others
    D_i = X.T - V_i
    # compute energies
    en_i = 0.5 * np.sum(D_eq_i * D_eq_i * k_i)
    # element-wise product of D_eq_i and ind_i
    D_eq_i_masked = D_eq_i * ind_i
    # compute the forces acting on node i
    F_i = D_i @ (D_eq_i_masked * k_i)
    return F_i, en_i

def compute_forces(E, G, A, X, sc_ind, k=50, k_sc=0.1):
    """ 
    Compute the forces on the nodes of the graph using a serial implementation.
    Parameters
    ----------
    E : np.ndarray
        The equilibrium matrix.
    G : networkx.Graph
        The graph.
    A : np.ndarray
        The adjacency matrix.
    X : np.ndarray
        The node positions.
    sc_ind : np.ndarray
        The shortcut indicator matrix.
    k : float, optional
        The spring constant for good edges.
    k_sc : float, optional
        The spring constant for shortcuts.
    Returns
    -------
    F : np.ndarray
        The forces on the nodes.
    total_energy : float
        The total potential energy of the system
    """
    forces = np.zeros((len(G.nodes), X.shape[1]))
    # for each node, compute the force
    total_energy = 0
    for i in G.nodes:
        forces[i, :], en_i = compute_forces_local(E, G, A, X, i, sc_ind, k=k, k_sc=k_sc)
        total_energy += en_i
    return forces, total_energy

def step_physics(F, G, dt, mass, damping=0.1):
    """ 
    Perform physics step.
    
    Parameters
    ----------
    F : np.ndarray
        The forces acting on the nodes of G.
    G : nx.Graph
        The input graph.
    dt : float
        The time step.
    mass : float
        The mass of the nodes.
    damping : float, optional
        The damping factor.
    Returns
    -------
    np.ndarray
        The updated graph G.
    """
    # for each node, update the position
    for i in G.nodes:
        damping_force = - damping * G.nodes[i]['vel']
        total_force = F[i] + damping_force
        a = total_force / mass
        G.nodes[i]['vel'] = G.nodes[i]['vel'] + a * dt
        G.nodes[i]['pos'] = G.nodes[i]['pos'] + G.nodes[i]['vel'] * dt
    return G

def update_adjacency_matrix(G, X):
    """ 
    Update the adjacency matrix of a graph G.
    
    Parameters
    ----------
    G : nx.Graph
        The input graph.
    X : np.ndarray
        The positions of the nodes of
        G in the embedding space.
    Returns
    -------
    np.ndarray
        The updated adjacency matrix of G.
    """
    n = len(G.nodes)
    A_new = np.zeros((n, n))
    for i, j in G.edges:
        dist = np.linalg.norm(X[i] - X[j])
        A_new[i, j] = dist
        A_new[j, i] = dist
    return A_new