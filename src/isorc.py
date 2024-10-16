import networkx as nx
import numpy as np
import numpy as np
from src.orcml import *
from src.graph_utils import *
from src.plotting import *
import sklearn.metrics as metrics
from tqdm import tqdm
import warnings

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

def equilibrium_matrix(sc_ind, D_G_prime, A, dist_scale=1.0):
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

def compute_forces_parallel(
        E, 
        G, 
        A, 
        X, 
        sc_ind, 
        k=5, 
        k_sc=0.1, 
        encourge_spread=True, 
        D_pw=None, 
        spread_force_scale=0.1
    ):
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
    pe : float
        The potential energy of the system
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
    # compute potential energy
    pe = 0.5 * np.sum(D_eq_masked * D_eq_masked * K)

    # if we want to encourage points to be spread out
    if encourge_spread:
        # small repulsion force to encourage spread

        # compute pairwise distances
        if D_pw is None:
            D_pw = metrics.pairwise_distances(X)

        D_batched_norm = np.linalg.norm(D_batched, axis=1)
        # fill diagonal with 1 to avoid division by zero
        np.fill_diagonal(D_batched_norm, 1)
        D_batched_unit = D_batched / D_batched_norm[:, :, np.newaxis].transpose(0, 2, 1)

        D_pw_sqrd = D_pw**2
        # fill diagonal with 1 to avoid division by zero
        np.fill_diagonal(D_pw_sqrd, 1)
        inv_D_pw_clamped_sqrd = 1 / D_pw_sqrd
        # compute the spread forces. Start with 1/D_pw^2 (capping )        
        F_spread = -1 * D_batched_unit.transpose(0, 2, 1) * inv_D_pw_clamped_sqrd[:, :, np.newaxis]
        # compute the sum of the spread forces
        F_spread_sum = np.sum(F_spread, axis=0)
        # normalize the spread forces
        F_spread_sum = F_spread_sum / np.linalg.norm(F_spread_sum, axis=1)[:, np.newaxis]
        # scale the spread forces by mean spring force
        F_spread_sum = spread_force_scale * F_spread_sum * np.mean(np.linalg.norm(F, axis=1))
        F += F_spread_sum
    return F, pe

def step_physics(F, G, dt, mass, damping=0.5):
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
    # compute kinetic energy
    ke = 0.0
    # for each node, update the position
    for i in G.nodes:
        ke += 0.5 * mass * np.linalg.norm(G.nodes[i]['vel'])**2
        if np.linalg.norm(G.nodes[i]['vel']) < 1e-6:
            damping_force_dir = np.zeros(3)
        else:
            damping_force_dir = -G.nodes[i]['vel'] / np.linalg.norm(G.nodes[i]['vel']) # unit vector
        damping_force_mag = damping * np.linalg.norm(F[i])
        total_force = F[i] + damping_force_mag * damping_force_dir
        a = total_force / mass
        G.nodes[i]['vel'] = G.nodes[i]['vel'] + a * dt
        G.nodes[i]['pos'] = G.nodes[i]['pos'] + G.nodes[i]['vel'] * dt
    return G, ke

def update_adjacency_matrix(X, edge_mask):
    """ 
    Update the adjacency matrix of a graph G.
    
    Parameters
    ----------
    X : np.ndarray
        The node positions.
    edge_mask : np.ndarray
        The edge mask.
    Returns
    -------
    A_new : np.ndarray
        The updated adjacency matrix of G.
    D : np.ndarray
        The pairwise distance matrix (for caching).
    """
    # compute pairwise distances
    D = metrics.pairwise_distances(X)
    # create adjacency matrix
    A_new = D * edge_mask
    return A_new, D


def simulate(
        G_ann, 
        A_orig, 
        dt, 
        mass=1, 
        n_steps=10, 
        n_frames=10, 
        return_frames=True
    ):
    """ 
    Simulate the physics of a graph.
    
    Parameters
    ----------
    G_ann : nx.Graph
        The input graph. The graph should have isorc annotation already.
    A_orig : np.ndarray
        The adjacency matrix of G_ann.
    dt : float
        The time step size.
    mass : float
        The mass of the nodes.
    n_steps : int
        The number of steps to simulate.
    n_frames : int
        The number of frames to return.
    return_frames : bool
        Whether to return frames.
    Returns
    -------
    G_ann : nx.Graph
        The updated graph.
    figs : list
        The list of frames.
    """
    edge_mask = np.where(A_orig > 0, 1, 0) # get edge mask
    # get index map
    reversed_indices = np.array(np.argsort(G_ann.nodes))
    # get initial positions
    X = np.array([G_ann.nodes[i]['pos'] for i in G_ann.nodes])[reversed_indices]
    # get initial adjacency matrix
    A, D_pw = update_adjacency_matrix(X, edge_mask)
    assert np.allclose(A, A_orig)
    # annotate the graph with isorc
    sc_ind, D_G_prime = isorc_summary(G_ann)
    # compute the system equilibrium matrix
    E = equilibrium_matrix(sc_ind, D_G_prime, A_orig)
    if return_frames:
        figs = []
        figs.append(plot_graph_3D(X, G_ann, title=None))
    else:
        figs = None
    render_freq = n_steps // n_frames
    # simulate the physics
    with tqdm(total=n_steps) as pbar:
        for step in range(n_steps):
            F, pe = compute_forces_parallel(E, G_ann, A, X, sc_ind, D_pw=D_pw) # compute forces and system energy        
            G_ann, ke = step_physics(F, G_ann, dt, mass=mass)
            X = np.array([G_ann.nodes[i]['pos'] for i in G_ann.nodes])[reversed_indices]
            A, D_pw = update_adjacency_matrix(X, edge_mask)
            # plot and update fig_dict
            if step % render_freq == 0 and return_frames:
                figs.append(plot_graph_3D(X, G_ann, title=None))
            pbar.update(1)
            # display energy on progress bar (not scientific notation)
            en = pe + ke # total energy
            pbar.set_postfix_str(f"Total Energy: {en:.2f}, Potential Energy: {pe:.2f}, Kinetic Energy: {ke:.2f}")
    return G_ann, figs