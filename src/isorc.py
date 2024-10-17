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

def equilibrium_matrix(sc_ind, D_G_prime, A, dist_scale=1.0, repulsion=False):
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
    if repulsion:
        # flip sign of shortcut edges
        E = np.where(sc_ind == 1, -1 * E, E)
    return E

def compute_forces_parallel(
        E, 
        G, 
        A, 
        X, 
        sc_ind, 
        k=5, 
        k_sc=0.1, 
        encourge_spread=False, 
        D_pw=None, 
        spread_force_scale=0.01,
        max_force_mag=1,
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
    # clip D_eq_masked to be within 2 standard deviations of the mean
    D_eq_masked = np.clip(D_eq_masked, -100 * np.std(D_eq_masked) + np.mean(D_eq_masked), 100 * np.std(D_eq_masked) + np.mean(D_eq_masked))
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
        F_spread_sum = np.sum(F_spread, axis=0) * spread_force_scale
        F += F_spread_sum
    return F, pe

def compute_damping_vectorized(F, v, damping_factor):
    """ 
    Compute the damping force.
    
    Parameters
    ----------
    F : np.ndarray
        The force acting on all nodes.
    v : np.ndarray
        The velocity of all nodes.
    damping_factor : float
        The damping factor.
    Returns
    -------
    np.ndarray
        The damping force.
    """
    n = F.shape[0]
    d = F.shape[1]
    v_norm = np.linalg.norm(v, axis=1)
    assert v_norm.shape == (n,)
    # if velocity is close to zero for any node, set norm to 1
    v_norm = np.where(v_norm < 1e-6, 1, v_norm)
    # if velocity is close to zero for any node, set damping force direction to zero
    damping_force_dir = -v / v_norm[:, np.newaxis]
    assert damping_force_dir.shape == (n, d)
    # F norm
    F_norm = np.linalg.norm(F, axis=1)
    assert F_norm.shape == (n,)
    # damping force magnitude
    damping_force_mag = damping_factor * F_norm # N x 1
    damping_force = damping_force_dir * damping_force_mag[:, np.newaxis]
    assert damping_force.shape == (n, d)
    return damping_force

def compute_damping(F_i, v_i, damping_factor):
    """ 
    Compute the damping force.
    
    Parameters
    ----------
    F_i : np.ndarray
        The force acting on the node.
    v_i : np.ndarray
        The velocity of the node.
    damping_factor : float
        The damping factor.
    Returns
    -------
    np.ndarray
        The damping force.
    """
    v_i_norm = np.linalg.norm(v_i)
    # if velocity is close to zero, set norm to 1
    v_i_norm = 1 if v_i_norm < 1e-6 else v_i_norm
    # if velocity is close to zero, set damping force direction to zero
    damping_force_dir = -v_i / v_i_norm
    # F norm
    F_norm = np.linalg.norm(F_i)
    # damping force magnitude
    damping_force_mag = damping_factor * F_norm
    damping_force = damping_force_dir * damping_force_mag
    return damping_force

class AdaGrad(object):
    def __init__(self, n, d, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.n = n # number of nodes
        self.d = d # dimension of the node positions
        self.grad_sq_sum = np.zeros((n, d))

    def update(self, grad):
        self.grad_sq_sum += grad**2
        return grad / (np.sqrt(self.grad_sq_sum) + self.eps)
    
class adaptive_timestep(object):
    def __init__(self, dt, adaptive=False, ema_alpha=0.9):
        self.dt = dt
        self.adaptive = adaptive
        self.ema_alpha = ema_alpha
        self.F_mag_max = 0

    def update(self, F):
        if not self.adaptive:
            return self.dt
        total_force_mag = np.sum(np.linalg.norm(F, axis=1))
        self.F_mag_max = (1-self.ema_alpha)*max(self.F_mag_max, total_force_mag) + self.ema_alpha*self.F_mag_max
        ratio = np.sqrt(self.F_mag_max / total_force_mag)
        dt_prime = self.dt * ratio
        self.dt = (1-self.ema_alpha) * dt_prime + self.ema_alpha * self.dt
        return self.dt



def step_physics(F, G, dt, mass, optimizer, damping=0.5):
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
    optimizer : AdaGrad
        The optimizer object.
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
        damping_force = compute_damping(F[i], G.nodes[i]['vel'], damping)
        total_force = F[i] + damping_force
        # total_force = optimizer.update(total_force)
        a = total_force / mass
        G.nodes[i]['vel'] = G.nodes[i]['vel'] + a * dt
        G.nodes[i]['pos'] = G.nodes[i]['pos'] + G.nodes[i]['vel'] * dt
    return G, ke

def step_physics_vectorized(
        F, 
        G, 
        dt, 
        mass, 
        optimizer, 
        damping=0.5, 
        max_force_mag=0.1
    ):
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
    # extract node positions and velocities
    pos = np.array([G.nodes[i]['pos'] for i in range(len(G.nodes))])
    vel = np.array([G.nodes[i]['vel'] for i in range(len(G.nodes))])

    # clip forces such that the magnitude of the force is at most max_force_mag
    if max_force_mag is not None:
        F_norm = np.linalg.norm(F, axis=1)
        F = np.where((F_norm > max_force_mag)[: , np.newaxis], F / F_norm[:, np.newaxis] * max_force_mag , F)

    # compute damping forces
    damping_force = compute_damping_vectorized(F, vel, damping)
    # compute total forces
    total_force = F + damping_force
    # compute update
    # total_force = optimizer.update(total_force)
    # compute acceleration
    a = total_force / mass
    # update velocity
    # vel = vel + a * dt
    # update position
    pos = pos + a * dt
    # compute kinetic energy
    ke = 0.5 * mass * np.sum(np.linalg.norm(vel, axis=1)**2)
    # update graph
    for i in G.nodes:
        G.nodes[i]['pos'] = pos[i]
        G.nodes[i]['vel'] = vel[i]
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

def compute_dt(F_ema, dt_init, F_curr, adaptive):
    """ 
    Compute the adaptive time step size.
    
    Parameters
    ----------
    F_ema : np.ndarray
        The ema of forces.
    dt_init : float
        The initial time step size.
    F_curr : np.ndarray
        The current forces.
    Returns
    -------
    dt : float
        The updated time step size.
    """
    if not adaptive:
        return dt_init
    # compute the ratio of the norm of the forces
    ratio = np.sqrt(np.linalg.norm(F_curr) / np.linalg.norm(F_ema))
    # compute the updated time step size
    dt = dt_init / ratio
    return dt

def simulate(
        G_ann, 
        A_orig, 
        dt, 
        mass=1, 
        n_steps=10, 
        n_frames=10, 
        return_frames=True,
        adaptive_dt=False,
        ema_alpha=0.9
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
    frames : dict
        The dict of frames.
    """
    # initialize optimizer
    optimizer = AdaGrad(n=len(G_ann.nodes), d=G_ann.nodes[0]['pos'].shape[0])
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
        Xs = [X]
        Gs = [G_ann]
    else:
        Xs = None
        Gs = None
    render_freq = n_steps // n_frames
    
    F, pe = compute_forces_parallel(
        E, 
        G_ann, 
        A, 
        X, 
        sc_ind, 
        D_pw=D_pw
    ) # compute forces and system energy     
    timestepper = adaptive_timestep(dt, adaptive=adaptive_dt, ema_alpha=ema_alpha)
    # simulate the physics
    with tqdm(total=n_steps) as pbar:
        for step in range(n_steps):
            F, pe = compute_forces_parallel(
                E, 
                G_ann, 
                A, 
                X, 
                sc_ind, 
                D_pw=D_pw
            ) # compute forces and system energy
            
            dt = timestepper.update(F)
            G_ann, ke = step_physics_vectorized(
                F, 
                G_ann, 
                dt, 
                optimizer=optimizer, 
                mass=mass,
            )

            X = np.array([G_ann.nodes[i]['pos'] for i in G_ann.nodes])[reversed_indices]
            A, D_pw = update_adjacency_matrix(X, edge_mask)
            # plot and update fig_dict
            if step % render_freq == 0 and return_frames:
                Xs.append(X)
                Gs.append(G_ann)
            pbar.update(1)
            # display energy on progress bar (not scientific notation)
            en = pe + ke # total energy
            pbar.set_postfix_str(f"Total Energy: {en:.2f}, Potential Energy: {pe:.2f}, Kinetic Energy: {ke:.2f}, dt: {dt:.3f}")
    frames = {
        "Xs": Xs,
        "Gs": Gs
    }
    return G_ann, frames