import numpy as np
from src.orcml import *
from src.utils.graph_utils import *
from src.plotting import *
import sklearn.metrics as metrics
from tqdm import tqdm
import torch


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

def isorc_embedding(G_ann, A, n_iter=1000, lr=0.1, dist_scale=1.0, frames=10):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reversed_indices = np.array(np.argsort(G_ann.nodes))
    # get initial positions
    X = np.array([G_ann.nodes[i]['pos'] for i in G_ann.nodes])[reversed_indices]
    X = torch.tensor(X).to(device).requires_grad_(True)
    edge_mask = np.where(A > 0, 1, 0)
    edge_mask = torch.tensor(edge_mask).to(device)

    pdist = torch.cdist(X, X, p=2)
    masked_pdist = pdist * edge_mask
    sc_ind, d_G_prime = isorc_summary(G_ann)
    E = torch.tensor(equilibrium_matrix(sc_ind, d_G_prime, A, dist_scale)). to(device)
    # optimize X so that pairwise distances are close to the equilibrium distances
    optimizer = torch.optim.Adam([X], lr=lr)

    step_per_frame = n_iter // frames
    X_frames = [X.clone().detach().cpu().numpy()]
    with tqdm(total=n_iter) as pbar:
        for i in range(n_iter):
            optimizer.zero_grad()
            pdist = torch.cdist(X, X, p=2)
            fro_norm = torch.norm(pdist)
            masked_pdist = pdist * edge_mask
            loss = torch.sum((masked_pdist - E)**2) - 0.1 * fro_norm
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})
            if i % step_per_frame == 0:
                X_frames.append(X.clone().detach().cpu().numpy())
    return X.detach().cpu().numpy(), X_frames