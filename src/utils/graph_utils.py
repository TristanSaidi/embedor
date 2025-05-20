import networkx as nx
import numpy as np
from sklearn import neighbors
from src.ollivier_ricci import OllivierRicci
import pynndescent
import time
import multiprocessing as mp
import scipy

_A = None

def compute_orc(G, nbrhood_size=1):
    """
    Compute the Ollivier-Ricci curvature on edges of a graph.
    Parameters
    ----------
    G : networkx.Graph
        The graph.
    nbrhood_size : int, optional
        Number of hops to consider for neighborhood.
    Returns
    -------
    G : networkx.Graph
        The graph with the Ollivier-Ricci curvatures as edge attributes.
    """
    orc = OllivierRicci(G, weight="unweighted", alpha=0.0, method='OTD', verbose='INFO', nbrhood_size=nbrhood_size)
    orc.compute_ricci_curvature()
    orcs = []
    for i, j, _ in orc.G.edges(data=True):
        orcs.append(orc.G[i][j]['ricciCurvature'])
    return {
        'G': orc.G,
        'orcs': orcs,
    }

def compute_beckmann_orc(G):
    """
    Compute the ORC using the Beckmann-2 distance.
    Parameters
    ----------
    G : networkx.Graph
        The graph.
    Returns
    -------
    G : networkx.Graph
        The graph with the Beckmann-2 distances as edge attributes.
    """
    A = nx.to_numpy_array(G, weight='weight', nodelist=list(range(len(G.nodes()))))
    A[A > 0] = 1
    # compute the Laplacian matrix
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    L = D - A
    # compute the pseudo-inverse of the Laplacian matrix
    time_start = time.time()
    L_pinv = scipy.linalg.pinv(L)
    time_end = time.time()
    print(f"Time taken to compute pseudo-inverse of Laplacian: {time_end - time_start} seconds")
    # get the edges of the graph
    edges = np.array(list(G.edges()))
    beckmann_orcs = _compute_beckmann_orc(edges, L_pinv, A)
    return {
        'G': G,
        'orcs': beckmann_orcs,
    }

def _distribute_density(edge):
    global _A
    x, y = edge
    # get 1-hop neighbors of x and y
    x_neighbors = np.where(_A[x, :] > 0)[0]
    y_neighbors = np.where(_A[y, :] > 0)[0]
    # exclude x and y from the neighbors
    x_neighbors = np.delete(x_neighbors, np.where(x_neighbors == x))
    y_neighbors = np.delete(y_neighbors, np.where(y_neighbors == y))
    # construct uniform distribution over neighbors
    assert len(x_neighbors) > 0, "x has no neighbors"
    assert len(y_neighbors) > 0, "y has no neighbors"
    mass_x = 1 / len(x_neighbors)
    mass_y = 1 / len(y_neighbors)
    mu = np.zeros(_A.shape[0])
    nu = np.zeros(_A.shape[0])
    mu[x_neighbors] = mass_x
    nu[y_neighbors] = mass_y
    return mu, nu

def distribute_densities(edges, A):
    """ 
    Distribute the densities of the edges over the 1-hop neighbors of the nodes
    in the edge. 
    """
    # use multiprocessing to speed up the for loop above
    global _A
    _A = A
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(_distribute_density, edges)
    mus = np.stack([result[0] for result in results], axis=1)
    nus = np.stack([result[1] for result in results], axis=1)
    return mus, nus

def _compute_beckmann_orc(edges, L_pinv, A):
    """ 
    Compute ORC using the Beckmann-2 distance instead of the Wasserstein-1 distance.
    """
    time_start = time.time()
    mus, nus = distribute_densities(edges, A)
    time_end = time.time()
    print(f"Time taken to distribute densities: {time_end - time_start} seconds")
    diff = mus - nus
    beckman_dists = np.sum(np.multiply(diff.T @ L_pinv, diff.T), axis=1)
    beckman_dists = beckman_dists ** 0.5
    beckmann_orcs = 1 - beckman_dists
    return beckmann_orcs


def get_nn_graph(data, exp_params):
    """ 
    Build the nearest neighbor graph.
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        The dataset.
    exp_params : dict
        The experimental parameters.
    Returns
    -------
    return_dict : dict
    """
    if exp_params['mode'] == 'nbrs':
        G, A = _get_nn_graph(data, mode=exp_params['mode'], n_neighbors=exp_params['n_neighbors']) # unpruned k-nn graph
    elif exp_params['mode'] == 'eps':
        G, A = _get_nn_graph(data, mode=exp_params['mode'], epsilon=exp_params['epsilon'])
    elif exp_params['mode'] == 'descent':
        G, A = _get_nn_graph(data, mode=exp_params['mode'], n_neighbors=exp_params['n_neighbors'])
    return {
        "G": G,
        "A": A,
    }


def _get_nn_graph(X, mode='nbrs', n_neighbors=None, epsilon=None):
    """
    Create a proximity graph from a dataset.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The dataset.
    mode : str, optional
        The mode of the graph construction. Either 'nbrs' or 'eps' or 'descent'.
    n_neighbors : int, optional
        The number of neighbors to consider when mode='nbrs'.
    epsilon : float, optional
        The epsilon parameter when mode='eps'.
    Returns
    -------
    G : networkx.Graph
        The proximity graph.
    """
    
    if mode == 'nbrs':
        assert n_neighbors is not None, "n_neighbors must be specified when mode='nbrs'."
        A = neighbors.kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance')
    elif mode == 'eps':
        assert epsilon is not None, "epsilon must be specified when mode='eps'."
        A = neighbors.radius_neighbors_graph(X, radius=epsilon, mode='distance')
    elif mode == 'descent':
        knn_search_index = pynndescent.NNDescent(
            n_neighbors=n_neighbors,
            data=X,
            metric='euclidean',
            verbose=False
        )
        indices, distances = knn_search_index.neighbor_graph
        # convert to adjacency matrix
        A = np.zeros((X.shape[0], X.shape[0]))
        for i, knn_i in enumerate(indices):
            d_knn_i = distances[i]
            for j, d_ij in zip(knn_i, d_knn_i):
                A[i, j] = d_ij
                A[j, i] = d_ij
    else:
        raise ValueError("Invalid mode. Choose 'nbrs' or 'eps'.")
    # symmetrize the adjacency matrix
    if type(A) != np.ndarray:
        A = A.toarray()
    A = np.maximum(A, A.T)
    assert np.allclose(A, A.T), "The adjacency matrix is not symmetric."
    # convert to networkx graph and symmetrize A
    n_points = X.shape[0]
    nodes = set()
    G = nx.Graph()
    for i in range(n_points):
        G.add_node(i)
        G.nodes[i]['pos'] = X[i] # store the position of the node
        for j in range(i+1, n_points):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j]) # weight is the euclidean distance
                nodes.add(i)
                nodes.add(j)
                # add unweighted entry in dict
                G[i][j]['unweighted'] = 1

    assert G.is_directed() == False, "The graph is directed."
    assert len(G.nodes()) == n_points, "The graph has isolated nodes."
    return G, A


def low_energy_edge_stats(embdng, full_graph, low_energy_graph, pctg=1.0):
    # find average edge distance for original graph in embedding space
    distances = np.zeros(len(full_graph.edges()))
    for idx, (i, j) in enumerate(full_graph.edges()):
        dist = np.linalg.norm(embdng[i] - embdng[j])
        distances[idx] = dist
    # find the average distance
    avg_distance = np.mean(distances)
    # find the std of the distances
    std_distance = np.std(distances)

    # now compute z-scores for each low energy edge
    z_scores = np.zeros(len(low_energy_graph.edges()))
    for idx, (i, j) in enumerate(low_energy_graph.edges()):
        dist = np.linalg.norm(embdng[i] - embdng[j])
        z_scores[idx] = (dist - avg_distance) / std_distance
    z_scores_sorted = np.sort(z_scores)
    # return mean and std of top 10% of z-scores
    top_z_scores = z_scores_sorted[-int(len(z_scores) * pctg):]
    mean_z_score = np.mean(top_z_scores)
    std_z_score = np.std(top_z_scores)
    return mean_z_score, std_z_score
