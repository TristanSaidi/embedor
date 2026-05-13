import networkx as nx
import numpy as np
from sklearn import neighbors
from src.ollivier_ricci import OllivierRicci
import GraphRicciCurvature.FormanRicci as fr
import pynndescent
import tqdm
import time
from src.utils.timing import timer

@timer
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

@timer
def compute_src(G, method="src-spt", delta=0.5, p=1.0, length_attr="unweighted"):
    """
    Compute Sobolev-Ricci Curvature on edges of a graph.
    Parameters
    ----------
    G : networkx.Graph
    method : str
        Tree method: "src-spt", "src-mst", or "src-randtree"
    delta : float
        Mass kept at self node (analogous to alpha in ORC)
    p : float
        Lp norm parameter
    length_attr : str
        Edge attribute to use as length
    """
    from src.curvature import (
        _neighbors_arrays, _edge_index_arrays,
        _build_tree_parent, _calculate_src_curvature
    )

    if not nx.get_edge_attributes(G, length_attr):
        for (u, v) in G.edges():
            G[u][v][length_attr] = 1.0

    nodes = list(G.nodes())
    root = nodes[0]  \

    neigh_idx_list, deg = _neighbors_arrays(G, nodes)
    u_idx, v_idx, L = _edge_index_arrays(G, nodes, length_attr=length_attr)
    order, parent, plen = _build_tree_parent(G, root, method=method, length_attr=length_attr)

    kappas, _, _, _ = _calculate_src_curvature(
        parent=parent,
        plen=plen,
        method=method,
        step=0,
        n=len(nodes),
        neigh_idx_list=neigh_idx_list,
        deg=deg,
        delta=delta,
        L=L,
        u_idx=u_idx,
        v_idx=v_idx,
        p=p,
    )

    srcs = []
    for e, (u, v, _) in enumerate(G.edges(data=True)):
        G[u][v]['srcCurvature'] = float(kappas[e])
        srcs.append(float(kappas[e]))

    return {
        'G': G,
        'orcs': srcs,  
    }

@timer
def compute_frc(G):
    """
    Compute the Forman-Ricci curvature on edges of a graph.
    Parameters
    ----------
    G : networkx.Graph
        The graph.
    nbrhood_size : int, optional
        Number of hops to consider for neighborhood.
    Returns
    -------
    G : networkx.Graph
        The graph with the Forman-Ricci curvatures as edge attributes.
    """
    frc = fr.FormanRicci(G, weight='unweighted')
    frc.compute_ricci_curvature()
    frcs = []
    for i, j, _ in frc.G.edges(data=True):
        frcs.append(frc.G[i][j]['formanCurvature'])
    return {
        'G': frc.G,
        'frcs': frcs,
    }


@timer
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

@timer
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
    time_start = time.time()
    if mode == 'nbrs':
        assert n_neighbors is not None, "n_neighbors must be specified when mode='nbrs'."
        A = neighbors.kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance')
    elif mode == 'eps':
        assert epsilon is not None, "epsilon must be specified when mode='eps'."
        A = neighbors.radius_neighbors_graph(X, radius=epsilon, mode='distance')
    elif mode == 'descent':
        n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(X.shape[0]))))
        knn_search_index = pynndescent.NNDescent(
            X,
            n_neighbors=n_neighbors,
            metric='euclidean',
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            low_memory=True,
            n_jobs=-1,
            verbose=False,
            compressed=False,
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
    time_end = time.time()
    print(f"\tTime taken to compute the adjacency matrix: {time_end - time_start:.2f} seconds")
    time_start = time.time()
    # symmetrize the adjacency matrix
    if type(A) != np.ndarray:
        A = A.toarray()
    A = np.maximum(A, A.T)
    assert np.allclose(A, A.T), "The adjacency matrix is not symmetric."
    # convert to networkx graph and symmetrize A
    n_points = X.shape[0]
    G = nx.from_numpy_array(A)
    nx.set_edge_attributes(G, 1, 'unweighted')

    assert G.is_directed() == False, "The graph is directed."
    assert len(G.nodes()) == n_points, "The graph has isolated nodes."
    time_end = time.time()
    print(f"\tTime taken to create the graph: {time_end - time_start:.2f} seconds")
    return G, A

@timer
def low_energy_edge_stats(embdng, full_graph, low_energy_graph):
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
    # return mean and std of top {100*pctg}% of z-scores
    mean_z_score = np.mean(z_scores)
    std_z_score = np.std(z_scores)
    return mean_z_score, std_z_score, z_scores

@timer
def low_distance_edge_stats(embdng, full_graph, apsp, frac=0.33):
    # find average edge distance for original graph in embedding space
    distances = np.zeros(len(full_graph.edges()))
    energy = np.zeros(len(full_graph.edges()))
    for idx, (i, j) in enumerate(full_graph.edges()):
        dist = np.linalg.norm(embdng[i] - embdng[j])
        distances[idx] = dist
        energy[idx] = apsp[i, j]
    z_scored_energies = (energy - np.mean(energy)) / np.std(energy)
    # take lowest frac*100% of edges with respect to energy
    sorted_indices = np.argsort(distances)
    bottom_indices = sorted_indices[:int(len(sorted_indices) * frac)]
    # get z scores of energies for these edges
    bottom_z_scores = z_scored_energies[bottom_indices]
    # return mean and std of top {100*pctg}% of z-scores
    mean_z_score = np.mean(bottom_z_scores)
    std_z_score = np.std(bottom_z_scores)
    return mean_z_score, std_z_score, bottom_z_scores
