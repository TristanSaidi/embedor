import networkx as nx
import numpy as np
from src.graph_utils import *

def prune_orcml(G, X, eps, lda, delta=0.8, verbose=False, reattach=True):
    """
    Prune the graph with the orcml method.
    Parameters
    ----------
    G : networkx.Graph
        The graph to prune.
    X : array-like, shape (n_samples, n_features)
        The dataset.
    eps : float
        The epsilon parameter for the proximity graph.
    lda : float
        The lambda parameter for pruning (see paper).
    delta : float, optional
        The delta (confidence) parameter for pruning (see paper).
    Returns
    -------
    G_pruned : networkx.Graph
        The pruned graph.
    """    
    # construct the candidate set C, and filtered graph G'
    C = []
    G_prime = nx.Graph()
    threshold = -1 + 4*(1-delta)
    candidate_edge_indices = []
    for idx, (i, j, d) in enumerate(G.edges(data=True)):
        if d['ricciCurvature'] < threshold:
            C.append((i,j))
            candidate_edge_indices.append(idx)
        else:
            G_prime.add_edge(i, j, weight=d["weight"])
            G_prime[i][j]['ricciCurvature'] = d['ricciCurvature']
            G_prime[i][j]['effective_eps'] = d['effective_eps']
    
    if verbose:
        print(f"Number of candidate edges: {len(C)}, Number of edges in G': {len(G.edges())}")
    # bookkeeping
    num_removed_edges = 0

    G_pruned = G_prime.copy()
    preserved_nodes = set(G_prime.nodes()) # start from G' and add nodes as we go
    preserved_edges = list(range(len(G.edges()))) # start from all edges and remove as we go

    for num, (i, j) in enumerate(C):
        # check distance d_G'(x,y) for all x,y in C
        threshold = ((1-lda)*np.pi**2)/(2*np.sqrt(24*lda))

        if eps is not None:
            threshold *= eps
        else:
            # find the edge distance for all edges incident to i or j
            dists = []
            for k in G.neighbors(i):
                dists.append(G[i][k]['weight'])
            for k in G.neighbors(j):
                dists.append(G[j][k]['weight'])
            effective_eps = np.mean(dists)
            threshold *= effective_eps

        if i not in G_prime.nodes() or j not in G_prime.nodes():
            continue
        try:
            d_G_prime = nx.shortest_path_length(G_prime, source=i, target=j, weight="weight") # use euclidean distance
        except nx.NetworkXNoPath:
            d_G_prime = np.inf

        if d_G_prime > threshold:
            num_removed_edges += 1
            preserved_edges.remove(candidate_edge_indices[num])
            if verbose:
                print(f"Removing Edge {num}: {i} - {j}")
                # print the ratio of d_G'(x,y) to eps
                if eps is not None:
                    print(f"d_G'(x,y)/eps: {d_G_prime/eps}")
                    print(f"Threshold/eps: {threshold/eps}")
                else:
                    print(f"d_G'(x,y)/effective_eps: {d_G_prime/effective_eps}")
                    print(f"Threshold/effective_eps: {threshold/effective_eps}")
                print()
        else:
            G_pruned.add_node(i)
            G_pruned.add_node(j)
            G_pruned.add_edge(i, j, weight=G[i][j]["weight"])
            G_pruned[i][j]['ricciCurvature'] = G[i][j]['ricciCurvature']
            G_pruned[i][j]['effective_eps'] = G[i][j]['effective_eps']
        
            preserved_nodes.add(i)
            preserved_nodes.add(j)

    if len(preserved_nodes) != len(G.nodes()) and reattach:
        print("Warning: There are isolated nodes in the graph. This will be artificially fixed.")
        print(f"Number of isolated nodes: {len(G.nodes()) - len(preserved_nodes)}")
        missing_nodes = set(G.nodes()).difference(preserved_nodes)
        for node_idx in missing_nodes:
            # find nearest neighbor
            isolated_node = X[node_idx]
            dists = np.linalg.norm(X - isolated_node, axis=1)
            dists[node_idx] = np.inf
            nearest_neighbor = np.argmin(dists)
            G_pruned.add_edge(node_idx, nearest_neighbor, weight=dists[nearest_neighbor])
            # assign this edge 0 curvature
            G_pruned[node_idx][nearest_neighbor]['ricciCurvature'] = 0
        assert len(G.nodes()) == len(G_pruned.nodes()), "The number of preserved nodes does not match the number of nodes in the pruned graph."

    preserved_orcs = []
    for i, j, d in G_pruned.edges(data=True):
        preserved_orcs.append(d['ricciCurvature'])
    G_prime_orcs = []
    for i, j, d in G_prime.edges(data=True):
        G_prime_orcs.append(d['ricciCurvature'])
    A_pruned = nx.adjacency_matrix(G_pruned).toarray()
    return {
        'G_pruned': G_pruned,
        'G_prime': G_prime,
        'A_pruned': A_pruned,
        'preserved_edges': preserved_edges,
        'preserved_orcs': preserved_orcs,
        'G_prime_orcs': G_prime_orcs,
        'preserved_nodes': preserved_nodes,
        'C': C,
    }
            

def get_pruned_unpruned_graph(data, exp_params, verbose=False, reattach=True):
    """ 
    Build the nearest neighbor graph and prune it with the orcml method.
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        The dataset.
    exp_params : dict
        The experimental parameters.
    verbose : bool, optional
        Whether to print verbose output for orcml algorithm.
    reattach : bool, optional
        Whether to reattach isolated nodes.
    Returns
    -------
    return_dict : dict
    """

    return_dict = get_nn_graph(data, exp_params)
    G, A = return_dict['G'], return_dict['A']
    return_dict = compute_orc(G)
    orcs = return_dict['orcs']
    pruned_orcml = prune_orcml(return_dict['G'], data, eps=exp_params['epsilon'], lda=exp_params['lda'], delta=exp_params['delta'], verbose=verbose, reattach=reattach)
    G_orcml = pruned_orcml['G_pruned']
    A_orcml = nx.adjacency_matrix(G_orcml).toarray()
    # symmetrize
    A_orcml = np.maximum(A_orcml, A_orcml.T)
    return {
        "G_original": G,
        "A_original": A,
        "G_orcml": G_orcml,
        "A_orcml": A_orcml,
        "preserved_edges": pruned_orcml['preserved_edges'],
        "G_orc": return_dict['G'], # unpruned graph with annotated orc
        "G_prime": pruned_orcml['G_prime'], # orc pruned graph without validation step
        "G_prime_orcs": pruned_orcml['G_prime_orcs'],
        "orcs": orcs,
        "C": pruned_orcml['C'],
    }

# create ORCML class

class ORCManL:

    def __init__(self, exp_params, verbose=False, reattach=True):
        """ 
        Initialize the ORCML class.
        Parameters
        ----------
        exp_params : dict
            The experimental parameters. Includes 'mode', 'n_neighbors', 'epsilon', 'lda', 'delta'.
        verbose : bool, optional
            Whether to print verbose output for ORCManL algorithm.
        reattach : bool, optional
            Whether to reattach isolated nodes.
        """
        self.exp_params = exp_params
        if 'epsilon' not in exp_params:
            self.exp_params['epsilon'] = None
        if 'n_neighbors' not in exp_params:
            self.exp_params['n_neighbors'] = None
        self.verbose = verbose
        self.reattach = reattach

    def fit(self, data):
        """
        Build nearest neighbor graph of data and apply the ORCManL algorithm.
        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            The dataset.
        Returns
        -------
        self : ORCManL
        """
        self.return_dict = get_pruned_unpruned_graph(data, self.exp_params, verbose=self.verbose, reattach=self.reattach)
        self.G_pruned = self.return_dict['G_orcml']
        self.A_pruned = self.return_dict['A_orcml']
        return
    
    def get_pruned_graph(self):
        """
        Get the pruned graph.
        Returns
        -------
        G_pruned : networkx.Graph
            The pruned graph.
        """
        return self.G_pruned