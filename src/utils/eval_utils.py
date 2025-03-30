import numpy as np
import scipy


def compute_metrics(edge_labels, preserved_edges, percent=True):
    """ 
    Compute metrics for edge preservation. 

    Returns
    -------
    percent_good_removed: float
        Percent of good edges removed.
    percent_bad_removed: float
        Percent of bad edges removed.
    """

    edge_labels = np.array(edge_labels)
    preserved_edges = np.array(preserved_edges)

    N_good_total = np.sum(edge_labels == 1)
    N_bad_total = np.sum(edge_labels == 0)
    N_good_preserved = np.sum(edge_labels[preserved_edges] == 1)
    N_bad_preserved = np.sum(edge_labels[preserved_edges] == 0)
    percent_good_removed = 1 - (N_good_preserved / N_good_total)
    percent_bad_removed = 1 - (N_bad_preserved / N_bad_total)
    if percent:
        return percent_good_removed, percent_bad_removed
    else:
        return N_good_total - N_good_preserved, N_bad_total - N_bad_preserved
    

def distortion(
    X_emb: np.array,
    intracluster_mask: np.array, # mask indicating whether points are in the same cluster
    X_ambient: np.array = None, # ambient data
    D_ambient: np.array = None # ambient distance matrix
):
    """
    Compute the distortion using the ambient embedding metric.
    """
    assert X_ambient is not None or D_ambient is not None, "Either X_ambient or D_ambient must be provided."
    ics = np.where(~intracluster_mask)
    if X_ambient is not None:
        amb_dists = scipy.spatial.distance.cdist(X_ambient, X_ambient)
    else:
        amb_dists = D_ambient.copy()
    
    pairwise_amb_dists = amb_dists
    pairwise_amb_dists[ics] = 1 # set intracluster distances to 1
    # set diagonal to 1
    np.fill_diagonal(pairwise_amb_dists, 1)

    pairwise_emb_dists = scipy.spatial.distance.cdist(X_emb, X_emb)
    pairwise_emb_dists[ics] = 1 # set intracluster distances to 1
    # set diagonal to 1
    np.fill_diagonal(pairwise_emb_dists, 1)

    # compute the distortion
    distortion = np.sum(np.abs(pairwise_amb_dists - pairwise_emb_dists) / pairwise_amb_dists)
    distortion /= np.sum(intracluster_mask.astype(int))
    return distortion