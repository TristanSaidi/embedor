from sklearn import manifold
import scipy
import numpy as np
import warnings
from sklearn.utils.graph import _fix_connected_components
import numpy as np
import umap

# embeddings
def UMAP(A, n_neighbors, n_components, X=None):
    """
    Compute the UMAP embedding of a graph.
    Parameters
    ----------
    A : array-like, shape (n_samples, n_samples)
        The adjacency matrix of the graph.
    n_components : int
        The number of components to keep.
    X : array-like, shape (n_samples, n_features), default=None
        Embeddings of the original data. To be used only if the graph is not connected.
    Returns
    -------
    Y : array-like, shape (n_samples, n_components)
        The UMAP embedding of the graph.
    """

    n_connected_components, component_labels = scipy.sparse.csgraph.connected_components(A)
    
    if n_connected_components > 1:
        if X is None:
            raise ValueError("The graph is not connected. Please provide the original data.")
        warnings.warn(
            (
                "The number of connected components of the neighbors graph "
                f"is {n_connected_components} > 1. Completing the graph to fit"
                " UMAP might be slow. Increase the number of neighbors to "
                "avoid this issue."
            ),
            stacklevel=2,
        )
        # use array validated by NearestNeighbors
        ambient_distances = scipy.spatial.distance.pdist(X, metric="euclidean")
        ambient_distances = scipy.spatial.distance.squareform(ambient_distances)

        A = _fix_connected_components(
            X=A,
            graph=ambient_distances,
            component_labels=component_labels,
            n_connected_components=n_connected_components,
            mode="distance",
            metric="precomputed",
        )

    distances = scipy.sparse.csgraph.shortest_path(A, directed=False)
    assert np.allclose(distances, distances.T), "The distance matrix is not symmetric."

    umap_obj = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='precomputed')
    Y = umap_obj.fit_transform(distances)
    return Y


def tsne(A, n_components, X=None):
    """
    Compute the t-SNE embedding of a graph.
    Parameters
    ----------
    A : array-like, shape (n_samples, n_samples)
        The adjacency matrix of the graph.
    n_components : int
        The number of components to keep.
    X : array-like, shape (n_samples, n_features), default=None
        Embeddings of the original data. To be used only if the graph is not connected.
    Returns
    -------
    Y : array-like, shape (n_samples, n_components)
        The t-SNE embedding of the graph.
    """

    n_connected_components, component_labels = scipy.sparse.csgraph.connected_components(A)
    
    if n_connected_components > 1:
        if X is None:
            raise ValueError("The graph is not connected. Please provide the original data.")
        warnings.warn(
            (
                "The number of connected components of the neighbors graph "
                f"is {n_connected_components} > 1. Completing the graph to fit"
                " tSNE might be slow. Increase the number of neighbors to "
                "avoid this issue."
            ),
            stacklevel=2,
        )
        # use array validated by NearestNeighbors
        ambient_distances = scipy.spatial.distance.pdist(X, metric="euclidean")
        ambient_distances = scipy.spatial.distance.squareform(ambient_distances)

        A = _fix_connected_components(
            X=A,
            graph=ambient_distances,
            component_labels=component_labels,
            n_connected_components=n_connected_components,
            mode="distance",
            metric="precomputed",
        )

    distances = scipy.sparse.csgraph.shortest_path(A, directed=False)
    assert np.allclose(distances, distances.T), "The distance matrix is not symmetric."

    tsne = manifold.TSNE(n_components=n_components, metric='precomputed', init='random')
    Y = tsne.fit_transform(distances)
    return Y