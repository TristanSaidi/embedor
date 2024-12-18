from sklearn import manifold
from sklearn.decomposition import PCA, KernelPCA
import scipy
import numpy as np
import warnings
from sklearn.utils.graph import _fix_connected_components
import numpy as np
import umap
import s_gd2
import scprep
# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from sklearn import manifold
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import scipy.spatial
import numpy as np

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


# Fast classical MDS using random svd
def cmdscale_fast(D, ndim):
    return classic(D=D, n_components=ndim)


def classic(D, n_components=2, random_state=None):
    """Fast CMDS using random SVD

    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances

    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`

    random_state : int, RandomState or None, optional (default: None)
        numpy random state

    Returns
    -------
    Y : array-like, embedded data [n_sample, ndim]
    """

    G = D**2
    D = D - D.mean(axis=0)[None, :]
    D = D - D.mean(axis=1)[:, None]
    D = D + D.mean()
    
    pca = PCA(
        n_components=n_components, svd_solver="randomized", random_state=random_state
    )
    Y = pca.fit_transform(G)
    return Y


def smacof(
    D,
    n_components=2,
    metric=True,
    init=None,
    random_state=None,
    verbose=0,
    max_iter=3000,
    eps=1e-6,
    n_jobs=1,
):
    """Metric and non-metric MDS using SMACOF

    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances

    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`

    metric : bool, optional (default: True)
        Use metric MDS. If False, uses non-metric MDS

    init : array-like or None, optional (default: None)
        Initialization state

    random_state : int, RandomState or None, optional (default: None)
        numpy random state

    verbose : int or bool, optional (default: 0)
        verbosity

    max_iter : int, optional (default: 3000)
        maximum iterations

    eps : float, optional (default: 1e-6)
        stopping criterion

    Returns
    -------
    Y : array-like, shape=[n_samples, n_components]
        embedded data
    """
    # Metric MDS from sklearn
    Y, _ = manifold.smacof(
        D,
        n_components=n_components,
        metric=metric,
        max_iter=max_iter,
        eps=eps,
        random_state=random_state,
        n_jobs=n_jobs,
        n_init=1,
        init=init,
        verbose=verbose,
    )
    return Y


def embed_MDS(
    X_dist,
    ndim=2,
    how="classic",
    solver="smacof",
    seed=None,
    verbose=0,
):
    """Performs classic, metric, and non-metric MDS

    Metric MDS is initialized using classic MDS,
    non-metric MDS is initialized using metric MDS.

    Parameters
    ----------
    X: ndarray [n_samples, n_features]
        2 dimensional input data array with n_samples

    n_dim : int, optional, default: 2
        number of dimensions in which the data will be embedded

    how : string, optional, default: 'classic'
        choose from ['classic', 'metric', 'nonmetric']
        which MDS algorithm is used for dimensionality reduction

    distance_metric : string, optional, default: 'euclidean'
        choose from ['cosine', 'euclidean']
        distance metric for MDS

    solver : {'sgd', 'smacof'}, optional (default: 'sgd')
        which solver to use for metric MDS. SGD is substantially faster,
        but produces slightly less optimal results. Note that SMACOF was used
        for all figures in the PHATE paper.

    n_jobs : integer, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used

    seed: integer or numpy.RandomState, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global numpy random number generator

    Returns
    -------
    Y : ndarray [n_samples, n_dim]
        low dimensional embedding of X using MDS
    """

    if how not in ["classic", "metric", "nonmetric"]:
        raise ValueError(
            "Allowable 'how' values for MDS: 'classic', "
            "'metric', or 'nonmetric'. "
            "'{}' was passed.".format(how)
        )
    if solver not in ["sgd", "smacof"]:
        raise ValueError(
            "Allowable 'solver' values for MDS: 'sgd' or "
            "'smacof'. "
            "'{}' was passed.".format(solver)
        )

    # initialize all by CMDS
    Y_classic = classic(X_dist, n_components=ndim, random_state=seed)
    if how == "classic":
        return Y_classic

    # metric is next fastest
    if solver == "sgd":
        try:
            # use sgd2 if it is available
            Y = sgd(X_dist, n_components=ndim, random_state=seed, init=Y_classic)
            if np.any(~np.isfinite(Y)):
                raise NotImplementedError
        except NotImplementedError:
            # sgd2 currently only supports n_components==2
            Y = smacof(
                X_dist,
                n_components=ndim,
                random_state=seed,
                init=Y_classic,
                metric=True,
            )
    elif solver == "smacof":
        Y = smacof(
            X_dist, n_components=ndim, random_state=seed, init=Y_classic, metric=True
        )
    else:
        raise RuntimeError
    if how == "metric":
        # re-orient to classic
        _, Y, _ = scipy.spatial.procrustes(Y_classic, Y)
        return Y

    # nonmetric is slowest
    Y = smacof(X_dist, n_components=ndim, random_state=seed, init=Y, metric=False)
    # re-orient to classic
    _, Y, _ = scipy.spatial.procrustes(Y_classic, Y)
    return Y


@scprep.utils._with_pkg(pkg="s_gd2", min_version="1.3")
def sgd(D, n_components=2, random_state=None, init=None):
    """Metric MDS using stochastic gradient descent

    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances

    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`

    random_state : int or None, optional (default: None)
        numpy random state

    init : array-like or None
        Initialization algorithm or state to use for MMDS

    Returns
    -------
    Y : array-like, embedded data [n_sample, ndim]
    """
    if not n_components == 2:
        raise NotImplementedError
    N = D.shape[0]
    D = squareform(D)
    # Metric MDS from s_gd2
    Y = s_gd2.mds_direct(N, D, init=init, random_seed=random_state)
    return Y


def smacof(
    D,
    n_components=2,
    metric=True,
    init=None,
    random_state=None,
    verbose=0,
    max_iter=3000,
    eps=1e-6,
    n_jobs=1,
):
    """Metric and non-metric MDS using SMACOF

    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances

    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`

    metric : bool, optional (default: True)
        Use metric MDS. If False, uses non-metric MDS

    init : array-like or None, optional (default: None)
        Initialization state

    random_state : int, RandomState or None, optional (default: None)
        numpy random state

    verbose : int or bool, optional (default: 0)
        verbosity

    max_iter : int, optional (default: 3000)
        maximum iterations

    eps : float, optional (default: 1e-6)
        stopping criterion

    Returns
    -------
    Y : array-like, shape=[n_samples, n_components]
        embedded data
    """
    # Metric MDS from sklearn
    Y, _ = manifold.smacof(
        D,
        n_components=n_components,
        metric=metric,
        max_iter=max_iter,
        eps=eps,
        random_state=random_state,
        n_jobs=n_jobs,
        n_init=1,
        init=init,
        verbose=verbose,
    )
    return Y



class Isomap(manifold.Isomap):

    def __init__(
        self,
        *,
        radius=None,
        n_components=2,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        metric_params=None,
    ):
        super().__init__(
            n_neighbors=None,
            radius=radius,
            n_components=n_components,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            path_method=path_method,
            neighbors_algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
            metric=metric,
            p=p,
            metric_params=metric_params,
        )
    
    def _fit_transform(self, X):
        if self.metric != "precomputed":
            raise ValueError("This Isomap implementation requires a precomputed distance matrix.")

        self.kernel_pca_ = KernelPCA(
            n_components=self.n_components,
            kernel="precomputed",
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
        ).set_output(transform="default")

        self.dist_matrix_ = X # metric is precomputed

        G = self.dist_matrix_**2
        G *= -0.5

        self.embedding_ = self.kernel_pca_.fit_transform(G)
        self._n_features_out = self.embedding_.shape[1]