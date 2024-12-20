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
from sklearn.utils import check_array, check_random_state, check_symmetric
from joblib import effective_n_jobs
from sklearn.utils.parallel import Parallel, delayed
from sklearn.metrics import euclidean_distances
from sklearn.isotonic import IsotonicRegression


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


class MDS(manifold.MDS):
    
    def __init__(
        self,
        *,
        n_components=2,
        metric=True,
        n_init=4,
        max_iter=300,
        verbose=0,
        eps=0.001,
        n_jobs=None,
        random_state=None,
        dissimilarity="precomputed",
    ):
        super().__init__(
            n_components=n_components,
            metric=metric,
            n_init=n_init,
            max_iter=max_iter,
            verbose=verbose,
            eps=eps,
            n_jobs=n_jobs,
            random_state=random_state,
            dissimilarity=dissimilarity,
        )

    def fit_transform(self, X, y=None, init=None, weight=None):
        """
        Fit the data from `X`, and returns the embedded coordinates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored
            Not used, present for API consistency by convention.

        init : ndarray of shape (n_samples, n_components), default=None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.

        weight : ndarray of shape (n_samples, n_samples), default=None
            symmetric weighting matrix of similarities.
            In default, weight is set to None, suggesting all weights are 1.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            X transformed in the new space.
        """
        X = self._validate_data(X)
        self.dissimilarity_matrix_ = X
        self.embedding_, self.stress_, self.n_iter_ = smacof(
            self.dissimilarity_matrix_,
            metric=self.metric,
            n_components=self.n_components,
            init=init,
            n_init=self.n_init,
            n_jobs=self.n_jobs,
            max_iter=self.max_iter,
            verbose=self.verbose,
            eps=self.eps,
            random_state=self.random_state,
            return_n_iter=True,
            normalized_stress=self.normalized_stress,
            weight=weight,
        )

        return self.embedding_
    
def smacof(
    dissimilarities,
    *,
    metric=True,
    n_components=2,
    init=None,
    n_init=8,
    n_jobs=None,
    max_iter=300,
    verbose=0,
    eps=1e-3,
    random_state=None,
    return_n_iter=False,
    normalized_stress="warn",
    weight=None,
):
    """Compute multidimensional scaling using the SMACOF algorithm.

    The SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm is a
    multidimensional scaling algorithm which minimizes an objective function
    (the *stress*) using a majorization technique. Stress majorization, also
    known as the Guttman Transform, guarantees a monotone convergence of
    stress, and is more powerful than traditional techniques such as gradient
    descent.

    The SMACOF algorithm for metric MDS can be summarized by the following
    steps:

    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.

    The nonmetric algorithm adds a monotonic regression step before computing
    the stress.

    Parameters
    ----------
    dissimilarities : array-like of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    metric : bool, default=True
        Compute metric or nonmetric SMACOF algorithm.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : array-like of shape (n_samples, n_components), default=None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    n_init : int, default=8
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress. If ``init`` is
        provided, this option is overridden and a single run is performed.

    n_jobs : int, default=None
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    normalized_stress : bool or "auto" default=False
        Whether use and return normed stress value (Stress-1) instead of raw
        stress calculated by default. Only supported in non-metric MDS.

    weight : ndarray of shape (n_samples, n_samples), default=None
        symmetric weighting matrix of similarities.
        In default, weight is set to None, suggesting all weights are 1.

        .. versionadded:: 1.2

    Returns
    -------
    X : ndarray of shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
        If `normalized_stress=True`, and `metric=False` returns Stress-1.
        A value of 0 indicates "perfect" fit, 0.025 excellent, 0.05 good,
        0.1 fair, and 0.2 poor [1]_.

    n_iter : int
        The number of iterations corresponding to the best stress. Returned
        only if ``return_n_iter`` is set to ``True``.

    References
    ----------
    .. [1] "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
           Psychometrika, 29 (1964)

    .. [2] "Multidimensional scaling by optimizing goodness of fit to a nonmetric
           hypothesis" Kruskal, J. Psychometrika, 29, (1964)

    .. [3] "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
           Groenen P. Springer Series in Statistics (1997)
    """

    dissimilarities = check_array(dissimilarities)
    random_state = check_random_state(random_state)

    # TODO(1.4): Remove
    if normalized_stress == "warn":
        warnings.warn(
            (
                "The default value of `normalized_stress` will change to `'auto'` in"
                " version 1.4. To suppress this warning, manually set the value of"
                " `normalized_stress`."
            ),
            FutureWarning,
        )
        normalized_stress = False

    if normalized_stress == "auto":
        normalized_stress = not metric

    if normalized_stress and metric:
        raise ValueError(
            "Normalized stress is not supported for metric MDS. Either set"
            " `normalized_stress=False` or use `metric=False`."
        )
    if hasattr(init, "__array__"):
        init = np.asarray(init).copy()
        if not n_init == 1:
            warnings.warn(
                "Explicit initial positions passed: "
                "performing only one init of the MDS instead of %d" % n_init
            )
            n_init = 1

    best_pos, best_stress = None, None

    if effective_n_jobs(n_jobs) == 1:
        for it in range(n_init):
            pos, stress, n_iter_ = _smacof_single(
                dissimilarities,
                metric=metric,
                n_components=n_components,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                eps=eps,
                random_state=random_state,
                normalized_stress=normalized_stress,
                weight=weight,
            )
            if best_stress is None or stress < best_stress:
                best_stress = stress
                best_pos = pos.copy()
                best_iter = n_iter_
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(_smacof_single)(
                dissimilarities,
                metric=metric,
                n_components=n_components,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                eps=eps,
                random_state=seed,
                normalized_stress=normalized_stress,
                weight=weight,
            )
            for seed in seeds
        )
        positions, stress, n_iters = zip(*results)
        best = np.argmin(stress)
        best_stress = stress[best]
        best_pos = positions[best]
        best_iter = n_iters[best]

    if return_n_iter:
        return best_pos, best_stress, best_iter
    else:
        return best_pos, best_stress


def _smacof_single(
    dissimilarities,
    metric=True,
    n_components=2,
    init=None,
    max_iter=300,
    verbose=0,
    eps=1e-3,
    random_state=None,
    normalized_stress=False,
    weight=None,
):
    """Computes multidimensional scaling using SMACOF algorithm. From https://github.com/scikit-learn/scikit-learn/blob/dc66fab2387214418d80034761f2c22797c7a651/sklearn/manifold/_mds.py

    Parameters
    ----------
    dissimilarities : ndarray of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    metric : bool, default=True
        Compute metric or nonmetric SMACOF algorithm.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : ndarray of shape (n_samples, n_components), default=None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    normalized_stress : bool, default=False
        Whether use and return normed stress value (Stress-1) instead of raw
        stress calculated by default. Only supported in non-metric MDS. The
        caller must ensure that if `normalized_stress=True` then `metric=False`

    weight : ndarray of shape (n_samples, n_samples), default=None
        symmetric weighting matrix of similarities.
        In default, weight is set to None, suggesting all weights are 1.

        .. versionadded:: 1.2

    Returns
    -------
    X : ndarray of shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
        If `normalized_stress=True`, and `metric=False` returns Stress-1.
        A value of 0 indicates "perfect" fit, 0.025 excellent, 0.05 good,
        0.1 fair, and 0.2 poor [1]_.

    n_iter : int
        The number of iterations corresponding to the best stress.

    References
    ----------
    .. [1] "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
           Psychometrika, 29 (1964)

    .. [2] "Multidimensional scaling by optimizing goodness of fit to a nonmetric
           hypothesis" Kruskal, J. Psychometrika, 29, (1964)

    .. [3] "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
           Groenen P. Springer Series in Statistics (1997)
    """
    dissimilarities = check_symmetric(dissimilarities, raise_exception=True)

    n_samples = dissimilarities.shape[0]
    random_state = check_random_state(random_state)

    sim_flat = ((1 - np.tri(n_samples)) * dissimilarities).ravel()
    sim_flat_w = sim_flat[sim_flat != 0]
    if init is None:
        # Randomly choose initial configuration
        X = random_state.uniform(size=n_samples * n_components)
        X = X.reshape((n_samples, n_components))
    else:
        # overrides the parameter p
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError(
                "init matrix should be of shape (%d, %d)" % (n_samples, n_components)
            )
        X = init

    old_stress = None
    ir = IsotonicRegression()
    for it in range(max_iter):
        # Compute distance and monotonic regression
        dis = euclidean_distances(X)

        if metric:
            disparities = dissimilarities
        else:
            dis_flat = dis.ravel()
            # dissimilarities with 0 are considered as missing values
            dis_flat_w = dis_flat[sim_flat != 0]

            # Compute the disparities using a monotonic regression
            disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)
            disparities = dis_flat.copy()
            disparities[sim_flat != 0] = disparities_flat
            disparities = disparities.reshape((n_samples, n_samples))
            disparities *= np.sqrt(
                (n_samples * (n_samples - 1) / 2) / (disparities**2).sum()
            )
        if weight is None:
            weight = np.ones(disparities.shape)
        # Compute stress
        stress = (weight.ravel() * (dis.ravel() - disparities.ravel()) ** 2).sum() / 2
        if normalized_stress:
            stress = np.sqrt(stress / ((weight.ravel() * disparities.ravel() ** 2).sum() / 2))
        # Update X using the Guttman transform
        dis[dis == 0] = 1e-5
        ratio = disparities / dis
        B = - ratio
        B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
        X = 1.0 / n_samples * np.dot(B, X)

        dis = np.sqrt((X**2).sum(axis=1)).sum()
        if verbose >= 2:
            print("it: %d, stress %s" % (it, stress))
        if old_stress is not None:
            if (old_stress - stress / dis) < eps:
                if verbose:
                    print("breaking at iteration %d with stress %s" % (it, stress))
                break
        old_stress = stress / dis

    return X, stress, it + 1