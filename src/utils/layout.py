import numpy as np
import networkx as nx
from networkx.utils import np_random_state


###########################################################################################
###########################################################################################
#################################### adapted from UMAP ####################################
###########################################################################################
###########################################################################################

import numba
from tqdm.auto import tqdm
import cmath

@numba.njit()
def rdist(x, y):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result

@numba.njit()
def log(x):
    """Natural logarithm function.

    Parameters
    ----------
    x: float
        The value to be clamped.

    Returns
    -------
    The natural logarithm of the input value.
    """
    return cmath.log(x).real

@numba.njit()
def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val

def _optimize_layout_euclidean_single_epoch(
    embedding,
    epochs_per_positive_sample,
    epochs_per_negative_sample,
    gamma,
    dim,
    alpha,
    epoch_of_next_positive_sample,
    epoch_of_next_negative_sample,
    n,
):
    attractive_loss = 0.0
    repulsive_loss = 0.0
    # iterate through each pairwise interaction in our graph
    for i in numba.prange(epochs_per_positive_sample.shape[0]):
        for j in numba.prange(i):
            if i == j:
                continue
            # current implementation: epoch_of_next_sample == epochs_per_sample (at the beginning)
            # this gets triggered if the number of epochs exceeds the next time sample [i] should be updated
            if epoch_of_next_positive_sample[i][j] <= n:

                current = embedding[i]
                other = embedding[j]

                dist_squared = rdist(current, other)

                # compute the loss
                f_ij = 1.0 / (1.0 + pow(dist_squared, 2.0))

                attractive_loss += -log(f_ij)

                if dist_squared > 0.0:
                    grad_coeff = -2.0 * 1.0 * 1.0 * pow(dist_squared, 1.0 - 1.0)
                    grad_coeff /= 1.0 * pow(dist_squared, 1.0) + 1.0
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    grad_d = grad_coeff * (current[d] - other[d])
                    current[d] += grad_d * alpha
                    other[d] += -grad_d * alpha

                epoch_of_next_positive_sample[i][j] += epochs_per_positive_sample[i][j] # update sample i in [epochs_per_sample] epochs
            
            if epoch_of_next_negative_sample[i][j] <= n:

                current = embedding[i]
                other = embedding[j]

                dist_squared = rdist(current, other)

                # compute the loss
                f_ij = 1.0 / (1.0 + pow(dist_squared, 2.0))

                repulsive_loss += -gamma * log(1+0.001-f_ij)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * 1.0
                    grad_coeff /= (0.001 + dist_squared) * (
                        1.0 * pow(dist_squared, 1.0) + 1
                    )
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = grad_coeff * (current[d] - other[d])
                    else:
                        grad_d = 0
                    current[d] += grad_d * alpha
                
                epoch_of_next_negative_sample[i][j] += epochs_per_negative_sample[i][j] # update sample i in [epochs_per_sample] epochs
    
    return attractive_loss, repulsive_loss

_nb_optimize_layout_euclidean_single_epoch = numba.njit(
    _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=True
)

def make_epochs_per_pair(weights, n_epochs, max_iter=None, min_iter=None):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n, n)
        The weights of how much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    norm_weights = weights / weights.sum()
    batch_size = n_epochs / norm_weights.max() # take large enough batch size so highest weight edge is sampled every epoch
    n_samples = (norm_weights * batch_size).astype(int) # number of epochs per sample
    result = np.zeros_like(weights, dtype=np.float64)
    result[n_samples > 0] = n_epochs / np.float64(n_samples[n_samples > 0])
    result[n_samples == 0] = n_epochs
    return result


def optimize_layout_euclidean(
    embedding,
    n_epochs,
    epochs_per_positive_sample,
    epochs_per_negative_sample,
    gamma=1.0,
    initial_alpha=1.0,
    verbose=True,
    tqdm_kwds=None,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).
    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.
    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.
    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.
    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.
    n_epochs: int, or list of int
        The number of training epochs to use in optimization, or a list of
        epochs at which to save the embedding. In case of a list, the optimization
        will use the maximum number of epochs in the list, and will return a list
        of embedding in the order of increasing epoch, regardless of the order in
        the epoch list.
    n_vertices: int
        The number of vertices (0-simplices) in the dataset.
    epochs_per_sample: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.
    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.
    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.
    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    densmap: bool (optional, default False)
        Whether to use the density-augmented densMAP objective
    densmap_kwds: dict (optional, default None)
        Auxiliary data for densMAP
    tqdm_kwds: dict (optional, default None)
        Keyword arguments for tqdm progress bar.
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = embedding.shape[1]
    alpha = initial_alpha

    epoch_of_next_positive_sample = epochs_per_positive_sample.copy()
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()

    # Fix for calling UMAP many times for small datasets, otherwise we spend here
    # a lot of time in compilation step (first call to numba function)
    optimize_fn = _nb_optimize_layout_euclidean_single_epoch

    if tqdm_kwds is None:
        tqdm_kwds = {}

    epochs_list = None
    embedding_list = []
    losses = []
    if isinstance(n_epochs, list):
        epochs_list = n_epochs
        n_epochs = max(epochs_list)

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    for n in range(n_epochs):
        # n := epoch
        attractive_loss, repulsive_loss = optimize_fn(
            embedding,
            epochs_per_positive_sample,
            epochs_per_negative_sample,
            gamma,
            dim,
            alpha,
            epoch_of_next_positive_sample,
            epoch_of_next_negative_sample,
            n,
        )
        if verbose:
            print(
                f"Epoch {n}: loss = {attractive_loss + repulsive_loss}"
            )
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if epochs_list is not None and n in epochs_list:
            embedding_list.append(embedding.copy())
        
        losses.append(attractive_loss + repulsive_loss)
    # Add the last embedding to the list as well
    if epochs_list is not None:
        embedding_list.append(embedding.copy())

    return embedding if epochs_list is None else embedding_list


from sklearn.manifold._utils import _binary_search_perplexity
MACHINE_EPSILON = np.finfo(np.double).eps
from scipy.spatial.distance import squareform

def joint_probabilities(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances.

    Parameters
    ----------
    distances : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _binary_search_perplexity(
        distances, desired_perplexity, verbose
    )
    P = conditional_P + conditional_P.T
    return squareform(P)
