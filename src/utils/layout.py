import numpy as np
import networkx as nx
from networkx.utils import np_random_state
from scipy.sparse import csr_matrix, issparse

###########################################################################################
###########################################################################################
#################################### adapted from UMAP ####################################
###########################################################################################
###########################################################################################

import numba
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
    subsample_indices, # unused in this function
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


def _optimize_layout_euclidean_single_epoch_subsampled(
    subsample_indices,
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
    for idx in numba.prange(subsample_indices.shape[1]):
        
        i = subsample_indices[0][idx]
        j = subsample_indices[1][idx]
        # current implementation: epoch_of_next_sample == epochs_per_sample (at the beginning)
        # this gets triggered if the number of epochs exceeds the next time sample [i] should be updated
        if epoch_of_next_positive_sample[idx] <= n:

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

            epoch_of_next_positive_sample[idx] += epochs_per_positive_sample[idx] # update sample i in [epochs_per_sample] epochs
        
        if epoch_of_next_negative_sample[idx] <= n:

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
            
            epoch_of_next_negative_sample[idx] += epochs_per_negative_sample[idx] # update sample i in [epochs_per_sample] epochs

    return attractive_loss, repulsive_loss


_nb_optimize_layout_euclidean_single_epoch = numba.njit(
    _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=True
)

_nb_optimize_layout_euclidean_single_epoch_subsampled = numba.njit(
    _optimize_layout_euclidean_single_epoch_subsampled, fastmath=True, parallel=True
)

def make_epochs_per_pair(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n, n)
        The weights of how much we wish to sample each pair.

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
    subsample_indices,
    embedding,
    n_epochs,
    epochs_per_positive_sample,
    epochs_per_negative_sample,
    gamma=1.0,
    initial_alpha=1.0,
    verbose=True,
):

    dim = embedding.shape[1]
    alpha = initial_alpha

    epoch_of_next_positive_sample = epochs_per_positive_sample.copy()
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()

    # Fix for calling UMAP many times for small datasets, otherwise we spend here
    # a lot of time in compilation step (first call to numba function)
    if subsample_indices is not None:
        optimize_fn = _nb_optimize_layout_euclidean_single_epoch_subsampled
    else:
        optimize_fn = _nb_optimize_layout_euclidean_single_epoch


    epochs_list = None
    embedding_list = []
    losses = []
    if isinstance(n_epochs, list):
        epochs_list = n_epochs
        n_epochs = max(epochs_list)

    for n in range(n_epochs):
        # n := epoch
        attractive_loss, repulsive_loss = optimize_fn(
            subsample_indices,
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


def joint_probabilities_nn(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.

    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).

    Parameters
    ----------
    distances : sparse matrix of shape (n_samples, n_samples)
        Distances of samples to its n_neighbors nearest neighbors. All other
        distances are left to zero (and are not materialized in memory).
        Matrix should be of CSR format.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : sparse matrix of shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors. Matrix
        will be of CSR format.
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)    
    conditional_P = _binary_search_perplexity(
        distances_data, desired_perplexity, verbose
    )
    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    P = csr_matrix(
        (conditional_P.ravel(), distances.indices, distances.indptr),
        shape=(n_samples, n_samples),
    )
    P = P + P.T
    # convert from csr back to np.ndarray
    P = P.tocsr().toarray()
    return squareform(P)