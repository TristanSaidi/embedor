import numpy as np
import networkx as nx
from networkx.utils import np_random_state


@np_random_state("seed")
def force_directed_layout(
    G,
    pos=None,
    *,
    max_iter=100,
    jitter_tolerance=1.0,
    weight=None,
    seed=None,
    dim=2,
):
   
    import numpy as np

    if len(G) == 0:
        return {}
    # parse optional pos positions
    if pos is None:
        pos = nx.random_layout(G, dim=dim, seed=seed)
        pos_arr = np.array(list(pos.values()))
    else:
        pos_arr = np.array(list(pos.values()))

    n = len(G)
    attraction_adjacency = np.zeros((n, dim))
    repulsion_adjacency = np.zeros((n, dim))

    repulsion_pairwise = np.zeros((n, dim))

    A = nx.to_numpy_array(G, weight=weight)
    np.fill_diagonal(A, 0)

    # adjacency matrix for attractive forces
    A_attraction = A.copy()
    A_attraction[A_attraction < 0] = 0

    # adjacency matrix for repulsive forces
    A_repulsion = -1 * A.copy()
    A_repulsion[A_repulsion < 0] = 0
    np.fill_diagonal(A_repulsion, 0)

    def estimate_factor(n, swing, traction, speed, speed_efficiency, jitter_tolerance):
        """Computes the scaling factor for the force in the ForceAtlas2 layout algorithm.

        This   helper  function   adjusts   the  speed   and
        efficiency  of the  layout generation  based on  the
        current state of  the system, such as  the number of
        nodes, current swing, and traction forces.

        Parameters
        ----------
        n : int
            Number of nodes in the graph.
        swing : float
            The current swing, representing the oscillation of the nodes.
        traction : float
            The current traction force, representing the attraction between nodes.
        speed : float
            The current speed of the layout generation.
        speed_efficiency : float
            The efficiency of the current speed, influencing how fast the layout converges.
        jitter_tolerance : float
            The tolerance for jitter, affecting how much speed adjustment is allowed.

        Returns
        -------
        tuple
            A tuple containing the updated speed and speed efficiency.

        Notes
        -----
        This function is a part of the ForceAtlas2 layout algorithm and is used to dynamically adjust the
        layout parameters to achieve an optimal and stable visualization.

        """
        import numpy as np

        # estimate jitter
        opt_jitter = 0.05 * np.sqrt(n)
        min_jitter = np.sqrt(opt_jitter)
        max_jitter = 10
        min_speed_efficiency = 0.05

        other = min(max_jitter, opt_jitter * traction / n**2)
        jitter = jitter_tolerance * max(min_jitter, other)

        if swing / traction > 2.0:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.5
            jitter = max(jitter, jitter_tolerance)
        if swing == 0:
            target_speed = np.inf
        else:
            target_speed = jitter * speed_efficiency * traction / swing

        if swing > jitter * traction:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.7
        elif speed < 1000:
            speed_efficiency *= 1.3

        max_rise = 0.5
        speed = speed + min(target_speed - speed, max_rise * speed)
        return speed, speed_efficiency

    speed = 1
    speed_efficiency = 1
    swing = 1
    traction = 1
    for it in range(max_iter):
        # compute pairwise difference
        diff = pos_arr[:, None] - pos_arr[None]
        
        # attraction
        attraction_adjacency = -np.einsum("ijk, ij -> ik", diff, A_attraction)

        # repulsion (adjacency)
        distance = np.linalg.norm(diff, axis=-1)
        d2 = distance
        # remove self-interaction
        np.fill_diagonal(d2, 1)
        factor = (A_repulsion / d2)
        repulsion_adjacency = -np.einsum("ijk, ij -> ik", diff, factor)

        # repulsion (pairwise)
        factor = (1 / d2) * (1/A.shape[0])
        repulsion_pairwise = np.einsum("ijk, ij -> ik", diff, factor)
        # total forces
        update = attraction_adjacency - repulsion_adjacency + repulsion_pairwise

        # print forces
        print("Attraction:", np.linalg.norm(attraction_adjacency))
        print("Repulsion:", np.linalg.norm(repulsion_adjacency))
        print("Repulsion (pairwise):", np.linalg.norm(repulsion_pairwise))

        # compute total swing and traction
        swing += (np.linalg.norm(pos_arr - update, axis=-1)).sum()
        traction += (0.5 * np.linalg.norm(pos_arr + update, axis=-1)).sum()

        speed, speed_efficiency = estimate_factor(
            n,
            swing,
            traction,
            speed,
            speed_efficiency,
            jitter_tolerance,
        )

        # update pos
        swinging = np.linalg.norm(update, axis=-1)
        factor = speed / (1 + np.sqrt(speed * swinging))
        pos_arr += update * factor[:, None]

        update_norm = np.mean(np.linalg.norm(update, axis=-1))
        print(f"Iteration: {it}, update norm: {update_norm}\n")

    return dict(zip(G, pos_arr))

###########################################################################################
###########################################################################################
#################################### adapted from UMAP ####################################
###########################################################################################
###########################################################################################

import numba
from tqdm.auto import tqdm

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

@numba.njit("i4(i8[:])")
def tau_rand_int(state):
    """A fast (pseudo)-random number generator.

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random int32 value
    """
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ (
        (((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ (
        (((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ (
        (((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]


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
    # iterate through each pairwise interaction in our graph
    for i in numba.prange(epochs_per_positive_sample.shape[0]):
        for j in numba.prange(i):
            # current implementation: epoch_of_next_sample == epochs_per_sample (at the beginning)
            # this gets triggered if the number of epochs exceeds the next time sample [i] should be updated
            if epoch_of_next_positive_sample[i][j] <= n:

                current = embedding[i]
                other = embedding[j]

                dist_squared = rdist(current, other)


                if dist_squared > 0.0:
                    grad_coeff = -2.0 * 1.0 * 1.0 * pow(dist_squared, 1.0 - 1.0)
                    grad_coeff /= 1.0 * pow(dist_squared, 1.0) + 1.0
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                    current[d] += grad_d * alpha
                    other[d] += -grad_d * alpha

                epoch_of_next_positive_sample[i][j] += epochs_per_positive_sample[i][j] # update sample i in [epochs_per_sample] epochs

            if epoch_of_next_negative_sample[i][j] <= n:

                current = embedding[i]
                other = embedding[j]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * 1.0
                    grad_coeff /= (0.001 + dist_squared) * (
                        1.0 * pow(dist_squared, 1.0) + 1
                    )
                elif i == j:
                    continue
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    else:
                        grad_d = 0
                    current[d] += grad_d * alpha
                
                epoch_of_next_negative_sample[i][j] += epochs_per_negative_sample[i][j] # update sample i in [epochs_per_sample] epochs


_nb_optimize_layout_euclidean_single_epoch = numba.njit(
    _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=False
)

############### UMAP version ###############

# Note: this implementation might be erroneous, as the result has no dependence
# on the choice of n_epochs. 

# def make_epochs_per_sample(weights, n_epochs):
#     """Given a set of weights and number of epochs generate the number of
#     epochs per sample for each weight.

#     Parameters
#     ----------
#     weights: array of shape (n_1_simplices)
#         The weights of how much we wish to sample each 1-simplex.

#     n_epochs: int
#         The total number of epochs we want to train for.

#     Returns
#     -------
#     An array of number of epochs per sample, one for each 1-simplex.
#     """
#     result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
#     n_samples = n_epochs * (weights / weights.max())
#     result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
#     return result

############### UMAP version ###############



# converts weights to the number of epochs to SKIP for each sample
# larger weight --> skip fewer epochs
def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights of how much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    norm_weights = weights / weights.sum()

    max_w, min_w = norm_weights.max(), norm_weights.min()
    n_samples = (n_epochs - 1)*(norm_weights - min_w)/(max_w - min_w) + 1

    result[n_samples > 0] = n_epochs/np.float64(n_samples[n_samples > 0])
    return result




# converts weights to the number of epochs to SKIP for each sample
# larger weight --> skip fewer epochs
def make_epochs_per_pair(weights, n_epochs):
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
    result = -1.0 * np.ones_like(weights, dtype=np.float64)
    norm_weights = weights / weights.sum()

    max_w, min_w = norm_weights.max(), norm_weights.min()
    n_samples = (n_epochs - 1)*(norm_weights - min_w)/(max_w - min_w) + 1

    result[n_samples > 0] = n_epochs/np.float64(n_samples[n_samples > 0])
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
    if isinstance(n_epochs, list):
        epochs_list = n_epochs
        n_epochs = max(epochs_list)

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    for n in tqdm(range(n_epochs), **tqdm_kwds):
        # n := epoch
        optimize_fn(
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

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

        if epochs_list is not None and n in epochs_list:
            embedding_list.append(embedding.copy())

    # Add the last embedding to the list as well
    if epochs_list is not None:
        embedding_list.append(embedding.copy())

    return embedding if epochs_list is None else embedding_list

