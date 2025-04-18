## Code from https://github.com/aghickok/curvature ##

import numpy as np
from scipy.special import gamma
import math

##############################################################################
# Torus sampling
##############################################################################

class Torus:
    
    def exact_curvatures(thetas, r, R):
        curvatures = [Torus.S_exact(theta, r, R) for theta in thetas]
        return curvatures

    def sample(N, r, R, double=False, supersample=False, supersample_factor=2.5):
        # N: number of points, r: minor radius, R: major radius
        # double: if True, return a second torus with half of the points rotated and offset
        # supersample: if True, sample N*supersample_factor points. Likely to be used for accurate geodesic distance computation

        if supersample:
            N_total = int(N*supersample_factor)
            subsample_indices = np.random.choice(N_total, N, replace=False)
        else:
            N_total = N

        psis = [np.random.random()*2*math.pi for i in range(N_total)]
        j = 0
        thetas = []
        while j < N_total:
            theta = np.random.random()*2*math.pi
            #eta = np.random.random()*2*(r/R) + 1 - (r/R)
            #if eta < 1 + (r/R)*math.cos(theta):
            eta = np.random.random()/math.pi
            if eta < (1 + (r/R)*math.cos(theta))/(2*math.pi):
                thetas.append(theta)
                j += 1
    
        def embed_torus(theta, psi):
            x = (R + r*math.cos(theta))*math.cos(psi)
            y = (R + r*math.cos(theta))*math.sin(psi)
            z = r*math.sin(theta)
            return [x, y, z]
    
        X = np.array([embed_torus(thetas[i], psis[i]) for i in range(N_total)])

        if double:
            # randomly pick half of points to rotate and offset
            indices = np.random.choice(N_total, N_total//2, replace=False)
            for i in indices:
                # rotate by pi/2 about x-axis
                x = X[i, 0]
                y = X[i, 1]
                z = X[i, 2]
                X[i, 0] = x
                X[i, 1] = z
                X[i, 2] = -y
                # offset
                X[i, 0] += R
            # get one-hot encoding of which points were rotated
            rotated = np.zeros(N_total)
            rotated[indices] = 1
        else:
            rotated = None
        
        if supersample:
            X_supersample = X.copy()
            X = X[subsample_indices]
            thetas = [thetas[i] for i in subsample_indices]
            if rotated is not None:
                rotated = rotated[subsample_indices]
        else:
            X_supersample = None
            subsample_indices = None
        return X, np.array(thetas), rotated, X_supersample, subsample_indices
    
    def S_exact(theta, r, R):
        # Analytic scalar curvature
        S = (2*math.cos(theta))/(r*(R + r*math.cos(theta)))
        return S
    
    def theta_index(theta, thetas):
        # Returns index in thetas of the angle closest to theta
        err = [abs(theta_ - theta) for theta_ in thetas]
        return np.argmin(err)
    
    def area(r, R):
        return (2 * math.pi * r) * (2 * math.pi * R)
    

##############################################################################
# Swiss roll sampling
##############################################################################

from sklearn.utils import check_random_state

def make_swiss_roll(n_samples=100, *, noise=0.0, random_state=None, hole=False):
    """Generate a swiss roll dataset.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of sample points on the Swiss Roll.

    noise : float, default=0.0
        The standard deviation of the gaussian noise.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    hole : bool, default=False
        If True generates the swiss roll with hole dataset.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        The points.

    t : ndarray of shape (n_samples,)
        The univariate position of the sample according to the main dimension
        of the points in the manifold.

    Notes
    -----
    The algorithm is from Marsland [1].

    References
    ----------
    .. [1] S. Marsland, "Machine Learning: An Algorithmic Perspective", 2nd edition,
           Chapter 6, 2014.
           https://homepages.ecs.vuw.ac.nz/~marslast/Code/Ch6/lle.py

    Examples
    --------
    >>> from sklearn.datasets import make_swiss_roll
    >>> X, t = make_swiss_roll(noise=0.05, random_state=0)
    >>> X.shape
    (100, 3)
    >>> t.shape
    (100,)
    """
    generator = check_random_state(random_state)

    if not hole:
        t = 1.5 * np.pi * (1 + 2 * generator.uniform(size=n_samples))
        y = 21 * generator.uniform(size=n_samples)
    else:
        corners = np.array(
            [[np.pi * (1.5 + i), j * 7] for i in range(3) for j in range(3)]
        )
        corners = np.delete(corners, 4, axis=0)
        corner_index = generator.choice(8, n_samples)
        parameters = generator.uniform(size=(2, n_samples)) * np.array([[np.pi], [7]])
        t, y = corners[corner_index].T + parameters

    x = t * np.cos(t)
    z = t * np.sin(t)

    X = np.vstack((x, y, z))
    X += noise * generator.standard_normal(size=(3, n_samples))
    X = X.T
    t = np.squeeze(t)

    return X, t, y


def make_circles(
    n_samples=100, *, noise=None, random_state=None, factor=0.8
):
    """Make a large circle containing a smaller circle in 2d.

    A simple toy dataset to visualize clustering and classification
    algorithms.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int or tuple of shape (2,), dtype=int, default=100
        If int, it is the total number of points generated.
        For odd numbers, the inner circle will have one point more than the
        outer circle.
        If two-element tuple, number of points in outer circle and inner
        circle.

        .. versionchanged:: 0.23
           Added two-element tuple.

    shuffle : bool, default=True
        Whether to shuffle the samples.

    noise : float, default=None
        Standard deviation of Gaussian noise added to the data.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    factor : float, default=.8
        Scale factor between inner and outer circle in the range `[0, 1)`.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels (0 or 1) for class membership of each sample.

    Examples
    --------
    >>> from sklearn.datasets import make_circles
    >>> X, y = make_circles(random_state=42)
    >>> X.shape
    (100, 2)
    >>> y.shape
    (100,)
    >>> list(y[:5])
    [1, 1, 1, 0, 0]
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    generator = check_random_state(random_state)
    # so as not to have the first point = last point, we set endpoint=False
    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
    outer_circ_x = np.cos(linspace_out)
    outer_circ_y = np.sin(linspace_out)
    inner_circ_x = np.cos(linspace_in) * factor
    inner_circ_y = np.sin(linspace_in) * factor

    # compute pairwise distances
    outer_circ_ang_dist = np.abs(linspace_out[:, None] - linspace_out[None, :])
    outer_circ_ang_dist = np.minimum(outer_circ_ang_dist, 2 * np.pi - outer_circ_ang_dist)
    # r * delta_theta
    outer_circ_dist = 1 * outer_circ_ang_dist

    inner_circ_ang_dist = np.abs(linspace_in[:, None] - linspace_in[None, :])
    inner_circ_ang_dist = np.minimum(inner_circ_ang_dist, 2 * np.pi - inner_circ_ang_dist)
    # r * delta_theta
    inner_circ_dist = factor * inner_circ_ang_dist

    geodesic_distances = np.ones((n_samples, n_samples)) * np.inf
    geodesic_distances[:n_samples_out, :n_samples_out] = outer_circ_dist
    geodesic_distances[n_samples_out:, n_samples_out:] = inner_circ_dist

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y, geodesic_distances

