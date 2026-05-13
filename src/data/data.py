from sklearn import datasets
from src.data.manifold import *
import numpy as np
import torchvision
import torch
import os

# REPO_ROOT = os.getenv('PYTHONPATH')
# DATA_DIR = os.path.join(REPO_ROOT, 'data')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(REPO_ROOT, 'data')

# Data generation functions

def concentric_circles(n_points, factor, noise, noise_thresh=0.275, dim=2):
    """ 
    Generate concentric circles with noise. 
    Parameters
    
    n_points : int
        The number of samples to generate.
    factor : float
        The scaling factor between the circles.
    noise : float
        The standard deviation of the Gaussian noise.
    supersample : bool
        If True, the circles are supersampled.
    supersample_factor : float
        The factor by which to supersample the circles.
    Returns
    -------
    Dictionary providing the following keys:
        data : array-like, shape (n_points, 2)
            The generated samples.
        cluster : array-like, shape (n_points,)
            The integer labels for class membership of each sample.
        data_supersample : array-like, shape (n_points*supersample_factor, 2)
            The supersampled circles.
        subsample_indices : list
            The indices of the subsampled circles.
    """

    N_total = n_points
    subsample_indices = None
    circles, cluster, geodesic_distances = make_circles(n_samples=N_total, factor=factor)
    # if dim = 3, add a third dimension of zeros
    if dim == 3:
        circles = np.concatenate([circles, np.zeros((circles.shape[0], 1))], axis=1)
    # clip noise and resample if necessary
    z =  noise*np.random.randn(*circles.shape)
    if noise_thresh is not None:
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
        while len(resample_indices) > 0:
            z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
            resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    noisy_circles = circles.copy() + z

    return_dict = {
        'data': noisy_circles,
        'cluster': cluster,
        'noiseless_data': circles,
        'geodesic_distances': geodesic_distances
    }
    return return_dict


def quadratics(n_points, noise, supersample=False, supersample_factor=2.5, noise_thresh=0.275, n_clusters=2):
    """
    Generate a dataset of quadratics.
    Parameters
    ----------
    n_points : int
        The number of points to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    Dictionary providing the following
    data : array-like, shape (n_points, 2)
        The generated samples.
    cluster : array-like, shape (n_points,)
        The integer labels for class membership of each sample.
    data_supersample : array-like, shape (n_points*supersample_factor, 2)
        The supersampled samples.
    subsample_indices : list
        The indices of the subsampled samples.
    """
    if n_clusters not in [2,3]:
        raise NotImplementedError("Only 2 or 3 clusters are supported.")
    X = np.random.uniform(-2, 2, (n_points, 1))
    Y = np.zeros((n_points, 1))
    # # bernoulli with p = 0.5 for each point
    # labels = np.random.binomial(1, 0.5, n_points)
    # Y[labels == 0] = 0.2*X[labels == 0]**2
    # Y[labels == 1] = 0.3*X[labels == 1]**2 + 1
    # data = np.concatenate([X, Y], axis=1)
    random = np.random.rand(n_points) # random number between 0 and 1
    labels = np.zeros(n_points)
    if n_clusters == 3:
        labels[random < 0.33] = 0
        labels[(random >= 0.33) & (random < 0.66)] = 1
        labels[random >= 0.66] = 2
    else:
        labels[random < 0.5] = 0
        labels[random >= 0.5] = 1
    
    for label in np.unique(labels):
        Y[labels == label] = 0.2*X[labels == label]**2 + label

    data = np.concatenate([X, Y], axis=1)
    # clip noise and resample if necessary
    z = noise*np.random.randn(n_points, 2)
    if noise_thresh is not None:
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
        while len(resample_indices) > 0:
            z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
            resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    data += z
    return_dict = {
        'data': data,
        'cluster': labels,
        'data_supersample': None,
        'subsample_indices': None
    }    
    return return_dict


def moons(n_points, noise, noise_thresh=0.275, sep=0.5, width=1, dim=2):
    """
    Generate a moons dataset.
    Parameters
    ----------
    n_points : int
        The number of points to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    Dictionary providing the following keys:
        data : array-like, shape (n_points, 2)
            The generated moons.
        cluster : array-like, shape (n_points,)
            The integer labels for class membership of each sample.
        data_supersample : array-like, shape (n_points*supersample_factor, 2)
            The supersampled moons.
        subsample_indices : list
            The indices of the subsampled moons.
    """

    N_total = n_points
    # moons, cluster = datasets.make_moons(n_samples=N_total, noise=0.0)

    n_samples_out = N_total // 2
    n_samples_in = N_total - n_samples_out
    
    outer_circ_x = width * np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = width * (1 - np.cos(np.linspace(0, np.pi, n_samples_in)))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - sep
    if dim == 3:
        # uniform over [-0.2, 0.2] for third dimension
        outer_circ_z = 0.8 * (2 * np.random.rand(N_total) - 1)
        moons = np.vstack(
            [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y), outer_circ_z]
        ).T
    else:
        moons = np.vstack(
            [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
        ).T
    cluster = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )

    # clip noise and resample if necessary
    z =  noise*np.random.randn(*moons.shape)
    if noise_thresh is not None:
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
        while len(resample_indices) > 0:
            z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
            resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    noisy_moons = moons.copy() + z

    return_dict = {
        'data': noisy_moons,
        'cluster': cluster,
        'noiseless_data': moons,
    }
    return return_dict




def us(n_points, noise, noise_thresh=0.275, sep=-1.25, width=1, dim=2):
    """
    Generate a us dataset.
    Parameters
    ----------
    n_points : int
        The number of points to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    Dictionary providing the following keys:
        data : array-like, shape (n_points, 2)
            The generated moons.
        cluster : array-like, shape (n_points,)
            The integer labels for class membership of each sample.
        data_supersample : array-like, shape (n_points*supersample_factor, 2)
            The supersampled moons.
        subsample_indices : list
            The indices of the subsampled moons.
    """

    N_total = n_points
    # moons, cluster = datasets.make_moons(n_samples=N_total, noise=0.0)

    n_samples_out = N_total // 2
    n_samples_in = N_total - n_samples_out
    
    outer_circ_x = width * np.cos(np.linspace(0, np.pi, n_samples_out)) + 1.0
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = width * (1 - np.cos(np.linspace(0, np.pi, n_samples_in)))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - sep
    if dim == 3:
        # uniform over [-0.2, 0.2] for third dimension
        outer_circ_z = 0.8 * (2 * np.random.rand(N_total) - 1)
        moons = np.vstack(
            [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y), outer_circ_z]
        ).T
    else:
        moons = np.vstack(
            [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
        ).T
    cluster = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )

    # clip noise and resample if necessary
    z =  noise*np.random.randn(*moons.shape)
    if noise_thresh is not None:
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
        while len(resample_indices) > 0:
            z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
            resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    noisy_moons = moons.copy() + z

    return_dict = {
        'data': noisy_moons,
        'cluster': cluster,
        'noiseless_data': moons,
    }
    return return_dict

def swiss_roll(n_points, noise, dim=3, noise_thresh=0.275, hole=False, double=False):
    """
    Generate a Swiss roll dataset.
    Parameters
    ----------
    n_points : int
        The number of points to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    Dictionary providing the following keys:
        swiss_roll : array-like, shape (n_points, dim)
            The generated Swiss roll.
        color : array-like, shape (n_points,)
            The color of each point.
        dim: int
            The dimension of the Swiss roll.
    """

    swiss_roll, t, y = make_swiss_roll(n_points, hole=hole)
    cluster = None
    if double:
        swiss_roll2, t_2, _ = make_swiss_roll(n_points, hole=hole)
        # # scale slightly
        # swiss_roll2[:, 2] = swiss_roll2[:, 2] * 0.7
        # swiss_roll2[:, 0] = swiss_roll2[:, 0] * 0.7
        
        # swiss_roll2[:, 2] += 0.5
        # swiss_roll2[:, 0] += -0.4

        # perturb each point along its normal vector
        n_x = -t_2 * np.cos(t_2) + np.sin(t_2)
        n_z = -t_2 * np.sin(t_2) - np.cos(t_2)
        norm = np.sqrt(n_x**2 + n_z**2)
        n_x /= norm
        n_z /= norm
        n_y = np.zeros_like(n_x)
        normal = np.stack([n_x, n_y, n_z], axis=1)
        perturbation = 3 * normal
        swiss_roll2 += perturbation

        swiss_roll = np.concatenate([swiss_roll, swiss_roll2], axis=0)
        cluster = np.zeros(swiss_roll.shape[0])
        cluster[swiss_roll.shape[0]//2:] = 1
    color = t
    # scale t to match that of the embedded manifold
    t = t * (89.37) / (3 * np.pi)
    # compute pairwise geodesic distances
    coordinates = np.stack([t, y], axis=1)
    import sklearn.metrics
    distances = sklearn.metrics.pairwise_distances(coordinates, metric='euclidean')
    
    if dim == 2:
        swiss_roll = swiss_roll[:, [0, 2]]
    # clip noise and resample if necessary
    z =  noise*np.random.randn(*swiss_roll.shape)
    if noise_thresh is not None:
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
        while len(resample_indices) > 0:
            z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
            resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    noisy_swiss_roll = swiss_roll.copy() + z

    return_dict = {
        'data': noisy_swiss_roll,
        'cluster': cluster,
        'color': color,
        'geodesic_distances': distances,
        'noiseless_data': swiss_roll,
        'coordinates': coordinates
    }
    return return_dict


def s_curve(n_points, noise, supersample=False, supersample_factor=2.5, noise_thresh=0.275, dim=2):
    """
    Generate an S-curve dataset.
    Parameters
    ----------
    n_points : int
        The number of points to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    Dictionary providing the following keys:
        data : array-like, shape (n_points, 3)
            The generated S-curve.
        cluster : array-like, shape (n_points,)
            The integer labels for class membership of each sample.
        data_supersample : array-like, shape (n_points*supersample_factor, 3)
            The supersampled S-curve.
        subsample_indices : list
            The indices of the subsampled S-curve.
    """
    if supersample:
        N_total = int(n_points * supersample_factor)
        subsample_indices = np.random.choice(N_total, n_points, replace=False)
    else:
        N_total = n_points
        subsample_indices = None
    s_curve, cluster = datasets.make_s_curve(n_samples=N_total, noise=0.0)
    if dim == 2:
        s_curve = s_curve[:, [0, 2]]
    if supersample:
        s_curve_supersample = s_curve.copy()
        s_curve = s_curve[subsample_indices]
        cluster = cluster[subsample_indices]
    else:
        s_curve_supersample = None

    # clip noise and resample if necessary
    z =  noise*np.random.randn(*s_curve.shape)
    resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    while len(resample_indices) > 0:
        z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    s_curve += z

    return_dict = {
        'data': s_curve,
        'cluster': None,
        'data_supersample': s_curve_supersample,
        'subsample_indices': subsample_indices
    }
    return return_dict


def cassini(n_points, noise, supersample=False, supersample_factor=2.5, noise_thresh=0.275, dim=2, third_dim_radial=False):
    """
    Generate a cassini oval dataset.
    Parameters
    ----------
    n_points : int
        The number of points to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    Dictionary providing the following keys:
        data : array-like, shape (n_points, 2)
            The generated cassini oval.
        cluster : array-like, shape (n_points,)
            The integer labels for class membership of each sample.
        data_supersample : array-like, shape (n_points*supersample_factor, 2)
            The supersampled cassini oval.
        subsample_indices : list
            The indices of the subsampled cassini oval.
    """
    if supersample:
        N_total = int(n_points * supersample_factor)
        subsample_indices = np.random.choice(N_total, n_points, replace=False)
    else:
        N_total = n_points
        subsample_indices = None
    cassini, cluster = Cassini.sample(N=N_total)
    if supersample:
        cassini_supersample = cassini.copy()
        cassini = cassini[subsample_indices]
    else:
        cassini_supersample = None
    if dim == 3:
        if third_dim_radial:
            # choose random rotation in [0, 2pi] about x axis for each point. Should be 3 x 3 x N
            thetas = np.random.uniform(0, 2*np.pi, cassini.shape[0])
            R = np.array([[np.ones(thetas.shape), np.zeros(thetas.shape), np.zeros(thetas.shape)],
                        [np.zeros(thetas.shape), np.cos(thetas), -np.sin(thetas)],
                        [np.zeros(thetas.shape), np.sin(thetas), np.cos(thetas)]])
            # transpose to N x 3 x 3
            R = np.transpose(R, (2, 0, 1))
            # add dimension for matrix multiplication
            cassini = np.concatenate([cassini, np.zeros((cassini.shape[0], 1))], axis=1)
            for i in range(cassini.shape[0]):
                cassini[i] = np.dot(R[i], cassini[i])
        else:
            # uniform in [-1, 1] for third dimension
            cassini = np.concatenate([cassini, 2*np.random.rand(cassini.shape[0], 1) - 1], axis=1)

    # clip noise and resample if necessary
    z =  noise*np.random.randn(*cassini.shape)
    if noise_thresh is not None:
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
        while len(resample_indices) > 0:
            z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
            resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    cassini += z

    return_dict = {
        'data': cassini,
        'cluster': cluster,
        'data_supersample': cassini_supersample,
        'subsample_indices': subsample_indices
    }
    return return_dict


def torus(n_points, noise, r=1.5, R=5, double=False,  noise_thresh=0.275):
    """
    Generate a 2-torus dataset.
    Parameters
    ----------
    n_points : int
        The number of points to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    torus : array-like, shape (n_points, 3)
        The generated torus.
    color : array-like, shape (n_points,)
        The color of each point.
    cluster : array-like, shape (n_points,)
        The cluster labels.
    torus_subsample : array-like, shape (n_points, 3)
        The subsampled torus.
    subsample_indices : list
        The indices of the subsampled torus.
    """
    if double and R <= 2*r:
        raise Warning("Double torii will intersect")
    torus, thetas, cluster, torus_subsample, subsample_indices = Torus.sample(N=n_points, r=r, R=R, double=double, supersample=False, supersample_factor=1)
    color = Torus.exact_curvatures(thetas, r, R)
    color = np.array(color)
    
    # clip noise and resample if necessary
    z =  noise*np.random.randn(*torus.shape)
    if noise_thresh is not None:
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
        while len(resample_indices) > 0:
            z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
            resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    noisy_torus = torus.copy() + z  

    return_dict = {
        'data': noisy_torus,
        'cluster': cluster,
        'color': color,
        'noiseless_data': torus,
    }
    return return_dict

def hyperboloid(n_points, noise, double=False, supersample=False, supersample_factor=2.5, noise_thresh=0.275):
    """ 
    Generate a hyperboloid dataset.
    Parameters
    ----------
    n_points : int
        The number of points to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    hyperboloid : array-like, shape (n_points, 3)
        The generated hyperboloid.
    color : array-like, shape (n_points,)
        The color of each point.
    """
    hyperboloid, cluster, hyperboloid_subsample, subsample_indices = Hyperboloid.sample(n_points, double=double, supersample=supersample, supersample_factor=supersample_factor)
    color = Hyperboloid.S(hyperboloid[:, 2]) # curvature (proxy) for color
    color = np.array(color)

    # clip noise and resample if necessary
    z =  noise*np.random.randn(*hyperboloid.shape)
    resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    while len(resample_indices) > 0:
        z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    hyperboloid += z

    return_dict = {
        'data': hyperboloid,
        'cluster': cluster,
        'color': color,
        'data_supersample': hyperboloid_subsample,
        'subsample_indices': subsample_indices
    }
    return return_dict

def parab_and_hyp(n_points, noise, double=False, supersample=False, supersample_factor=2.5, noise_thresh=0.275):
    """
    Generate a paraboloid and hyperboloid dataset.
    Parameters
    
    n_points : int
        The number of samples to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    Dictionary providing the following
    data : array-like, shape (n_points, 3)
        The generated samples.
    cluster : array-like, shape (n_points,)
        The integer labels for class membership of each sample.
    color : array-like, shape (n_points,)
        The color of each point.
    data_supersample : array-like, shape (n_points*supersample_factor, 3)
        The supersampled samples.
    subsample_indices : list
        The indices of the subsampled samples.
    """

    paraboloid, _ = Paraboloid.sample(N=n_points//2, r=2, z_max=0.75, offset=[0.0, 0.0, 1.75])
    hyperboloid, _, _, _ = Hyperboloid.sample(N=n_points//2, a=0.6, c=1.0, B=4, double=False)
    # rotate so that the hyperboloid is in the x-y plane
    hyperboloid = np.dot(hyperboloid, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    # concatenate with the paraboloid
    parab_and_hyp = np.concatenate([paraboloid, hyperboloid], axis=0)

    # assign cluster labels
    cluster = np.zeros(parab_and_hyp.shape[0])
    cluster[parab_and_hyp.shape[0]//2:] = 1

    # clip noise and resample if necessary
    z =  noise*np.random.randn(*parab_and_hyp.shape)
    resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    while len(resample_indices) > 0:
        z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    parab_and_hyp += z

    return_dict = {
        'data': parab_and_hyp,
        'cluster': cluster,
        'color': cluster,
        'data_supersample': None,
        'subsample_indices': None
    }
    return return_dict

def double_paraboloid(n_points, noise, supersample=False, supersample_factor=2.5, noise_thresh=0.275):
    """
    Generate a double paraboloid dataset.
    Parameters
    ----------
    n_points : int
        The number of points to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    Dictionary providing the following
    data : array-like, shape (n_points, 3)
        The generated samples.
    cluster : array-like, shape (n_points,)
        The integer labels for class membership of each sample.
    color : array-like, shape (n_points,)
        The color of each point.
    data_supersample : array-like, shape (n_points*supersample_factor, 3)
        The supersampled samples.
    subsample_indices : list
        The indices of the subsampled samples.
    """

    paraboloid1, _ = Paraboloid.sample(N=n_points//2, r=4, z_max=0.1, offset=[0.0, 0.0, 0.75])
    paraboloid2, _ = Paraboloid.sample(N=n_points//2, r=4, z_max=0.1, offset=[0.0, 0.0, 0.75])
    double_paraboloid = np.concatenate([paraboloid1, -1 * paraboloid2], axis=0)

    # assign cluster labels
    cluster = np.zeros(double_paraboloid.shape[0])
    cluster[double_paraboloid.shape[0]//2:] = 1

    # clip noise and resample if necessary
    z =  noise*np.random.randn(*double_paraboloid.shape)
    resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    while len(resample_indices) > 0:
        z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    double_paraboloid += z

    return_dict = {
        'data': double_paraboloid,
        'cluster': cluster,
        'color': cluster,
        'data_supersample': None,
        'subsample_indices': None
    }
    return return_dict


def mixture_of_gaussians(n_points, noise, supersample=False, supersample_factor=2.5, noise_thresh=0.275):
    """
    Generate a mixture of Gaussians dataset.
    Parameters
    ----------
    n_points : int
        The number of points to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    Dictionary providing the following
    data : array-like, shape (n_points, 2)
        The generated samples.
    cluster : array-like, shape (n_points,)
        The integer labels for class membership of each sample.
    color : array-like, shape (n_points,)
        The color of each point.
    data_supersample : array-like, shape (n_points*supersample_factor, 2)
        The supersampled samples.
    subsample_indices : list
        The indices of the subsampled samples.
    """

    n_clusters = 3
    n_points_per_cluster = n_points // n_clusters
    n_points = n_points_per_cluster * n_clusters # ensures n_points is divisible by n_clusters
    means = np.array([
        [-0.5, 0.0],
        [0.5, 0.0],
        [0.0, 0.86]
    ])

    data = np.zeros((n_points, 2))
    cluster = np.zeros(n_points)
    for i in range(n_clusters):
        data[i*n_points_per_cluster:(i+1)*n_points_per_cluster] = means[i]
        cluster[i*n_points_per_cluster:(i+1)*n_points_per_cluster] = i

    # clip noise and resample if necessary
    z =  noise*np.random.randn(*data.shape)
    resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    while len(resample_indices) > 0:
        z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    data += z

    return_dict = {
        'data': data,
        'cluster': cluster,
        'color': cluster,
        'data_supersample': None,
        'subsample_indices': None
    }
    return return_dict

def spheres(n_points, noise, supersample=False, supersample_factor=2.5, noise_thresh=0.275):

    """
    Generate a dataset of spheres.
    Parameters
    ----------
    n_points : int
        The number of points to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    Dictionary providing the following
    data : array-like, shape (n_points, 3)
        The generated samples.
    cluster : array-like, shape (n_points,)
        The integer labels for class membership of each sample.
    color : array-like, shape (n_points,)
        The color of each point.
    data_supersample : array-like, shape (n_points*supersample_factor, 3)
        The supersampled samples.
    subsample_indices : list
        The indices of the subsampled samples.
    """
    if supersample:
        N_total = int(n_points * supersample_factor)
        subsample_indices = np.random.choice(N_total, n_points, replace=False)
    else:
        N_total = n_points
        subsample_indices = None
    # bernoulli with p = 0.5 for each point
    cluster = np.random.binomial(1, 0.5, N_total)
    sphere_1 = Sphere.sample(N=sum(cluster), n=2, R=1.0)
    sphere_2 = Sphere.sample(N=N_total-sum(cluster), n=2, R=1.0)
    sphere_2 += np.array([0, 2.3, 0]) # offset
   
    spheres = np.zeros((N_total, 3))
    spheres[cluster == 1] = sphere_1
    spheres[cluster == 0] = sphere_2
    # spheres = np.concatenate([sphere_1, sphere_2], axis=0)

    # clip noise and resample if necessary
    z =  noise*np.random.randn(*spheres.shape)
    resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    while len(resample_indices) > 0:
        z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    spheres += z

    return_dict = {
        'data': spheres,
        'cluster': cluster,
        'color': cluster,
        'data_supersample': None,
        'subsample_indices': subsample_indices
    }
    return return_dict


def get_mnist_data(n_samples, label=None):
    """
    Get n_samples MNIST data points with the specified label. If label is None, get n_samples random data points.
    Parameters:

    n_samples: int
        Number of data points to get
    label: int or None
        Label of the data points to get. If None, get random data points.
    Returns:
    ----------
    mnist_data: np.ndarray
        n_samples x 784 array of MNIST data points
    mnist_labels: np.ndarray
        n_samples array of MNIST labels
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1))
    ])
    mnist = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    mnist_data = torch.stack([x for x, _ in mnist]).numpy().astype(np.float64)
    mnist_labels = torch.tensor([y for _, y in mnist]).numpy().astype(np.float64)
    if label is not None:
        label_indices = np.where(mnist_labels == label)[0]
        np.random.seed(0)
        np.random.shuffle(label_indices)
        label_indices = label_indices[:n_samples]
        mnist_data = mnist_data[label_indices]
        mnist_labels = mnist_labels[label_indices]
    else:
        np.random.seed(0)
        indices = np.random.choice(mnist_data.shape[0], n_samples, replace=False)
        mnist_data = mnist_data[indices]
        mnist_labels = mnist_labels[indices]
    return mnist_data, mnist_labels


def get_fmnist_data(n_samples, label=None):
    """
    Get n_samples Fashion MNIST data points with the specified label. If label is None, get n_samples random data points.
    Parameters:

    n_samples: int
        Number of data points to get
    label: int or None
        Label of the data points to get. If None, get random data points.
    Returns:
    ----------
    fmnist_data: np.ndarray
        n_samples x 784 array of Fashion MNIST data points
    fmnist_labels: np.ndarray
        n_samples array of Fashion MNIST labels
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1))
    ])
    fmnist = torchvision.datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
    fmnist_data = torch.stack([x for x, _ in fmnist]).numpy().astype(np.float64)
    # scale so distances are in a reasonable range
    fmnist_data /= 40
    fmnist_labels = torch.tensor([y for _, y in fmnist]).numpy().astype(np.float64)
    if label is not None:
        label_indices = np.where(fmnist_labels == label)[0]
        np.random.seed(0)
        np.random.shuffle(label_indices)
        label_indices = label_indices[:n_samples]
        fmnist_data = fmnist_data[label_indices]
        fmnist_labels = fmnist_labels[label_indices]
    else:
        np.random.seed(0)
        indices = np.random.choice(fmnist_data.shape[0], n_samples, replace=False)
        fmnist_data = fmnist_data[indices]
        fmnist_labels = fmnist_labels[indices]
    return fmnist_data, fmnist_labels


# kmnist: path data/KMNIST/t10k-images-idx3-ubyte.gz
def get_kmnist_data(n_samples, label=None):
    """
    Get n_samples KMNIST data points with the specified label. If label is None, get n_samples random data points.
    Parameters:

    n_samples: int
        Number of data points to get
    label: int or None
        Label of the data points to get. If None, get random data points.
    Returns:
    ----------
    kmnist_data: np.ndarray
        n_samples x 784 array of KMNIST data points
    kmnist_labels: np.ndarray
        n_samples array of KMNIST labels
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1))
    ])
    kmnist = torchvision.datasets.KMNIST(DATA_DIR, train=True, download=True, transform=transform)
    kmnist_data = torch.stack([x for x, _ in kmnist]).numpy().astype(np.float64)
    # scale so distances are in a reasonable range
    kmnist_data /= 40
    kmnist_labels = torch.tensor([y for _, y in kmnist]).numpy().astype(np.float64)
    if label is not None:
        label_indices = np.where(kmnist_labels == label)[0]
        np.random.seed(0)
        np.random.shuffle(label_indices)
        label_indices = label_indices[:n_samples]
        kmnist_data = kmnist_data[label_indices]
        kmnist_labels = kmnist_labels[label_indices]
    else:
        np.random.seed(0)
        indices = np.random.choice(kmnist_data.shape[0], n_samples, replace=False)
        kmnist_data = kmnist_data[indices]
        kmnist_labels = kmnist_labels[indices]
    return kmnist_data, kmnist_labels


### from https://github.com/KrishnaswamyLab/PHATE?tab=readme-ov-file
def gen_dla(
    n_dim=100, n_branch=3, branch_length=1000, rand_multiplier=2, seed=37, sigma=4
):
    np.random.seed(seed)
    M = np.cumsum(-1 + rand_multiplier * np.random.rand(branch_length, n_dim), 0)
    for i in range(n_branch - 1):
        ind = np.random.randint(branch_length)
        new_branch = np.cumsum(
            -1 + rand_multiplier * np.random.rand(branch_length, n_dim), 0
        )
        M = np.concatenate([M, new_branch + M[ind, :]])

    noise = np.random.normal(0, sigma, M.shape)
    noisy_M = M + noise

    # returns the group labels for each point to make it easier to visualize
    # embeddings
    C = np.array([i // branch_length for i in range(n_branch * branch_length)])

    return noisy_M, M

def gen_tree(n_points=5000, ndim=100, sigma=0.01):
    # choose random (unit) direction
    u = np.random.randn(ndim)
    u /= np.linalg.norm(u)
    # generate n_points/5 time values in [0, 1]
    t = np.random.rand(n_points // 5)
    # generate n_points/5 points along the line in the direction of u
    points = np.outer(t, u)
    # point closest to zero is one root
    root_idx_base = np.argmin(np.linalg.norm(points, axis=1))
    # point furthest from zero is another root
    root_idx_top = np.argmax(np.linalg.norm(points, axis=1))
    # pick two new directions for both roots
    u_base_1 = np.random.randn(ndim)
    u_base_1 /= np.linalg.norm(u_base_1)
    u_base_2 = np.random.randn(ndim)
    u_base_2 /= np.linalg.norm(u_base_2)
    u_top_1 = np.random.randn(ndim)
    u_top_1 /= np.linalg.norm(u_top_1)
    u_top_2 = np.random.randn(ndim)
    u_top_2 /= np.linalg.norm(u_top_2)
    # generate new points in the direction of the new directions stepping from the roots
    points_base_1 = points[root_idx_base] + np.outer(np.linspace(0, 1, n_points // 5), u_base_1)
    points_base_2 = points[root_idx_base] + np.outer(np.linspace(0, 1, n_points // 5), u_base_2)
    points_top_1 = points[root_idx_top] + np.outer(np.linspace(0, 1, n_points // 5), u_top_1)
    points_top_2 = points[root_idx_top] + np.outer(np.linspace(0, 1, n_points // 5), u_top_2)
    # concatenate all points
    points = np.concatenate([points, points_base_1, points_base_2, points_top_1, points_top_2], axis=0)
    # add noise
    noise = sigma * np.random.randn(*points.shape)
    noisy_points = noise.copy() + points
    return noisy_points, points



import scprep
import os

def get_embryoid_body_data(n_points=5000):
    download_path = os.path.expanduser(DATA_DIR)
    sparse=True
    T1 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T0_1A"), sparse=sparse, gene_labels='both')
    T2 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T2_3B"), sparse=sparse, gene_labels='both')
    T3 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T4_5C"), sparse=sparse, gene_labels='both')
    T4 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T6_7D"), sparse=sparse, gene_labels='both')
    T5 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T8_9E"), sparse=sparse, gene_labels='both')
    filtered_batches = []
    for batch in [T1, T2, T3, T4, T5]:
        batch = scprep.filter.filter_library_size(batch, percentile=20, keep_cells='above')
        batch = scprep.filter.filter_library_size(batch, percentile=75, keep_cells='below')
        filtered_batches.append(batch)
    del T1, T2, T3, T4, T5 # removes objects from memory
    EBT_counts, sample_labels = scprep.utils.combine_batches(
        filtered_batches, 
        ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"],
        append_to_cell_names=True
    )
    del filtered_batches # removes objects from memory
    EBT_counts = scprep.filter.filter_rare_genes(EBT_counts, min_cells=10)
    EBT_counts = scprep.normalize.library_size_normalize(EBT_counts)
    mito_genes = scprep.select.get_gene_set(EBT_counts, starts_with="MT-") # Get all mitochondrial genes. There are 14, FYI.

    EBT_counts, sample_labels = scprep.filter.filter_gene_set_expression(
        EBT_counts, sample_labels, genes=mito_genes, 
        percentile=90, keep_cells='below')

    EBT_counts = scprep.transform.sqrt(EBT_counts)

    subsample_indices = np.random.choice(
        EBT_counts.shape[0], 
        size=n_points, 
        replace=False
    )
    EBT_counts_subsampled = EBT_counts.iloc[subsample_indices, :]
    sample_labels_subsampled = sample_labels.iloc[subsample_indices]
    # numpy arrays
    EBT_counts_subsampled = EBT_counts_subsampled.values
    sample_labels_subsampled = sample_labels_subsampled.values
    # convert from strings to ints by enumerating the unique labels
    unique_labels = np.unique(sample_labels_subsampled)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    sample_labels_subsampled = np.array([label_to_int[label] for label in sample_labels_subsampled])

    return EBT_counts_subsampled, sample_labels_subsampled

import wot
import pandas as pd

def get_developmental_data(n_points):
    
    # Path to input files
    VAR_DS_PATH = DATA_DIR + '/developmental/ExprMatrix.var.genes.h5ad'
    CELL_DAYS_PATH = DATA_DIR + '/developmental/cell_days.txt'

    days_df = pd.read_csv(CELL_DAYS_PATH, index_col='id', sep='\t')

    adata_var = wot.io.read_dataset(VAR_DS_PATH, obs=[days_df])
    X = adata_var.X
    days = adata_var.obs['day'].values
    random_indices = np.random.choice(X.shape[0], n_points, replace=False)
    X = X[random_indices, :]
    days = days[random_indices]

    return X, days



def get_chimp_data(n_points):
    
    # Path to input files
    data_path = DATA_DIR+'/chimp/chimp.data.npy'
    labels_path = DATA_DIR+'/chimp/chimp.labels.npy'

    data = np.load(data_path)
    labels = np.load(labels_path)

    # subsample data
    if n_points < data.shape[0]:
        random_indices = np.random.choice(data.shape[0], n_points, replace=False)
        data = data[random_indices, :]
        labels = labels[random_indices]
    return data, labels

def get_macosko_data(n_points):
    import gzip
    import pickle
    data_path = DATA_DIR+'/macosko/macosko_2015.pkl.gz'
    with gzip.open(data_path, 'rb') as f:
        data_and_labels = pickle.load(f)
    # get 50 pcs
    data = data_and_labels['pca_50']
    y_str = data_and_labels['CellType1']
    y_int = np.zeros(y_str.shape)
    # convert to int
    for i, label in enumerate(np.unique(y_str)):
        y_int[y_str == label] = i
    # subsample data
    if n_points < data.shape[0]:
        random_indices = np.random.choice(data.shape[0], n_points, replace=False)
        data = data[random_indices, :]
        y_int = y_int[random_indices]
    return data, y_int
    