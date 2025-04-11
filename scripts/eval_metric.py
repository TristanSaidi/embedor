from src.data.data import *
from src.embedor import *
from src.plotting import *
import pandas as pd
# from sklearn_extra.cluster import KMedoids
from sklearn.cluster import HDBSCAN
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_rand_score
import argparse

sns.set_theme()
# diffusion distance
from pydiffmap import diffusion_map as dm

def energy_to_affinity(energy, sigma):
    affinity = np.exp(-sigma*energy**2)
    return affinity

### diffusion distances
def diffusion_distances(W, ts):
    def diffusion_transition_matrix(W, epsilon):
        """
        Compute the transition matrix for diffusion maps from a weighted adjacency matrix.

        Parameters:
            W (numpy.ndarray): Weighted adjacency matrix (NxN).

        Returns:
            numpy.ndarray: Transition matrix (NxN).
        """
        # symmetrize the matrix
        W = (W + W.T) / 2
        edge_mask = W > 0
        # convert to affinity matrix
        W = np.exp(-W**2 / epsilon)
        W[~edge_mask] = 0  # Set non-edges to zero
        D = np.sum(W, axis=1)  # Compute the degree vector
        D_inv = np.diag(1.0 / D)  # Compute D^(-1)
        # Compute the transition matrix
        transition_mat = D_inv @ W  # D^(-1) * W
        assert np.allclose(np.sum(transition_mat, axis=1), 1), "Transition matrix rows do not sum to 1."
        return transition_mat   
    
    # Compute the transition matrix
    P = diffusion_transition_matrix(W, epsilon=0.8)
    # Compute diffusion distances for each t
    eigenvalues, eigenvectors = np.linalg.eig(P)

    def diffusion_distance(t, eigenvalues, eigenvectors):
        # add dimension to eigenvalues 
        eigenvalues = np.expand_dims(eigenvalues, axis=0)
        # compute diffusion distance as D_t(x,y) = (sum_{j=1}^N (lambda_j^t * (phi_j(x) - phi_j(y))^2))^(1/2)
        eigenval_t = eigenvalues**t
        # broadcast by adding rows
        eigenval_t = np.repeat(eigenval_t, eigenvectors.shape[0], axis=0)
        # elementwise multiplication
        diff_coords = np.multiply(eigenval_t, eigenvectors)
        # take pairwise distances, where each row is a point
        # and each column is a dimension
        D_t = pairwise_distances(diff_coords)
        return D_t
    
    D_t_list = []
    for t in ts:
        D_t = diffusion_distance(t, eigenvalues, eigenvectors)
        D_t_list.append(D_t)
    return D_t_list

def spectral_clustering(affinities):
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='kmeans')
    sc.fit(affinities)
    labels = sc.labels_
    return labels

exp_params = {
    'p': 5
}

def metric_eval(n_points):
    save_path = '/home/tristan/Research/Fa24/isorc/outputs/metric_eval'
    os.makedirs(save_path, exist_ok=True)
    
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.join(save_path, dt_string), exist_ok=False)

    ari_dict = {}

    circles_path = os.path.join(save_path, dt_string, 'circles')
    os.makedirs(circles_path, exist_ok=False)
    # concentric circles
    noise = 0.1
    noise_thresh = None
    return_dict = concentric_circles(n_points=n_points, factor=0.4, noise=noise, noise_thresh=noise_thresh)
    labels = return_dict['cluster']

    # embedor apsp
    embedor = EmbedOR(exp_params)
    embedor.fit(return_dict['data'])
    print("Computed the metric")
    apsp_energy = embedor.apsp_energy.copy()

    # isomap apsp
    print("Computing isomap metric")
    A_euc = nx.to_numpy_array(embedor.G, weight='weight')
    apsp_euc = scipy.sparse.csgraph.shortest_path(A_euc, unweighted=False, directed=False)

    # multiscale diffusion distances
    print("Computing diffusion distances")
    ts = [20, 50, 100, 150]
    # compute pairwise distance by taking euclidean distance of columns
    diff_dists = diffusion_distances(A_euc, ts)

    # ambient distances
    print("Computing ambient distances")
    ambient_dists = pairwise_distances(return_dict['data'])

    apsp_affinities = energy_to_affinity(apsp_energy, 1)
    sc_energy = spectral_clustering(apsp_affinities)
    print("Plotting figures")
    plt.figure(figsize=(10, 10))
    sns.heatmap(apsp_affinities, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(circles_path, 'energy_affinities.png'))
    plt.close()

    euc_affinities = energy_to_affinity(apsp_euc, 1)
    sc_euc = spectral_clustering(euc_affinities)
    # plot heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(euc_affinities, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(circles_path, 'euc_affinities.png'))
    plt.close()

    # plot heatmap
    diff_affinities = energy_to_affinity(diff_dists[0], 1)
    sc_diff = spectral_clustering(diff_affinities)
    plt.figure(figsize=(10, 10))
    sns.heatmap(diff_affinities, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(circles_path, f'diff_affinities_t_{ts[0]}.png'))
    plt.close()

    # plot heatmap
    diff_affinities2 = energy_to_affinity(diff_dists[1], 1)
    sc_diff2 = spectral_clustering(diff_affinities2)
    plt.figure(figsize=(10, 10))
    sns.heatmap(diff_affinities2, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(circles_path, f'diff_affinities_t_{ts[1]}.png'))
    plt.close()

    # plot heatmap
    diff_affinities3 = energy_to_affinity(diff_dists[2], 1)
    sc_diff3 = spectral_clustering(diff_affinities3)
    plt.figure(figsize=(10, 10))
    sns.heatmap(diff_affinities3, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(circles_path, f'diff_affinities_t_{ts[2]}.png'))
    plt.close()

    # plot ambient distances
    ambient_affinities = energy_to_affinity(ambient_dists, 1)
    sc_ambient = spectral_clustering(ambient_affinities)
    plt.figure(figsize=(10, 10))
    sns.heatmap(ambient_affinities, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(circles_path, 'ambient_affinities.png'))
    plt.close()

    # compute adjusted rand index
    circles_dict = {}
    # adjusted rand index
    ari_energy = adjusted_rand_score(labels, sc_energy)
    ari_euc = adjusted_rand_score(labels, sc_euc)
    ari_diff = adjusted_rand_score(labels, sc_diff)
    ari_diff2 = adjusted_rand_score(labels, sc_diff2)
    ari_diff3 = adjusted_rand_score(labels, sc_diff3)
    ari_ambient = adjusted_rand_score(labels, sc_ambient)
    print(f'Adjusted Rand Index (energy): {ari_energy}')
    print(f'Adjusted Rand Index (euclidean): {ari_euc}')
    print(f'Adjusted Rand Index (diffusion, t={ts[0]}): {ari_diff}')
    print(f'Adjusted Rand Index (diffusion, t={ts[1]}): {ari_diff2}')
    print(f'Adjusted Rand Index (diffusion, t={ts[2]}): {ari_diff3}')
    print(f'Adjusted Rand Index (ambient): {ari_ambient}')
    circles_dict['ari_energy'] = ari_energy
    circles_dict['ari_euc'] = ari_euc
    circles_dict[f'ari_diff_t_{ts[0]}'] = ari_diff
    circles_dict[f'ari_diff_t_{ts[1]}'] = ari_diff2  
    circles_dict[f'ari_diff_t_{ts[2]}'] = ari_diff3
    circles_dict['ari_ambient'] = ari_ambient  
    ari_dict['circles'] = circles_dict


    toroids_path = os.path.join(save_path, dt_string, 'tori')
    os.makedirs(toroids_path, exist_ok=False)
    noise = 0.4
    noise_thresh = None
    return_dict = torus(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=False, double=True)
    labels = return_dict['cluster']
    # sort by label
    sort_idx = np.argsort(labels)
    return_dict['data'] = return_dict['data'][sort_idx]
    labels = labels[sort_idx]

    # embedor apsp
    embedor = EmbedOR()
    embedor.fit(return_dict['data'])
    print("Computed the metric")
    apsp_energy = embedor.apsp_energy.copy()

    # isomap apsp
    print("Computing isomap metric")
    A_euc = nx.to_numpy_array(embedor.G, weight='weight')
    apsp_euc = scipy.sparse.csgraph.shortest_path(A_euc, unweighted=False, directed=False)

    # multiscale diffusion distances
    print("Computing diffusion distances")
    ts = [20, 50, 100, 150]
    # compute pairwise distance by taking euclidean distance of columns
    diff_dists = diffusion_distances(A_euc, ts)

    # ambient distances
    print("Computing ambient distances")
    ambient_dists = pairwise_distances(return_dict['data'])

    apsp_affinities = energy_to_affinity(apsp_energy/apsp_energy.mean(), 1)
    plt.figure(figsize=(10, 10))
    sc_energy = spectral_clustering(apsp_affinities)
    print("Plotting figures")
    sns.heatmap(apsp_affinities, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(toroids_path, 'energy_affinities.png'))
    plt.close()

    euc_affinities = energy_to_affinity(apsp_euc/apsp_euc.mean(), 1)
    sc_euc = spectral_clustering(euc_affinities)
    plt.figure(figsize=(10, 10))
    sns.heatmap(euc_affinities, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(toroids_path, 'euc_affinities.png'))
    plt.close()

    diff_affinities = energy_to_affinity(diff_dists[0]/diff_dists[0].mean(), 1)
    sc_diff = spectral_clustering(diff_affinities)
    plt.figure(figsize=(10, 10))
    sns.heatmap(diff_affinities, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(toroids_path, f'diff_affinities_t_{ts[0]}.png'))
    plt.close()

    # plot heatmap
    diff_affinities2 = energy_to_affinity(diff_dists[1]/diff_dists[1].mean(), 1)
    sc_diff2 = spectral_clustering(diff_affinities2)
    plt.figure(figsize=(10, 10))
    sns.heatmap(diff_affinities2, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(toroids_path, f'diff_affinities_t_{ts[1]}.png'))
    plt.close()

    # plot heatmap
    diff_affinities3 = energy_to_affinity(diff_dists[2]/diff_dists[2].mean(), 1)
    sc_diff3 = spectral_clustering(diff_affinities3)
    plt.figure(figsize=(10, 10))
    sns.heatmap(diff_affinities3, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(toroids_path, f'diff_affinities_t_{ts[2]}.png'))
    plt.close()
    
    # plot ambient distances
    ambient_affinities = energy_to_affinity(ambient_dists/ambient_dists.mean(), 1)
    sc_ambient = spectral_clustering(ambient_affinities)
    plt.figure(figsize=(10, 10))
    sns.heatmap(ambient_affinities, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(toroids_path, 'ambient_affinities.png'))
    plt.close()

    # compute adjusted rand index
    toroids_dict = {}
    # adjusted rand index
    ari_energy = adjusted_rand_score(labels, sc_energy)
    ari_euc = adjusted_rand_score(labels, sc_euc)
    ari_diff = adjusted_rand_score(labels, sc_diff)
    ari_diff2 = adjusted_rand_score(labels, sc_diff2)
    ari_diff3 = adjusted_rand_score(labels, sc_diff3)
    ari_ambient = adjusted_rand_score(labels, sc_ambient)
    print(f'Adjusted Rand Index (energy): {ari_energy}')
    print(f'Adjusted Rand Index (euclidean): {ari_euc}')
    print(f'Adjusted Rand Index (diffusion, t={ts[0]}): {ari_diff}')
    print(f'Adjusted Rand Index (diffusion, t={ts[1]}): {ari_diff2}')
    print(f'Adjusted Rand Index (diffusion, t={ts[2]}): {ari_diff3}')
    print(f'Adjusted Rand Index (ambient): {ari_ambient}')
    toroids_dict['ari_energy'] = ari_energy
    toroids_dict['ari_euc'] = ari_euc
    toroids_dict[f'ari_diff_t_{ts[0]}'] = ari_diff
    toroids_dict[f'ari_diff_t_{ts[1]}'] = ari_diff2
    toroids_dict[f'ari_diff_t_{ts[2]}'] = ari_diff3
    toroids_dict['ari_ambient'] = ari_ambient
    ari_dict['tori'] = toroids_dict

    # moons
    moons_path = os.path.join(save_path, dt_string, 'moons')
    os.makedirs(moons_path, exist_ok=False)
    noise = 0.1
    noise_thresh = None
    return_dict = moons(n_points=n_points, noise=noise, noise_thresh=noise_thresh)
    labels = return_dict['cluster']
    # sort by label
    sort_idx = np.argsort(labels)
    return_dict['data'] = return_dict['data'][sort_idx]
    labels = labels[sort_idx]
    
    # embedor apsp
    embedor = EmbedOR()
    embedor.fit(return_dict['data'])
    print("Computed the metric")
    apsp_energy = embedor.apsp_energy.copy()
    # isomap apsp
    print("Computing isomap metric")
    A_euc = nx.to_numpy_array(embedor.G, weight='weight')
    apsp_euc = scipy.sparse.csgraph.shortest_path(A_euc, unweighted=False, directed=False)
    # multiscale diffusion distances
    print("Computing diffusion distances")
    ts = [20, 50, 100, 150]
    # compute pairwise distance by taking euclidean distance of columns
    diff_dists = diffusion_distances(A_euc, ts)
    # ambient distances
    print("Computing ambient distances")
    ambient_dists = pairwise_distances(return_dict['data'])
    apsp_affinities = energy_to_affinity(apsp_energy/apsp_energy.mean(), 1)
    plt.figure(figsize=(10, 10))
    sc_energy = spectral_clustering(apsp_affinities)
    print("Plotting figures")
    sns.heatmap(apsp_affinities, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(moons_path, 'energy_affinities.png'))
    plt.close()
    euc_affinities = energy_to_affinity(apsp_euc/apsp_euc.mean(), 1)
    sc_euc = spectral_clustering(euc_affinities)
    plt.figure(figsize=(10, 10))
    sns.heatmap(euc_affinities, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(moons_path, 'euc_affinities.png'))
    plt.close()
    diff_affinities = energy_to_affinity(diff_dists[0]/diff_dists[0].mean(), 1)
    sc_diff = spectral_clustering(diff_affinities)
    plt.figure(figsize=(10, 10))
    sns.heatmap(diff_affinities, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(moons_path, f'diff_affinities_t_{ts[0]}.png'))
    plt.close()
    # plot heatmap
    diff_affinities2 = energy_to_affinity(diff_dists[1]/diff_dists[1].mean(), 1)
    sc_diff2 = spectral_clustering(diff_affinities2)
    plt.figure(figsize=(10, 10))
    sns.heatmap(diff_affinities2, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(moons_path, f'diff_affinities_t_{ts[1]}.png')) 
    plt.close()
    # plot heatmap
    diff_affinities3 = energy_to_affinity(diff_dists[2]/diff_dists[2].mean(), 1)
    sc_diff3 = spectral_clustering(diff_affinities3)
    plt.figure(figsize=(10, 10))
    sns.heatmap(diff_affinities3, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(moons_path, f'diff_affinities_t_{ts[2]}.png'))
    plt.close()
    # plot ambient distances
    ambient_affinities = energy_to_affinity(ambient_dists/ambient_dists.mean(), 1)
    sc_ambient = spectral_clustering(ambient_affinities)
    plt.figure(figsize=(10, 10))
    sns.heatmap(ambient_affinities, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(moons_path, 'ambient_affinities.png'))
    plt.close()
    # compute adjusted rand index
    moons_dict = {}
    # adjusted rand index
    ari_energy = adjusted_rand_score(labels, sc_energy)
    ari_euc = adjusted_rand_score(labels, sc_euc)
    ari_diff = adjusted_rand_score(labels, sc_diff)
    ari_diff2 = adjusted_rand_score(labels, sc_diff2)
    ari_diff3 = adjusted_rand_score(labels, sc_diff3)
    ari_ambient = adjusted_rand_score(labels, sc_ambient)
    print(f'Adjusted Rand Index (energy): {ari_energy}')
    print(f'Adjusted Rand Index (euclidean): {ari_euc}')
    print(f'Adjusted Rand Index (diffusion, t={ts[0]}): {ari_diff}')
    print(f'Adjusted Rand Index (diffusion, t={ts[1]}): {ari_diff2}')
    print(f'Adjusted Rand Index (diffusion, t={ts[2]}): {ari_diff3}')
    print(f'Adjusted Rand Index (ambient): {ari_ambient}')
    moons_dict['ari_energy'] = ari_energy
    moons_dict['ari_euc'] = ari_euc
    moons_dict[f'ari_diff_t_{ts[0]}'] = ari_diff
    moons_dict[f'ari_diff_t_{ts[1]}'] = ari_diff2
    moons_dict[f'ari_diff_t_{ts[2]}'] = ari_diff3
    moons_dict['ari_ambient'] = ari_ambient
    ari_dict['moons'] = moons_dict

    # save results
    import json
    with open(os.path.join(save_path, dt_string, 'ari_results.json'), 'w') as f:
        json.dump(ari_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the metric.")
    parser.add_argument("--n_points", type=int, default=5000, help="Number of points to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()
    seed = args.seed    
    np.random.seed(seed)
    metric_eval(args.n_points)