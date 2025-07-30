# %%
from src.data.data import *
from src.embedor import *
from src.plotting import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving figures
import seaborn as sns
import argparse
import umap
import numpy as np
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path

import phate

REPO_ROOT = os.getenv('PYTHONPATH')
print(f'REPO_ROOT: {REPO_ROOT}')

output_dir = os.path.join(REPO_ROOT, 'outputs', 'landmark')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

n_points = 5000
n_landmarks_array = [500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
np.random.seed(42)



# %%
def est_geodesic_distance(X, k=15, scale_factor=1):
    """
    Estimate the geodesic distance using the knn graph.
    :param k: number of neighbors
    :return: geodesic distance matrix. Note: infinite values get snapped to a scaled version of the max finite value
    """
    knn = kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    knn = knn.toarray()
    knn[knn == 0] = np.inf
    # knn[knn > 0] = 1
    dists = shortest_path(knn, method='D', directed=False)
    # max finite value
    max_dist = np.nanmax(dists[~np.isinf(dists)])
    dists[np.isinf(dists)] = max_dist * scale_factor
    return dists


# %%
# circles

from src.data.data import *
noise = 0.1
noise_thresh = None
return_dict = concentric_circles(n_points=n_points, factor=0.4, noise=noise, noise_thresh=noise_thresh)
gt_geodesic_distance = est_geodesic_distance(return_dict['noiseless_data'], k=15)

spearman_corrs_circles = []
for n_landmarks in n_landmarks_array:
    embedor = EmbedOR(n_landmarks=n_landmarks)
    embedding = embedor.fit_transform(return_dict['data'])
    pdist_embedor = squareform(pdist(embedding, metric='euclidean'))
    spearman_corr_embedor, _ = spearmanr(pdist_embedor.flatten(), gt_geodesic_distance.flatten())
    print(f'Spearman correlation EmbedOR: {spearman_corr_embedor}')
    spearman_corrs_circles.append(spearman_corr_embedor)


# %%
noise = 1
noise_thresh = None
return_dict = swiss_roll(n_points=n_points, noise=noise, noise_thresh=noise_thresh)
gt_geodesic_distance = est_geodesic_distance(return_dict['noiseless_data'], k=15)

spearman_corrs_swiss_roll = []
for n_landmarks in n_landmarks_array:
    embedor = EmbedOR(n_landmarks=n_landmarks)
    embedding = embedor.fit_transform(return_dict['data'])
    pdist_embedor = squareform(pdist(embedding, metric='euclidean'))
    spearman_corr_embedor, _ = spearmanr(pdist_embedor.flatten(), gt_geodesic_distance.flatten())
    print(f'Spearman correlation EmbedOR: {spearman_corr_embedor}')
    spearman_corrs_swiss_roll.append(spearman_corr_embedor)


# %%
noise = 0.5
noise_thresh = None
return_dict = torus(n_points=n_points, noise=noise, noise_thresh=noise_thresh, double=True)
gt_geodesic_distance = est_geodesic_distance(return_dict['noiseless_data'], k=15)

spearman_corrs_torus = []
for n_landmarks in n_landmarks_array:
    embedor = EmbedOR(n_landmarks=n_landmarks)
    embedding = embedor.fit_transform(return_dict['data'])
    pdist_embedor = squareform(pdist(embedding, metric='euclidean'))
    spearman_corr_embedor, _ = spearmanr(pdist_embedor.flatten(), gt_geodesic_distance.flatten())
    print(f'Spearman correlation EmbedOR: {spearman_corr_embedor}')
    spearman_corrs_torus.append(spearman_corr_embedor)

# %%
noisy_tree, tree = gen_dla(n_dim=100, n_branch=8, sigma=4, branch_length=500)
return_dict = {'data': noisy_tree, 'noiseless_data': tree}
gt_geodesic_distance = est_geodesic_distance(return_dict['noiseless_data'], k=15)

spearman_corrs_tree = []
for n_landmarks in n_landmarks_array:
    embedor = EmbedOR(n_landmarks=n_landmarks)
    embedding = embedor.fit_transform(noisy_tree)
    pdist_embedor = squareform(pdist(embedding, metric='euclidean'))
    spearman_corr_embedor, _ = spearmanr(pdist_embedor.flatten(), gt_geodesic_distance.flatten())
    print(f'Spearman correlation EmbedOR: {spearman_corr_embedor}')
    spearman_corrs_tree.append(spearman_corr_embedor)

# %%
plt.figure(figsize=(10, 6))
plt.plot(n_landmarks_array, spearman_corrs_circles, label='Circles', color='blue', marker='o')
plt.plot(n_landmarks_array, spearman_corrs_swiss_roll, label='Swiss Roll', color='orange', marker='o')
plt.plot(n_landmarks_array, spearman_corrs_torus, label='Torus', color='green', marker='o')
plt.plot(n_landmarks_array, spearman_corrs_tree, label='Tree', color='red', marker='o')
plt.xlabel('Number of landmarks', fontsize=14)
plt.ylabel('Spearman Correlation', fontsize=14)
plt.xticks(n_landmarks_array, fontsize=12)
plt.ylim(0, 1)
plt.legend()
plt.savefig(os.path.join(output_dir, 'lmk_spearman_correlations.png'), dpi=1200)
