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

output_dir = os.path.join(REPO_ROOT, 'outputs', 'frc_approx')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# datetime
import datetime
datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(output_dir, datetime_str)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

n_points = 5000
n_iter = 10
np.random.seed(42)



# %%
noise = 0.1
noise_thresh = None
return_dict = concentric_circles(n_points=n_points, factor=0.4, noise=noise, noise_thresh=noise_thresh)

emb = EmbedOR().fit_transform(return_dict['data'])
emb_frc = EmbedOR(edge_weight='frc').fit_transform(return_dict['data'])

# %%
plt.figure(figsize=(10, 6))
plot_data_2D(emb, None, None)
plt.savefig(os.path.join(output_dir, 'emb_circles.png'), dpi=1200)
plt.figure(figsize=(10, 6))
plot_data_2D(emb_frc, None, None)
plt.savefig(os.path.join(output_dir, 'emb_circles_frc.png'), dpi=1200)
plt.figure(figsize=(10, 6))

# %%
noise = 1
noise_thresh = None
return_dict = swiss_roll(n_points=n_points, noise=noise, noise_thresh=noise_thresh)

emb = EmbedOR().fit_transform(return_dict['data'])
emb_frc = EmbedOR(edge_weight='frc').fit_transform(return_dict['data'])

plt.figure(figsize=(10, 6))
plot_data_2D(emb, None, None)
plt.savefig(os.path.join(output_dir, 'emb_swiss_roll.png'), dpi=1200)
plt.figure(figsize=(10, 6))
plot_data_2D(emb_frc, None, None)
plt.savefig(os.path.join(output_dir, 'emb_swiss_roll_frc.png'), dpi=1200)
# %%
noise = 0.5
noise_thresh = None
return_dict = torus(n_points=n_points, noise=noise, noise_thresh=noise_thresh, double=True)


emb = EmbedOR().fit_transform(return_dict['data'])
emb_frc = EmbedOR(edge_weight='frc').fit_transform(return_dict['data'])

plt.figure(figsize=(10, 6))
plot_data_2D(emb, None, None)
plt.savefig(os.path.join(output_dir, 'emb_torus.png'), dpi=1200)
plt.figure(figsize=(10, 6))
plot_data_2D(emb_frc, None, None)
plt.savefig(os.path.join(output_dir, 'emb_torus_frc.png'), dpi=1200)

# %%
noisy_tree, tree = gen_tree(n_points=n_points)

emb = EmbedOR().fit_transform(noisy_tree)
emb_frc = EmbedOR(edge_weight='frc').fit_transform(noisy_tree)

plt.figure(figsize=(10, 6))
plot_data_2D(emb, None, None)
plt.savefig(os.path.join(output_dir, 'emb_tree.png'), dpi=1200)
plt.figure(figsize=(10, 6))
plot_data_2D(emb_frc, None, None)
plt.savefig(os.path.join(output_dir, 'emb_tree_frc.png'), dpi=1200)


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

spearman_corrs_circles = []
for it in range(n_iter):
    print(f"Iteration {it+1}/{n_iter}")
    return_dict = concentric_circles(n_points=n_points, factor=0.4, noise=noise, noise_thresh=noise_thresh)
    gt_geodesic_distance = est_geodesic_distance(return_dict['noiseless_data'], k=15)
    
    embedor = EmbedOR(edge_weight='frc')
    embedding = embedor.fit_transform(return_dict['data'])
    pdist_embedor = squareform(pdist(embedding, metric='euclidean'))
    spearman_corr_embedor, _ = spearmanr(pdist_embedor.flatten(), gt_geodesic_distance.flatten())
    print(f'Spearman correlation EmbedOR: {spearman_corr_embedor}')
    spearman_corrs_circles.append(spearman_corr_embedor)

spearman_corrs_circles_mean = np.mean(spearman_corrs_circles)
spearman_corrs_circles_std = np.std(spearman_corrs_circles)

# %%
noise = 1
noise_thresh = None

spearman_corrs_swiss_roll = []
for it in range(n_iter):
    print(f"Iteration {it+1}/{n_iter}")
    return_dict = swiss_roll(n_points=n_points, noise=noise, noise_thresh=noise_thresh)
    gt_geodesic_distance = est_geodesic_distance(return_dict['noiseless_data'], k=15)
    
    embedor = EmbedOR(edge_weight='frc')
    embedding = embedor.fit_transform(return_dict['data'])
    pdist_embedor = squareform(pdist(embedding, metric='euclidean'))
    spearman_corr_embedor, _ = spearmanr(pdist_embedor.flatten(), gt_geodesic_distance.flatten())
    print(f'Spearman correlation EmbedOR: {spearman_corr_embedor}')
    spearman_corrs_swiss_roll.append(spearman_corr_embedor)
spearman_corrs_swiss_roll_mean = np.mean(spearman_corrs_swiss_roll)
spearman_corrs_swiss_roll_std = np.std(spearman_corrs_swiss_roll)
# %%
noise = 0.5
noise_thresh = None

spearman_corrs_torus = []
for it in range(n_iter):
    print(f"Iteration {it+1}/{n_iter}")
    return_dict = torus(n_points=n_points, noise=noise, noise_thresh=noise_thresh, double=True)
    gt_geodesic_distance = est_geodesic_distance(return_dict['noiseless_data'], k=15)
    
    embedor = EmbedOR(edge_weight='frc')
    embedding = embedor.fit_transform(return_dict['data'])
    pdist_embedor = squareform(pdist(embedding, metric='euclidean'))
    spearman_corr_embedor, _ = spearmanr(pdist_embedor.flatten(), gt_geodesic_distance.flatten())
    print(f'Spearman correlation EmbedOR: {spearman_corr_embedor}')
    spearman_corrs_torus.append(spearman_corr_embedor)
spearman_corrs_torus_mean = np.mean(spearman_corrs_torus)
spearman_corrs_torus_std = np.std(spearman_corrs_torus)
# %%

spearman_corrs_tree = []
for it in range(n_iter):
    print(f"Iteration {it+1}/{n_iter}")
    noisy_tree, tree = gen_tree(n_points=n_points)
    return_dict = {'data': noisy_tree, 'noiseless_data': tree}
    gt_geodesic_distance = est_geodesic_distance(return_dict['noiseless_data'], k=15)
    embedor = EmbedOR(edge_weight='frc')
    embedding = embedor.fit_transform(return_dict['data'])
    pdist_embedor = squareform(pdist(embedding, metric='euclidean'))
    spearman_corr_embedor, _ = spearmanr(pdist_embedor.flatten(), gt_geodesic_distance.flatten())
    print(f'Spearman correlation EmbedOR: {spearman_corr_embedor}')
    spearman_corrs_tree.append(spearman_corr_embedor)
spearman_corrs_tree_mean = np.mean(spearman_corrs_tree)
spearman_corrs_tree_std = np.std(spearman_corrs_tree)

print(f"Spearman correlation for circles: {spearman_corrs_circles_mean:.4f} ± {spearman_corrs_circles_std:.4f}")
print(f"Spearman correlation for swiss roll: {spearman_corrs_swiss_roll_mean:.4f} ± {spearman_corrs_swiss_roll_std:.4f}")
print(f"Spearman correlation for torus: {spearman_corrs_torus_mean:.4f} ± {spearman_corrs_torus_std:.4f}")
print(f"Spearman correlation for tree: {spearman_corrs_tree_mean:.4f} ± {spearman_corrs_tree_std:.4f}")

# save to a dict
results = {
    'circles': {
        'mean': spearman_corrs_circles_mean,
        'std': spearman_corrs_circles_std
    },
    'swiss_roll': {
        'mean': spearman_corrs_swiss_roll_mean,
        'std': spearman_corrs_swiss_roll_std
    },
    'torus': {
        'mean': spearman_corrs_torus_mean,
        'std': spearman_corrs_torus_std
    },
    'tree': {
        'mean': spearman_corrs_tree_mean,
        'std': spearman_corrs_tree_std
    }
}
# save to a json
import json
with open(os.path.join(output_dir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=4)