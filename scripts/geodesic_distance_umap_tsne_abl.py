from src.data.data import *
from src.embedor import *
from src.plotting import *
from src.utils.orcmanl import ORCManL
import pandas as pd
import matplotlib
import seaborn as sns
import argparse
import umap
import numpy as np
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
import json
import phate

n_points = 5000
# set seed
np.random.seed(0)

REPO_ROOT = os.getenv('PYTHONPATH')
output_dir = os.path.join(REPO_ROOT, 'outputs', 'geodesic_distance_umap_tsne_abl')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
datetime_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(output_dir, datetime_str)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

def geodesic_dist_experiment(n_iter=10, dataset='circles'):

    # make output path
    output_path = os.path.join(output_dir, f"{dataset}_n_iter_{n_iter}.json")

    # umap parameters to vary
    min_dists = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    negative_sample_rate = [1, 2, 3, 4, 5, 10]
    # tsne parameters to vary
    perplexity = [5, 10, 15, 30, 40, 50, 100]
    early_exaggeration = [2, 4, 8, 12, 16, 24]

    # create a dictionary to store the results
    save_dict = {
        "umap": {},
        "tsne": {}
    }
    for min_dist in min_dists:
        save_dict["umap"][f"min_dist_{min_dist}"] = []
    for neg_sample in negative_sample_rate:
        save_dict["umap"][f"neg_sample_rate_{neg_sample}"] = []
    for perp in perplexity:
        save_dict["tsne"][f"perplexity_{perp}"] = []
    for early_exag in early_exaggeration:
        save_dict["tsne"][f"early_exaggeration_{early_exag}"] = []

    for it in range(n_iter):
        print(f"Iteration {it+1}/{n_iter}")

        if dataset == 'circles':
            data_dict = concentric_circles(n_points=n_points, factor=0.4, noise=0.1, noise_thresh=None)
        elif dataset == 'swiss_roll':
            data_dict = swiss_roll(n_points=n_points, noise=1, noise_thresh=None)
        elif dataset == 'torus':
            data_dict = torus(n_points=n_points, noise=0.5, noise_thresh=None, double=True)
        elif dataset == 'tree':
            noisy_tree, tree = gen_tree(n_points=n_points)
            data_dict = {'data': noisy_tree, 'noiseless_data': tree}
    
        gt_geodesic_distance = est_geodesic_distance(data_dict['noiseless_data'], k=15)

        for min_dist in min_dists:
            umap_emb = umap.UMAP(n_neighbors=15, min_dist=min_dist).fit_transform(data_dict['data'])
            pdist_umap = squareform(pdist(umap_emb, metric='euclidean'))
            spearman_corr_umap, _ = spearmanr(pdist_umap.flatten(), gt_geodesic_distance.flatten())
            save_dict["umap"][f"min_dist_{min_dist}"].append(spearman_corr_umap)
        for neg_sample in negative_sample_rate:
            umap_emb = umap.UMAP(n_neighbors=15, negative_sample_rate=neg_sample).fit_transform(data_dict['data'])
            pdist_umap = squareform(pdist(umap_emb, metric='euclidean'))
            spearman_corr_umap, _ = spearmanr(pdist_umap.flatten(), gt_geodesic_distance.flatten())
            save_dict["umap"][f"neg_sample_rate_{neg_sample}"].append(spearman_corr_umap)
        for perp in perplexity:
            tsne_emb = TSNE(n_components=2, perplexity=perp).fit_transform(data_dict['data'])
            pdist_tsne = squareform(pdist(tsne_emb, metric='euclidean'))
            spearman_corr_tsne, _ = spearmanr(pdist_tsne.flatten(), gt_geodesic_distance.flatten())
            save_dict["tsne"][f"perplexity_{perp}"].append(spearman_corr_tsne)
        for early_exag in early_exaggeration:
            tsne_emb = TSNE(n_components=2, perplexity=30, early_exaggeration=early_exag).fit_transform(data_dict['data'])
            pdist_tsne = squareform(pdist(tsne_emb, metric='euclidean'))
            spearman_corr_tsne, _ = spearmanr(pdist_tsne.flatten(), gt_geodesic_distance.flatten())
            save_dict["tsne"][f"early_exaggeration_{early_exag}"].append(spearman_corr_tsne)


    print('*'*100)
    print(f"Results for dataset: {dataset}")
    for min_dist in min_dists:
        print(f"UMAP min_dist={min_dist}: Spearman correlation = {np.mean(save_dict['umap'][f'min_dist_{min_dist}']):.4f} ± {np.std(save_dict['umap'][f'min_dist_{min_dist}']):.4f}")
    for neg_sample in negative_sample_rate:
        print(f"UMAP neg_sample_rate={neg_sample}: Spearman correlation = {np.mean(save_dict['umap'][f'neg_sample_rate_{neg_sample}']):.4f} ± {np.std(save_dict['umap'][f'neg_sample_rate_{neg_sample}']):.4f}")
    for perp in perplexity:
        print(f"TSNE perplexity={perp}: Spearman correlation = {np.mean(save_dict['tsne'][f'perplexity_{perp}']):.4f} ± {np.std(save_dict['tsne'][f'perplexity_{perp}']):.4f}")
    for early_exag in early_exaggeration:
        print(f"TSNE early_exaggeration={early_exag}: Spearman correlation = {np.mean(save_dict['tsne'][f'early_exaggeration_{early_exag}']):.4f} ± {np.std(save_dict['tsne'][f'early_exaggeration_{early_exag}']):.4f}")
    print('*'*100)
    print()

    # save to json
    with open(output_path, 'w') as f:
        json.dump(save_dict, f, indent=4)


geodesic_dist_experiment(n_iter=10, dataset='circles')
geodesic_dist_experiment(n_iter=10, dataset='swiss_roll')
geodesic_dist_experiment(n_iter=10, dataset='torus')
geodesic_dist_experiment(n_iter=10, dataset='tree')