from src.data.data import *
from src.embedor import *
from src.plotting import *
from src.utils.graph_utils import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import argparse
import umap
import numpy as np
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
import phate
import pickle
import json
from src.utils import *
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

sns.set_theme()
# diffusion distance

exp_params = {
    'p': 3
}

def compute_correlation(emb, time):
    # compute pairwise distances
    emb_dist = pdist(emb)
    assert len(emb_dist.shape) == 1, "Distance matrix shape mismatch"
    time_dist = np.abs(time - time[:, np.newaxis])
    time_dist = squareform(time_dist)
    assert len(time_dist.shape) == 1, "Time distance matrix shape mismatch"
    
    # compute spearman correlation
    spearman_corr, _ = spearmanr(emb_dist, time_dist)
    # compute pearson correlation
    pearson_corr, _ = pearsonr(emb_dist, time_dist)
    return spearman_corr, pearson_corr


def umap_tsne_param_abl(n_points, dataset):
    assert dataset in ['mnist', 'fmnist', 'developmental', 'macosko', 'chimp'], "Dataset not available"
    print(f"Running UMAP+tSNE parameter ablation with {n_points} points.")
    save_path = f'/burg/iicd/users/tls2160/research/Fa24/isorc/outputs/umap_tsne_abl'
    os.makedirs(save_path, exist_ok=True)

    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.join(save_path, dt_string), exist_ok=True)

    if dataset == 'mnist':
        X, y = get_mnist_data(n_samples=n_points, label=None)
    elif dataset == 'fmnist':
        X, y = get_fmnist_data(n_samples=n_points, label=None)
    elif dataset == 'macosko':
        X, y = get_macosko_data(n_points=n_points)
    elif dataset == 'chimp':
        X, y = get_chimp_data(n_points=n_points)
    elif dataset == 'developmental':
        developmental_data, days = get_developmental_data(n_points=n_points)
        developmental_data = np.asarray(developmental_data.todense())
        # find nan indices in time and drop them from all embeddings
        mask = ~np.isnan(days)
        if not np.all(mask):
            print(f"Found NaN entries in time, dropping them from all embeddings.")
            days = days[mask]
            developmental_data = developmental_data[mask]
        X = developmental_data
        y = days
    

    embedor = EmbedOR(exp_params)
    embedding = embedor.fit_transform(X)
    
    # extracting graph with 33% lowest energy edges
    edge_energies = embedor.distances
    # sort the edges by energy
    indices = np.argsort(edge_energies)
    # get the top 100 edges
    top_indices = indices[:len(embedor.G.edges()) // 3]
    desired_edges = [pair for i, pair in enumerate(embedor.G.edges()) if i in top_indices]
    # create a new graph with the desired edges
    low_energy_graph = embedor.G.copy()
    # remove all edges from the graph
    low_energy_graph.remove_edges_from(low_energy_graph.edges())
    # add the desired edges to the graph
    low_energy_graph.add_edges_from(desired_edges)

    exp_dict = {"min_dist":{}, "nsr":{}, "perplexity": {}, "ee":{}, "embedor":{}, "data": X, "labels": y}
    exp_dict["embedor"]["embedding"] = embedding

    # compute spearman and pearson correlation for the EmbedOR embedding
    if dataset == 'developmental':
        spearman_corr, pearson_corr = compute_correlation(embedding, y)
        exp_dict["embedor"]["spearman_corr"] = spearman_corr
        exp_dict["embedor"]["pearson_corr"] = pearson_corr
        print(f"EmbedOR Spearman: {spearman_corr}, Pearson: {pearson_corr}")

    _, _, z_scores = low_energy_edge_stats(embedding, embedor.G, low_energy_graph)
    exp_dict["embedor"]["z_scores_low_energy"] = z_scores
    _, _, bottom_z_scores = low_distance_edge_stats(embedding, embedor.G, embedor.apsp)
    exp_dict["embedor"]["z_scores_low_distance"] = bottom_z_scores
    
    print(f"EmbedOR low energy edge z-scores: {np.mean(z_scores)}, {np.std(z_scores)}")
    print(f"EmbedOR low distance edge z-scores: {np.mean(bottom_z_scores)}, {np.std(bottom_z_scores)}")
    print()

    print("Running UMAP parameter ablation... varying min_dist")
    # umap parameters to vary
    min_dists = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    for min_dist in min_dists:
        exp_dict["min_dist"][f"min_dist_{min_dist}"] = {}
        umap_emb = umap.UMAP(n_neighbors=15, min_dist=min_dist, metric='euclidean').fit_transform(X)
        if dataset == 'developmental':
            umap_emb = umap_emb[mask]  # apply the mask to the embedding
            spearman_corr, pearson_corr = compute_correlation(umap_emb, y)
            exp_dict["min_dist"][f"min_dist_{min_dist}"]["spearman_corr"] = spearman_corr
            exp_dict["min_dist"][f"min_dist_{min_dist}"]["pearson_corr"] = pearson_corr
            print(f"UMAP min_dist={min_dist} Spearman: {spearman_corr}, Pearson: {pearson_corr}")
        exp_dict["min_dist"][f"min_dist_{min_dist}"]["embedding"] = umap_emb
        _, _, z_scores = low_energy_edge_stats(umap_emb, embedor.G, low_energy_graph)
        exp_dict["min_dist"][f"min_dist_{min_dist}"]["z_scores_low_energy"] = z_scores
        _, _, bottom_z_scores = low_distance_edge_stats(umap_emb, embedor.G, embedor.apsp)
        exp_dict["min_dist"][f"min_dist_{min_dist}"]["z_scores_low_distance"] = bottom_z_scores
        print(f"UMAP min_dist={min_dist} low energy edge z-scores: {np.mean(z_scores)}, {np.std(z_scores)}")
        print(f"UMAP min_dist={min_dist} low distance edge z-scores: {np.mean(bottom_z_scores)}, {np.std(bottom_z_scores)}")
        print()

    negative_sample_rate = [1, 2, 3, 4, 5, 10]
    print("Running UMAP parameter ablation... varying negative sample rate")
    for nsr in negative_sample_rate:
        exp_dict["nsr"][f"nsr_{nsr}"] = {}
        umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', negative_sample_rate=nsr).fit_transform(X)
        if dataset == 'developmental':
            umap_emb = umap_emb[mask]
            spearman_corr, pearson_corr = compute_correlation(umap_emb, y)
            exp_dict["nsr"][f"nsr_{nsr}"]["spearman_corr"] = spearman_corr
            exp_dict["nsr"][f"nsr_{nsr}"]["pearson_corr"] = pearson_corr
            print(f"UMAP nsr={nsr} Spearman: {spearman_corr}, Pearson: {pearson_corr}")
        exp_dict["nsr"][f"nsr_{nsr}"]["embedding"] = umap_emb
        _, _, z_scores = low_energy_edge_stats(umap_emb, embedor.G, low_energy_graph)
        exp_dict["nsr"][f"nsr_{nsr}"]["z_scores_low_energy"] = z_scores
        _, _, bottom_z_scores = low_distance_edge_stats(umap_emb, embedor.G, embedor.apsp)
        exp_dict["nsr"][f"nsr_{nsr}"]["z_scores_low_distance"] = bottom_z_scores
        print(f"UMAP nsr={nsr} low energy edge z-scores: {np.mean(z_scores)}, {np.std(z_scores)}")
        print(f"UMAP nsr={nsr} low distance edge z-scores: {np.mean(bottom_z_scores)}, {np.std(bottom_z_scores)}")
        print()

    perplexity = [5, 10, 15, 30, 40, 50, 100]
    for p in perplexity:
        print(f'Running t-SNE with perplexity={p}')
        exp_dict["perplexity"][f"perplexity_{p}"] = {}
        tsne_emb = TSNE(n_components=2, perplexity=p, random_state=0).fit_transform(X)
        if dataset == 'developmental':
            tsne_emb = tsne_emb[mask]
            spearman_corr, pearson_corr = compute_correlation(tsne_emb, y)
            exp_dict["perplexity"][f"perplexity_{p}"]["spearman_corr"] = spearman_corr
            exp_dict["perplexity"][f"perplexity_{p}"]["pearson_corr"] = pearson_corr
            print(f"t-SNE perplexity={p} Spearman: {spearman_corr}, Pearson: {pearson_corr}")
        exp_dict["perplexity"][f"perplexity_{p}"]["embedding"] = tsne_emb
        _, _, z_scores = low_energy_edge_stats(tsne_emb, embedor.G, low_energy_graph)
        exp_dict["perplexity"][f"perplexity_{p}"]["z_scores_low_energy"] = z_scores
        _, _, bottom_z_scores = low_distance_edge_stats(tsne_emb, embedor.G, embedor.apsp)
        exp_dict["perplexity"][f"perplexity_{p}"]["z_scores_low_distance"] = bottom_z_scores
        print(f"t-SNE perplexity={p} low energy edge z-scores: {np.mean(z_scores)}, {np.std(z_scores)}")
        print(f"t-SNE perplexity={p} low distance edge z-scores: {np.mean(bottom_z_scores)}, {np.std(bottom_z_scores)}")
        print()

    early_exaggeration = [2, 4, 8, 12, 16, 24]
    for ee in early_exaggeration:
        print(f'Running t-SNE with early_exaggeration={ee}')
        exp_dict["ee"][f"ee_{ee}"] = {}
        tsne_emb = TSNE(n_components=2, perplexity=30, early_exaggeration=ee, random_state=42).fit_transform(X)
        if dataset == 'developmental':
            tsne_emb = tsne_emb[mask]
            spearman_corr, pearson_corr = compute_correlation(tsne_emb, y)
            exp_dict["ee"][f"ee_{ee}"] = {}
            exp_dict["ee"][f"ee_{ee}"]["spearman_corr"] = spearman_corr
            exp_dict["ee"][f"ee_{ee}"]["pearson_corr"] = pearson_corr
            print(f"t-SNE early_exaggeration={ee} Spearman: {spearman_corr}, Pearson: {pearson_corr}")
        exp_dict["ee"][f"ee_{ee}"]["embedding"] = tsne_emb
        _, _, z_scores = low_energy_edge_stats(tsne_emb, embedor.G, low_energy_graph)
        exp_dict["ee"][f"ee_{ee}"]["z_scores_low_energy"] = z_scores
        _, _, bottom_z_scores = low_distance_edge_stats(tsne_emb, embedor.G, embedor.apsp)
        exp_dict["ee"][f"ee_{ee}"]["z_scores_low_distance"] = bottom_z_scores
        print(f"t-SNE early_exaggeration={ee} low energy edge z-scores: {np.mean(z_scores)}, {np.std(z_scores)}")
        print(f"t-SNE early_exaggeration={ee} low distance edge z-scores: {np.mean(bottom_z_scores)}, {np.std(bottom_z_scores)}")
        print()

    # save the exp_dict
    with open(os.path.join(save_path, f'umap_tsne_param_ablation_{dataset}.pkl'), 'wb') as f:
        pickle.dump(exp_dict, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameter ablation for umap and tsne.")
    parser.add_argument("--n_points", type=int, default=5000, help="Number of points to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--dataset", type=str, default="mnist")
    args = parser.parse_args()
    seed = args.seed    
    np.random.seed(seed)
    umap_tsne_param_abl(args.n_points, args.dataset)