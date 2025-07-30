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
np.random.seed(42)

output_dir = "../outputs/geodesic_distance"
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
    
    spearman_corrs_embedor = []
    spearman_corrs_umap = []
    spearman_corrs_umap_orcml = []
    spearman_corrs_tsne = []
    spearman_corrs_tsne_orcml = []
    spearman_corrs_phate = []
    spearman_corrs_isomap = []
    spearman_corrs_spectral = []
    spearman_corrs_embedor_euc = []

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
        embedor = EmbedOR()
        embedding = embedor.fit_transform(data_dict['data'])

        # compute ORCManL
        orcml = ORCManL()
        orcml.fit(data_dict['data'])

        G_pruned_nk = nk.nxadapter.nx2nk(orcml.G_pruned)
        apsp = nk.distance.APSP(G_pruned_nk).run().getDistances()
        indices = list(orcml.G_pruned.nodes())
        inverse_indices = [indices.index(i) for i in range(len(indices))]
        apsp = np.array(apsp)
        apsp = apsp[inverse_indices, :][:, inverse_indices]  # reorder to match original indices
        # clamp to 1e10
        apsp[apsp > 1e10] = 1e10


        umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(data_dict['data'])
        umap_orcml_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='precomputed').fit_transform(apsp)
        tsne_emb = TSNE(n_components=2, perplexity=30).fit_transform(data_dict['data'])
        tsne_orcml_emb = TSNE(n_components=2, perplexity=30, metric='precomputed', init='random').fit_transform(apsp)
        phate_emb = phate.PHATE().fit_transform(data_dict['data'])
        isomap_emb = Isomap().fit_transform(data_dict['data'])
        spectral_emb = SpectralEmbedding().fit_transform(data_dict['data'])
        embedor_euc = EmbedOR(edge_weight='euclidean')
        embedding_euc = embedor_euc.fit_transform(data_dict['data'])
        # compute pairwise distances and calculate correlation
        pdist_embedor = squareform(pdist(embedding, metric='euclidean'))
        pdist_umap = squareform(pdist(umap_emb, metric='euclidean'))
        pdist_umap_orcml = squareform(pdist(umap_orcml_emb, metric='euclidean'))
        pdist_tsne = squareform(pdist(tsne_emb, metric='euclidean'))
        pdist_tsne_orcml = squareform(pdist(tsne_orcml_emb, metric='euclidean'))
        pdist_phate = squareform(pdist(phate_emb, metric='euclidean'))
        pdist_isomap = squareform(pdist(isomap_emb, metric='euclidean'))
        pdist_spectral = squareform(pdist(spectral_emb, metric='euclidean'))
        pdist_embedor_euc = squareform(pdist(embedding_euc, metric='euclidean'))
        # compute pearson and spearman correlation
        spearman_corr_embedor, _ = spearmanr(pdist_embedor.flatten(), gt_geodesic_distance.flatten())
        spearman_corr_umap, _ = spearmanr(pdist_umap.flatten(), gt_geodesic_distance.flatten())
        spearman_corr_umap_orcml, _ = spearmanr(pdist_umap_orcml.flatten(), gt_geodesic_distance.flatten())
        spearman_corr_tsne, _ = spearmanr(pdist_tsne.flatten(), gt_geodesic_distance.flatten())
        spearman_corr_tsne_orcml, _ = spearmanr(pdist_tsne_orcml.flatten(), gt_geodesic_distance.flatten())
        spearman_corr_phate, _ = spearmanr(pdist_phate.flatten(), gt_geodesic_distance.flatten())
        spearman_corr_isomap, _ = spearmanr(pdist_isomap.flatten(), gt_geodesic_distance.flatten())
        spearman_corr_spectral, _ = spearmanr(pdist_spectral.flatten(), gt_geodesic_distance.flatten())
        spearman_corr_embedor_euc, _ = spearmanr(pdist_embedor_euc.flatten(), gt_geodesic_distance.flatten())
        spearman_corrs_embedor.append(spearman_corr_embedor)
        spearman_corrs_umap.append(spearman_corr_umap)
        spearman_corrs_umap_orcml.append(spearman_corr_umap_orcml)
        spearman_corrs_tsne.append(spearman_corr_tsne)
        spearman_corrs_tsne_orcml.append(spearman_corr_tsne_orcml)
        spearman_corrs_phate.append(spearman_corr_phate)
        spearman_corrs_isomap.append(spearman_corr_isomap)
        spearman_corrs_spectral.append(spearman_corr_spectral)
        spearman_corrs_embedor_euc.append(spearman_corr_embedor_euc)
    # compute mean and std of the correlations
    spearman_corr_embedor = np.mean(spearman_corrs_embedor)
    spearman_corr_umap = np.mean(spearman_corrs_umap)
    spearman_corr_umap_orcml = np.mean(spearman_corrs_umap_orcml)
    spearman_corr_tsne = np.mean(spearman_corrs_tsne)
    spearman_corr_tsne_orcml = np.mean(spearman_corrs_tsne_orcml)
    spearman_corr_phate = np.mean(spearman_corrs_phate)
    spearman_corr_isomap = np.mean(spearman_corrs_isomap)
    spearman_corr_spectral = np.mean(spearman_corrs_spectral)
    spearman_corr_embedor_euc = np.mean(spearman_corrs_embedor_euc)
    # compute std of the correlations
    spearman_corr_embedor_std = np.std(spearman_corrs_embedor)
    spearman_corr_umap_std = np.std(spearman_corrs_umap)
    spearman_corr_umap_orcml_std = np.std(spearman_corrs_umap_orcml)
    spearman_corr_tsne_std = np.std(spearman_corrs_tsne)
    spearman_corr_tsne_orcml_std = np.std(spearman_corrs_tsne_orcml)
    spearman_corr_phate_std = np.std(spearman_corrs_phate)
    spearman_corr_isomap_std = np.std(spearman_corrs_isomap)
    spearman_corr_spectral_std = np.std(spearman_corrs_spectral)
    spearman_corr_embedor_euc_std = np.std(spearman_corrs_embedor_euc)
    print('*'*100)
    print(f"Spearman correlation (EmbedOR): {spearman_corr_embedor:.4f} ± {spearman_corr_embedor_std:.4f}")
    print(f"Spearman correlation (UMAP): {spearman_corr_umap:.4f} ± {spearman_corr_umap_std:.4f}")
    print(f"Spearman correlation (UMAP + ORCManL): {spearman_corr_umap_orcml:.4f} ± {spearman_corr_umap_orcml_std:.4f}")
    print(f"Spearman correlation (t-SNE): {spearman_corr_tsne:.4f} ± {spearman_corr_tsne_std:.4f}")
    print(f"Spearman correlation (t-SNE + ORCManL): {spearman_corr_tsne_orcml:.4f} ± {spearman_corr_tsne_orcml_std:.4f}")
    print(f"Spearman correlation (PHATE): {spearman_corr_phate:.4f} ± {spearman_corr_phate_std:.4f}")
    print(f"Spearman correlation (Isomap): {spearman_corr_isomap:.4f} ± {spearman_corr_isomap_std:.4f}")
    print(f"Spearman correlation (Spectral Embedding): {spearman_corr_spectral:.4f} ± {spearman_corr_spectral_std:.4f}")
    print(f"Spearman correlation (EmbedOR Euclidean): {spearman_corr_embedor_euc:.4f} ± {spearman_corr_embedor_euc_std:.4f}")
    print('*'*100)
    print()

    save_dict = {
        'embedor_spearman_corr_mean': spearman_corr_embedor,
        'embedor_spearman_corr_std': spearman_corr_embedor_std,
        'umap_spearman_corr_mean': spearman_corr_umap,
        'umap_spearman_corr_std': spearman_corr_umap_std,
        'umap_orcml_spearman_corr_mean': spearman_corr_umap_orcml,
        'umap_orcml_spearman_corr_std': spearman_corr_umap_orcml_std,
        'tsne_spearman_corr_mean': spearman_corr_tsne,
        'tsne_spearman_corr_std': spearman_corr_tsne_std,
        'tsne_orcml_spearman_corr_mean': spearman_corr_tsne_orcml,
        'tsne_orcml_spearman_corr_std': spearman_corr_tsne_orcml_std,
        'phate_spearman_corr_mean': spearman_corr_phate,
        'phate_spearman_corr_std': spearman_corr_phate_std,
        'isomap_spearman_corr_mean': spearman_corr_isomap,
        'isomap_spearman_corr_std': spearman_corr_isomap_std,
        'spectral_spearman_corr_mean': spearman_corr_spectral,
        'spectral_spearman_corr_std': spearman_corr_spectral_std,
        'embedor_euc_spearman_corr_mean': spearman_corr_embedor_euc,
        'embedor_euc_spearman_corr_std': spearman_corr_embedor_euc
    }

    # save to json
    with open(output_path, 'w') as f:
        json.dump(save_dict, f, indent=4)


geodesic_dist_experiment(n_iter=10, dataset='circles')
geodesic_dist_experiment(n_iter=10, dataset='swiss_roll')
geodesic_dist_experiment(n_iter=10, dataset='torus')
geodesic_dist_experiment(n_iter=10, dataset='tree')