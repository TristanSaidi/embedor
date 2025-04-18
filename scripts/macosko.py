from src.data.data import *
from src.embedor import *
from src.plotting import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import argparse
import umap
import numpy as np
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
import phate
import json
from src.utils import *

sns.set_theme()
# diffusion distance

exp_params = {
    'p': 3
}


def developmental(n_points):
    save_path = '/burg/iicd/users/tls2160/research/Fa24/isorc/outputs/macosko'
    os.makedirs(save_path, exist_ok=True)

    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.join(save_path, dt_string), exist_ok=True)

    macosko_path = os.path.join(save_path, dt_string, 'macosko')
    os.makedirs(macosko_path, exist_ok=True)

    data, labels = get_macosko_data(n_points=n_points)

    stats_dict = {}

    embedor = EmbedOR(exp_params)
    embedding = embedor.fit_transform(data)
    embedor_euc = EmbedOR(exp_params, metric='euclidean')
    embedding_euc = embedor_euc.fit_transform(data)
    umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(data)
    tsne_emb = TSNE(n_components=2, perplexity=30, n_iter=300, init='random').fit_transform(data)
    phate_emb = phate.PHATE(n_jobs=-2).fit_transform(data)
    spectral_emb = SpectralEmbedding(n_components=2).fit_transform(data)
    iso_emb = Isomap(n_neighbors=15, n_components=2).fit_transform(data)

    # plot with 33% lowest energy edges
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

    # plot with 2% highest energy edges
    bottom_indices = indices[-len(embedor.G.edges()) // 50:]
    desired_edges = [pair for i, pair in enumerate(embedor.G.edges()) if i in bottom_indices]
    # create a new graph with the desired edges
    high_energy_graph = embedor.G.copy()
    # remove all edges from the graph
    high_energy_graph.remove_edges_from(high_energy_graph.edges())
    # add the desired edges to the graph
    high_energy_graph.add_edges_from(desired_edges)

    # plot with thickness dependent on affinity computed from energy
    affinities = embedor.affinities
    max_thickness = 0.5
    edge_widths = np.array(affinities)**1.5 * (max_thickness / (np.max(affinities))**1.5)
    
    # compute z-scores for low energy edges
    z_scores_mean, z_scores_std = low_energy_edge_stats(embedding, embedor.G, low_energy_graph)
    z_scores_mean_euc, z_scores_std_euc = low_energy_edge_stats(embedding_euc, embedor_euc.G, low_energy_graph)
    z_scores_mean_umap, z_scores_std_umap = low_energy_edge_stats(umap_emb, embedor.G, low_energy_graph)
    z_scores_mean_tsne, z_scores_std_tsne = low_energy_edge_stats(tsne_emb, embedor.G, low_energy_graph)
    z_scores_mean_phate, z_scores_std_phate = low_energy_edge_stats(phate_emb, embedor.G, low_energy_graph)
    z_scores_mean_spectral, z_scores_std_spectral = low_energy_edge_stats(spectral_emb, embedor.G, low_energy_graph)
    z_scores_mean_iso, z_scores_std_iso = low_energy_edge_stats(iso_emb, embedor.G, low_energy_graph)

    stats_dict['eb'] = {
        'embedor': {
            'z_scores_mean': z_scores_mean,
            'z_scores_std': z_scores_std
        },
        'embedor_euc': {
            'z_scores_mean': z_scores_mean_euc,
            'z_scores_std': z_scores_std_euc
        },
        'umap': {
            'z_scores_mean': z_scores_mean_umap,
            'z_scores_std': z_scores_std_umap
        },
        'tsne': {
            'z_scores_mean': z_scores_mean_tsne,
            'z_scores_std': z_scores_std_tsne
        },
        'phate': {
            'z_scores_mean': z_scores_mean_phate,
            'z_scores_std': z_scores_std_phate
        },
        'spectral': {
            'z_scores_mean': z_scores_mean_spectral,
            'z_scores_std': z_scores_std_spectral
        },
        'iso': {
            'z_scores_mean': z_scores_mean_iso,
            'z_scores_std': z_scores_std_iso
        }
    }

    # save figures
    embedor_path = os.path.join(macosko_path, 'embedor')
    os.makedirs(embedor_path, exist_ok=False)
    plt.figure(figsize=(10, 10))
    plot_graph_2D(embedding, embedor.G, node_color=labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    plt.savefig(os.path.join(embedor_path, 'class_annot.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(embedding, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    plt.savefig(os.path.join(embedor_path, 'low_energy_graph.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(embedding, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    plt.savefig(os.path.join(embedor_path, 'high_energy_graph.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(embedding, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    plt.savefig(os.path.join(embedor_path, 'variable_edge_widths.png'))
    plt.close()

    embedor_euc_path = os.path.join(macosko_path, 'embedor_euc')
    os.makedirs(embedor_euc_path, exist_ok=False)
    plt.figure(figsize=(10, 10))
    plot_graph_2D(embedding_euc, embedor_euc.G, node_color=labels[embedor_euc.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    plt.savefig(os.path.join(embedor_euc_path, 'class_annot.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(embedding_euc, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    plt.savefig(os.path.join(embedor_euc_path, 'low_energy_graph.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(embedding_euc, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    plt.savefig(os.path.join(embedor_euc_path, 'high_energy_graph.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(embedding_euc, embedor_euc.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    plt.savefig(os.path.join(embedor_euc_path, 'variable_edge_widths.png'))
    plt.close()
    
    umap_path = os.path.join(macosko_path, 'umap')
    os.makedirs(umap_path, exist_ok=False)
    plt.figure(figsize=(10, 10))
    plot_graph_2D(umap_emb, embedor.G, node_color=labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    plt.savefig(os.path.join(umap_path, 'class_annot.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(umap_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    plt.savefig(os.path.join(umap_path, 'low_energy_graph.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(umap_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    plt.savefig(os.path.join(umap_path, 'high_energy_graph.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(umap_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    plt.savefig(os.path.join(umap_path, 'variable_edge_widths.png'))
    plt.close()
    
    tsne_path = os.path.join(macosko_path, 'tsne')
    os.makedirs(tsne_path, exist_ok=False)
    plt.figure(figsize=(10, 10))
    plot_graph_2D(tsne_emb, embedor.G, node_color=labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    plt.savefig(os.path.join(tsne_path, 'class_annot.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(tsne_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    plt.savefig(os.path.join(tsne_path, 'low_energy_graph.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(tsne_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    plt.savefig(os.path.join(tsne_path, 'high_energy_graph.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(tsne_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    plt.savefig(os.path.join(tsne_path, 'variable_edge_widths.png'))
    plt.close()

    phate_path = os.path.join(macosko_path, 'phate')
    os.makedirs(phate_path, exist_ok=False)
    plt.figure(figsize=(10, 10))
    plot_graph_2D(phate_emb, embedor.G, node_color=labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    plt.savefig(os.path.join(phate_path, 'class_annot.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(phate_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    plt.savefig(os.path.join(phate_path, 'low_energy_graph.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(phate_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    plt.savefig(os.path.join(phate_path, 'high_energy_graph.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(phate_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    plt.savefig(os.path.join(phate_path, 'variable_edge_widths.png'))
    plt.close()
    
    spectral_path = os.path.join(macosko_path, 'spectral')
    os.makedirs(spectral_path, exist_ok=False)
    plt.figure(figsize=(10, 10))
    plot_graph_2D(spectral_emb, embedor.G, node_color=labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    plt.savefig(os.path.join(spectral_path, 'class_annot.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(spectral_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    plt.savefig(os.path.join(spectral_path, 'low_energy_graph.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(spectral_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    plt.savefig(os.path.join(spectral_path, 'high_energy_graph.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(spectral_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    plt.savefig(os.path.join(spectral_path, 'variable_edge_widths.png'))
    plt.close()
    
    iso_path = os.path.join(macosko_path, 'iso')
    os.makedirs(iso_path, exist_ok=False)
    plt.figure(figsize=(10, 10))
    plot_graph_2D(iso_emb, embedor.G, node_color=labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    plt.savefig(os.path.join(iso_path, 'class_annot.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(iso_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    plt.savefig(os.path.join(iso_path, 'low_energy_graph.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(iso_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    plt.savefig(os.path.join(iso_path, 'high_energy_graph.png'))
    plt.close()
    plt.figure(figsize=(10, 10))
    plot_graph_2D(iso_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    plt.savefig(os.path.join(iso_path, 'variable_edge_widths.png'))
    plt.close()


    # save stats
    stats_path = os.path.join(save_path, dt_string, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"Stats saved to {stats_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EmbedOR on macosko dataset.")
    parser.add_argument("--n_points", type=int, default=5000, help="Number of points to use.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()
    seed = args.seed    
    np.random.seed(seed)
    developmental(args.n_points)