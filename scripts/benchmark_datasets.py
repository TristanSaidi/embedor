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

sns.set_theme()
# diffusion distance

exp_params = {
    'alpha': 3
}

def benchmark_datasets(n_points):
    save_path = '/home/tristan/Research/Fa24/isorc/outputs/benchmark_datasets'
    os.makedirs(save_path, exist_ok=True)
    
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.join(save_path, dt_string), exist_ok=False)

    # mnist_path = os.path.join(save_path, dt_string, 'mnist')
    # os.makedirs(mnist_path, exist_ok=False)

    # mnist_data, mnist_labels = get_mnist_data(n_samples=n_points, label=None)

    # embedor = EmbedOR(exp_params, fast=True)
    # embedding = embedor.fit_transform(mnist_data)
    # umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(mnist_data)
    # tsne_emb = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(mnist_data)
    # phate_emb = phate.PHATE(n_jobs=-2).fit_transform(mnist_data)
    # spectral_emb = SpectralEmbedding(n_components=2, affinity='rbf').fit_transform(mnist_data)
    # iso_emb = Isomap(n_neighbors=15, n_components=2).fit_transform(mnist_data)

    # # plot with 33% lowest energy edges
    # edge_energies = embedor.energies
    # # sort the edges by energy
    # indices = np.argsort(edge_energies)
    # # get the top 100 edges
    # top_indices = indices[:len(embedor.G.edges()) // 3]
    # desired_edges = [pair for i, pair in enumerate(embedor.G.edges()) if i in top_indices]
    # # create a new graph with the desired edges
    # low_energy_graph = embedor.G.copy()
    # # remove all edges from the graph
    # low_energy_graph.remove_edges_from(low_energy_graph.edges())
    # # add the desired edges to the graph
    # low_energy_graph.add_edges_from(desired_edges)

    # # plot with 2% highest energy edges
    # bottom_indices = indices[-len(embedor.G.edges()) // 50:]
    # desired_edges = [pair for i, pair in enumerate(embedor.G.edges()) if i in bottom_indices]
    # # create a new graph with the desired edges
    # high_energy_graph = embedor.G.copy()
    # # remove all edges from the graph
    # high_energy_graph.remove_edges_from(high_energy_graph.edges())
    # # add the desired edges to the graph
    # high_energy_graph.add_edges_from(desired_edges)

    # # plot with thickness dependent on affinity computed from energy
    # affinities = embedor.affinities
    # max_thickness = 0.5
    # edge_widths = np.array(affinities)**1.5 * (max_thickness / (np.max(affinities))**1.5)
    
    # # save figures
    # embedor_path = os.path.join(mnist_path, 'embedor')
    # os.makedirs(embedor_path, exist_ok=False)
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(embedding, embedor.G, node_color=mnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(embedor_path, 'class_annot.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(embedding, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(embedor_path, 'low_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(embedding, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(embedor_path, 'high_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(embedding, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(embedor_path, 'variable_edge_widths.png'))
    # plt.close()
    
    # umap_path = os.path.join(mnist_path, 'umap')
    # os.makedirs(umap_path, exist_ok=False)
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(umap_emb, embedor.G, node_color=mnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(umap_path, 'class_annot.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(umap_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(umap_path, 'low_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(umap_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(umap_path, 'high_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(umap_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(umap_path, 'variable_edge_widths.png'))
    # plt.close()
    
    # tsne_path = os.path.join(mnist_path, 'tsne')
    # os.makedirs(tsne_path, exist_ok=False)
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(tsne_emb, embedor.G, node_color=mnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(tsne_path, 'class_annot.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(tsne_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(tsne_path, 'low_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(tsne_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(tsne_path, 'high_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(tsne_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(tsne_path, 'variable_edge_widths.png'))
    # plt.close()

    # phate_path = os.path.join(mnist_path, 'phate')
    # os.makedirs(phate_path, exist_ok=False)
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(phate_emb, embedor.G, node_color=mnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(phate_path, 'class_annot.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(phate_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(phate_path, 'low_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(phate_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(phate_path, 'high_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(phate_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(phate_path, 'variable_edge_widths.png'))
    # plt.close()
    
    # spectral_path = os.path.join(mnist_path, 'spectral')
    # os.makedirs(spectral_path, exist_ok=False)
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(spectral_emb, embedor.G, node_color=mnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(spectral_path, 'class_annot.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(spectral_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(spectral_path, 'low_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(spectral_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(spectral_path, 'high_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(spectral_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(spectral_path, 'variable_edge_widths.png'))
    # plt.close()
    
    # iso_path = os.path.join(mnist_path, 'iso')
    # os.makedirs(iso_path, exist_ok=False)
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(iso_emb, embedor.G, node_color=mnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(iso_path, 'class_annot.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(iso_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(iso_path, 'low_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(iso_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(iso_path, 'high_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(iso_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(iso_path, 'variable_edge_widths.png'))
    # plt.close()

    # # fashion_mnist
    # fashion_mnist_path = os.path.join(save_path, dt_string, 'fashion_mnist')
    # os.makedirs(fashion_mnist_path, exist_ok=False)
    # fashion_mnist_data, fashion_mnist_labels = get_fmnist_data(n_samples=n_points, label=None)

    # embedor = EmbedOR(exp_params, fast=True)
    # embedding = embedor.fit_transform(fashion_mnist_data)
    # umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(fashion_mnist_data)
    # tsne_emb = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(fashion_mnist_data)
    # phate_emb = phate.PHATE(n_jobs=-2).fit_transform(fashion_mnist_data)
    # spectral_emb = SpectralEmbedding(n_components=2, affinity='rbf').fit_transform(fashion_mnist_data)
    # iso_emb = Isomap(n_neighbors=15, n_components=2).fit_transform(fashion_mnist_data)

    # # plot with 33% lowest energy edges
    # edge_energies = embedor.energies
    # # sort the edges by energy
    # indices = np.argsort(edge_energies)
    # # get the top 100 edges
    # top_indices = indices[:len(embedor.G.edges()) // 3]
    # desired_edges = [pair for i, pair in enumerate(embedor.G.edges()) if i in top_indices]
    # # create a new graph with the desired edges
    # low_energy_graph = embedor.G.copy()
    # # remove all edges from the graph
    # low_energy_graph.remove_edges_from(low_energy_graph.edges())
    # # add the desired edges to the graph
    # low_energy_graph.add_edges_from(desired_edges)

    # # plot with 2% highest energy edges
    # bottom_indices = indices[-len(embedor.G.edges()) // 50:]
    # desired_edges = [pair for i, pair in enumerate(embedor.G.edges()) if i in bottom_indices]
    # # create a new graph with the desired edges
    # high_energy_graph = embedor.G.copy()
    # # remove all edges from the graph
    # high_energy_graph.remove_edges_from(high_energy_graph.edges())
    # # add the desired edges to the graph
    # high_energy_graph.add_edges_from(desired_edges)

    # # plot with thickness dependent on affinity computed from energy
    # affinities = embedor.affinities
    # max_thickness = 0.5
    # edge_widths = np.array(affinities)**1.5 * (max_thickness / (np.max(affinities))**1.5)
    
    # # save figures

    # # save figures
    # embedor_path = os.path.join(fashion_mnist_path, 'embedor')
    # os.makedirs(embedor_path, exist_ok=False)
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(embedding, embedor.G, node_color=fashion_mnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(embedor_path, 'class_annot.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(embedding, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(embedor_path, 'low_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(embedding, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(embedor_path, 'high_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(embedding, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(embedor_path, 'variable_edge_widths.png'))
    # plt.close()
    
    # umap_path = os.path.join(fashion_mnist_path, 'umap')
    # os.makedirs(umap_path, exist_ok=False)
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(umap_emb, embedor.G, node_color=fashion_mnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(umap_path, 'class_annot.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(umap_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(umap_path, 'low_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(umap_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(umap_path, 'high_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(umap_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(umap_path, 'variable_edge_widths.png'))
    # plt.close()
    
    # tsne_path = os.path.join(fashion_mnist_path, 'tsne')
    # os.makedirs(tsne_path, exist_ok=False)
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(tsne_emb, embedor.G, node_color=fashion_mnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(tsne_path, 'class_annot.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(tsne_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(tsne_path, 'low_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(tsne_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(tsne_path, 'high_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(tsne_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(tsne_path, 'variable_edge_widths.png'))
    # plt.close()

    # phate_path = os.path.join(fashion_mnist_path, 'phate')
    # os.makedirs(phate_path, exist_ok=False)
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(phate_emb, embedor.G, node_color=fashion_mnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(phate_path, 'class_annot.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(phate_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(phate_path, 'low_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(phate_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(phate_path, 'high_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(phate_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(phate_path, 'variable_edge_widths.png'))
    # plt.close()
    
    # spectral_path = os.path.join(fashion_mnist_path, 'spectral')
    # os.makedirs(spectral_path, exist_ok=False)
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(spectral_emb, embedor.G, node_color=fashion_mnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(spectral_path, 'class_annot.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(spectral_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(spectral_path, 'low_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(spectral_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(spectral_path, 'high_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(spectral_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(spectral_path, 'variable_edge_widths.png'))
    # plt.close()
    
    # iso_path = os.path.join(fashion_mnist_path, 'iso')
    # os.makedirs(iso_path, exist_ok=False)
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(iso_emb, embedor.G, node_color=fashion_mnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(iso_path, 'class_annot.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(iso_emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(iso_path, 'low_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(iso_emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
    # plt.savefig(os.path.join(iso_path, 'high_energy_graph.png'))
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plot_graph_2D(iso_emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
    # plt.savefig(os.path.join(iso_path, 'variable_edge_widths.png'))
    # plt.close()

    # kmnist
    kmnist_path = os.path.join(save_path, dt_string, 'kmnist')
    os.makedirs(kmnist_path, exist_ok=False)
    kmnist_data, kmnist_labels = get_kmnist_data(n_samples=n_points, label=None)

    embedor = EmbedOR(exp_params, fast=True)
    embedding = embedor.fit_transform(kmnist_data)
    umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(kmnist_data)
    tsne_emb = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(kmnist_data)
    phate_emb = phate.PHATE(n_jobs=-2).fit_transform(kmnist_data)
    spectral_emb = SpectralEmbedding(n_components=2, affinity='rbf').fit_transform(kmnist_data)
    iso_emb = Isomap(n_neighbors=15, n_components=2).fit_transform(kmnist_data)

    # plot with 33% lowest energy edges
    edge_energies = embedor.energies
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
    
    # save figures

    # save figures
    embedor_path = os.path.join(kmnist_path, 'embedor')
    os.makedirs(embedor_path, exist_ok=False)
    plt.figure(figsize=(10, 10))
    plot_graph_2D(embedding, embedor.G, node_color=kmnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
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
    
    umap_path = os.path.join(kmnist_path, 'umap')
    os.makedirs(umap_path, exist_ok=False)
    plt.figure(figsize=(10, 10))
    plot_graph_2D(umap_emb, embedor.G, node_color=kmnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
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
    
    tsne_path = os.path.join(kmnist_path, 'tsne')
    os.makedirs(tsne_path, exist_ok=False)
    plt.figure(figsize=(10, 10))
    plot_graph_2D(tsne_emb, embedor.G, node_color=kmnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
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

    phate_path = os.path.join(kmnist_path, 'phate')
    os.makedirs(phate_path, exist_ok=False)
    plt.figure(figsize=(10, 10))
    plot_graph_2D(phate_emb, embedor.G, node_color=kmnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the metric.")
    parser.add_argument("--n_points", type=int, default=5000, help="Number of points to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()
    seed = args.seed    
    np.random.seed(seed)
    benchmark_datasets(args.n_points)