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



def benchmark_datasets(n_points):
    save_path = '/burg/iicd/users/tls2160/research/Fa24/isorc/outputs/parameter_ablations'
    os.makedirs(save_path, exist_ok=True)

    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.join(save_path, dt_string), exist_ok=False)

    mnist_data, mnist_labels = get_mnist_data(n_samples=n_points, label=None)

    repulsion_path = os.path.join(save_path, dt_string, 'repulsion')
    os.makedirs(repulsion_path, exist_ok=False)

    ps = [0, 3, 6, 9]
    exp_params = {}
    for p in ps:
        print(f'\nRunning with p = {p}')
        exp_params['p'] = p
        embedor = EmbedOR(exp_params)
        embedding = embedor.fit_transform(mnist_data)
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

        # save figures
        embedor_path = os.path.join(repulsion_path, f'repulsion_{p}')
        os.makedirs(embedor_path, exist_ok=False)
        plt.figure(figsize=(10, 10))
        plot_graph_2D(embedding, embedor.G, node_color=mnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
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

    perplexity_path = os.path.join(save_path, dt_string, 'perplexity')
    os.makedirs(perplexity_path, exist_ok=False)
    perplexities = [50, 100, 150, 200]
    exp_params = {}

    for perplexity in perplexities:
        print(f'\nRunning with perplexity = {perplexity}')
        exp_params['perplexity'] = perplexity
        embedor = EmbedOR(exp_params)
        embedding = embedor.fit_transform(mnist_data)
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

        # save figures
        embedor_path = os.path.join(perplexity_path, f'perplexity_{perplexity}')
        os.makedirs(embedor_path, exist_ok=False)
        plt.figure(figsize=(10, 10))
        plot_graph_2D(embedding, embedor.G, node_color=mnist_labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the metric.")
    parser.add_argument("--n_points", type=int, default=5000, help="Number of points to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()
    seed = args.seed    
    np.random.seed(seed)
    benchmark_datasets(args.n_points)