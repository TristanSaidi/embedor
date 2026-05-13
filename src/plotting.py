import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os

# plotting functions

def plot_data_2D(X, color, title=None, node_size=10, axes=False, exp_name=None, filename=None, cmap=plt.cm.viridis):
    """
    Plot the data with the points colored by class membership.
    Parameters
    
    X : array-like, shape (n_samples, 2)
        The coordinates of the points.
    y : array-like, shape (n_samples,)
        The integer labels for class membership of each point.
    title : str
        The title of the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=color, cmap=cmap, s=node_size)
    plt.title(title)
    plt.gca().set_aspect('equal')
    if not axes:
        plt.gca().set_axis_off()
    if filename is not None and exp_name is not None:
        os.makedirs('figures', exist_ok=True)
        exp_dir = os.path.join('figures', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, filename)
        plt.savefig(path)



def plot_graph_2D(X, graph, title=None, node_color='#1f78b4', edge_color='lightgray', node_size=1, edge_width=1.0, colorbar=False, exp_name=None, filename=None, cmap=plt.cm.Spectral):
    """
    Plot the graph with the desired node or edge coloring.
    Parameters
    
    X : array-like, shape (n_samples, 2)
        The coordinates of the nodes.
    graph : networkx.Graph
        The graph to plot.
    title : str
        The title of the plot.
    node_color : str
        The color of the nodes.
    edge_color : str
        The color of the edges.
    """
    if type(edge_color) == str:
        edge_cmap = plt.cm.viridis
    else:
        edge_cmap = plt.cm.coolwarm
    plt.figure(figsize=(6,6), dpi=200)
    if type(edge_color) != str:
        mean, std = np.mean(edge_color), np.std(edge_color)
        edge_vmin = mean - 2*std
        edge_vmax = mean + 2*std
        # edge_vmin, edge_vmax = np.min(edge_color), np.max(edge_color)
    else:
        edge_vmin, edge_vmax = -1, 1
    nx.draw(graph, X, node_color=node_color, edge_color=edge_color, node_size=node_size, cmap=cmap, edge_cmap=edge_cmap, edge_vmin=edge_vmin, edge_vmax=edge_vmax, width=edge_width)
    plt.title(title)
    plt.gca().set_aspect('equal')
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-1, vmax=1))
        sm._A = []
        plt.colorbar(sm)
    if filename is not None and exp_name is not None:
        os.makedirs('figures', exist_ok=True)
        exp_dir = os.path.join('figures', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, filename)
        plt.savefig(path)
