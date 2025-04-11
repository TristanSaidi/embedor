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
    'p': 3
}

def synthetic_data(n_points):
    save_path = '/home/tristan/Research/Fa24/isorc/outputs/synthetic_data'
    os.makedirs(save_path, exist_ok=True)
    
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.join(save_path, dt_string), exist_ok=False)

    circles_path = os.path.join(save_path, dt_string, 'circles')
    os.makedirs(circles_path, exist_ok=False)
    # concentric circles
    noise = 0.1
    noise_thresh = None
    return_dict = concentric_circles(n_points=n_points, factor=0.4, noise=noise, noise_thresh=noise_thresh)
    plot_data_2D(return_dict['data'], color=None, node_size=1.5)
    plt.savefig(os.path.join(save_path, 'circles.png'), dpi=1200)

    print("Running EmbedOR...")
    embedor = EmbedOR(exp_params)
    embedding = embedor.fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(embedding, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(circles_path, 'embedor.png'), dpi=1200)
    plt.close()

    print("Running EmbedOR with isomap metric...")
    embedor = EmbedOR(exp_params, metric='euclidean')
    embedding = embedor.fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(embedding, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(circles_path, 'embedor_isomap.png'), dpi=1200)
    plt.close()

    print("Running UMAP...")
    umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(umap_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(circles_path, 'umap.png'), dpi=1200)
    plt.close()

    print("Running t-SNE...")
    tsne_emb = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(tsne_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(circles_path, 'tsne.png'), dpi=1200)
    plt.close()

    print("Running PHATE...")
    phate_emb = phate.PHATE(n_jobs=-2).fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(phate_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(circles_path, 'phate.png'), dpi=1200)
    plt.close()

    print("Running spectral embedding...")
    spectral_emb = SpectralEmbedding(n_components=2, affinity='rbf').fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(spectral_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(circles_path, 'spectral.png'), dpi=1200)
    plt.close()

    print("Running Isomap...")
    isomap_emb = Isomap(n_neighbors=15, n_components=2).fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(isomap_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(circles_path, 'isomap.png'), dpi=1200)
    plt.close()

    # swiss roll
    swiss_roll_path = os.path.join(save_path, dt_string, 'swiss_roll')
    os.makedirs(swiss_roll_path, exist_ok=False)

    noise = 0
    noise_thresh = None
    return_dict = swiss_roll(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=False)

    # apply a rotation by 45 degrees about the z-axis and a rotation by 45 degrees about the x-axis
    zrot = 0
    xrot = np.pi/32
    Rz = np.array([[np.cos(zrot), -np.sin(zrot), 0],
                [np.sin(zrot), np.cos(zrot), 0],
                [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                [0, np.cos(xrot), -np.sin(xrot)],
                [0, np.sin(xrot), np.cos(xrot)]])
    R = np.dot(Rz, Rx)
    X = return_dict['data']
    Y = np.dot(X, R.T)
    plot_data_2D(Y[:, [0,2]], color=None, node_size=1.5)
    plt.savefig(os.path.join(save_path, 'swiss_roll.png'), dpi=1200)
    plt.close()

    noise = 1
    noise_thresh = None
    return_dict = swiss_roll(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=False)
    
    print("Running EmbedOR...")
    embedor = EmbedOR(exp_params)
    embedding = embedor.fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(embedding, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(swiss_roll_path, 'embedor.png'), dpi=1200)
    plt.close()

    print("Running EmbedOR with isomap metric...")
    embedor = EmbedOR(exp_params, metric='euclidean')
    embedding = embedor.fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(embedding, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(swiss_roll_path, 'embedor_isomap.png'), dpi=1200)
    plt.close()

    print("Running UMAP...")
    umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(umap_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(swiss_roll_path, 'umap.png'), dpi=1200)
    plt.close()

    print("Running t-SNE...")
    tsne_emb = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(tsne_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(swiss_roll_path, 'tsne.png'), dpi=1200)
    plt.close()

    print("Running PHATE...")
    phate_emb = phate.PHATE(n_jobs=-2).fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(phate_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(swiss_roll_path, 'phate.png'), dpi=1200)
    plt.close()
    
    print("Running spectral embedding...")
    spectral_emb = SpectralEmbedding(n_components=2, affinity='rbf').fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(spectral_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(swiss_roll_path, 'spectral.png'), dpi=1200)
    plt.close()

    print("Running Isomap...")
    isomap_emb = Isomap(n_neighbors=15, n_components=2).fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(isomap_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(swiss_roll_path, 'isomap.png'), dpi=1200)
    plt.close()

    # tori
    tori_path = os.path.join(save_path, dt_string, 'tori')
    os.makedirs(tori_path, exist_ok=False)
    
    noise = 0.5
    noise_thresh = None

    print("Running torus...")
    return_dict = torus(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=False, double=True)
    plot_data_2D(return_dict['data'], color=None, title=None)
    plt.savefig(os.path.join(save_path, 'torus.png'), dpi=1200)
    plt.close()

    print("Running EmbedOR...")
    embedor = EmbedOR(exp_params)
    embedding = embedor.fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(embedding, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(tori_path, 'embedor.png'), dpi=1200)
    plt.close()

    print("Running EmbedOR with isomap metric...")
    embedor = EmbedOR(exp_params, metric='euclidean')
    embedding = embedor.fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(embedding, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(tori_path, 'embedor_isomap.png'), dpi=1200)
    plt.close()

    print("Running UMAP...")
    umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(umap_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(tori_path, 'umap.png'), dpi=1200)
    plt.close()

    print("Running t-SNE...")
    tsne_emb = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(tsne_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(tori_path, 'tsne.png'), dpi=1200)
    plt.close()

    print("Running PHATE...")
    phate_emb = phate.PHATE(n_jobs=-2).fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(phate_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(tori_path, 'phate.png'), dpi=1200)
    plt.close()

    print("Running spectral embedding...")
    spectral_emb = SpectralEmbedding(n_components=2, affinity='rbf').fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(spectral_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(tori_path, 'spectral.png'), dpi=1200)
    plt.close()

    print("Running Isomap...")
    isomap_emb = Isomap(n_neighbors=15, n_components=2).fit_transform(return_dict['data'])
    plt.figure(figsize=(10, 10))
    plot_data_2D(isomap_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(tori_path, 'isomap.png'), dpi=1200)
    plt.close()

    # tree
    tree_path = os.path.join(save_path, dt_string, 'tree')
    os.makedirs(tree_path, exist_ok=False)
    X, _ = gen_dla(n_dim=100, n_branch=8, sigma=4, branch_length=500)

    print("Running tree...")
    embedor = EmbedOR(exp_params)
    embedding = embedor.fit_transform(X)
    plt.figure(figsize=(10, 10))
    plot_data_2D(embedding, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(tree_path, 'embedor.png'), dpi=1200)
    plt.close()

    print("Running EmbedOR with isomap metric...")
    embedor = EmbedOR(exp_params, metric='euclidean')
    embedding = embedor.fit_transform(X)
    plt.figure(figsize=(10, 10))
    plot_data_2D(embedding, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(tree_path, 'embedor_isomap.png'), dpi=1200)
    plt.close()

    print("Running UMAP...")
    umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(X)
    plt.figure(figsize=(10, 10))
    plot_data_2D(umap_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(tree_path, 'umap.png'), dpi=1200)
    plt.close()

    print("Running t-SNE...")
    tsne_emb = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(X)
    plt.figure(figsize=(10, 10))
    plot_data_2D(tsne_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(tree_path, 'tsne.png'), dpi=1200)
    plt.close()

    print("Running PHATE...")
    phate_emb = phate.PHATE(n_jobs=-2).fit_transform(X)
    plt.figure(figsize=(10, 10))
    plot_data_2D(phate_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(tree_path, 'phate.png'), dpi=1200)
    plt.close()

    print("Running spectral embedding...")
    spectral_emb = SpectralEmbedding(n_components=2, affinity='rbf').fit_transform(X)
    plt.figure(figsize=(10, 10))
    plot_data_2D(spectral_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(tree_path, 'spectral.png'), dpi=1200)
    plt.close()

    print("Running Isomap...")
    isomap_emb = Isomap(n_neighbors=15, n_components=2).fit_transform(X)
    plt.figure(figsize=(10, 10))
    plot_data_2D(isomap_emb, color=None, title=None, node_size=1.5)
    plt.savefig(os.path.join(tree_path, 'isomap.png'), dpi=1200)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the metric.")
    parser.add_argument("--n_points", type=int, default=5000, help="Number of points to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()
    seed = args.seed    
    np.random.seed(seed)
    synthetic_data(args.n_points)