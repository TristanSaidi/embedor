from src.data.data import *
from src.plotting import *
from src.utils.graph_utils import *
# from src.isorc import *
from src.embedor import *
from sklearn.manifold import TSNE
import umap
import numpy as np
from sklearn.manifold import TSNE
import phate

save_path = '/home/tristan/Research/Sp25/embedor/outputs/tSNE_failure_cases'
os.makedirs(save_path, exist_ok=True)

np.random.seed(0)

# moons
# visualization for paper
return_dict = moons(n_points=10000, noise=0.00, noise_thresh=None, sep=0.2, width=0.3, dim=3)
X = return_dict['data']
plt.figure()
ax = plt.axes(projection='3d')
ax.set_proj_type('ortho')
ax.scatter3D(X[:, 0], X[:, 2], X[:, 1], c=return_dict['cluster'], cmap=plt.cm.berlin, s=1)
# get rid of the grid and axes
ax.grid(False)
ax.set_axis_off()
ax.view_init(elev=20, azim=55)
plt.savefig(os.path.join(save_path, 'moons.pdf'), dpi=1200)

return_dict = moons(n_points=5000, noise=0.04, noise_thresh=None, sep=0.2, width=0.3, dim=3)
X = return_dict['data']
plt.figure()
ax = plt.axes(projection='3d')
ax.set_proj_type('ortho')
ax.scatter3D(X[:, 0], X[:, 2], X[:, 1], c=return_dict['cluster'], cmap=plt.cm.berlin, s=1)
# get rid of the grid and axes
ax.grid(False)
ax.set_axis_off()
ax.view_init(elev=20, azim=55)

# run embedor
embedor = EmbedOR()
X_emb = embedor.fit_transform(X)
plt.figure(figsize=(10, 10))
plot_graph_2D(X_emb, embedor.G, node_color=return_dict['cluster'][embedor.G.nodes()], edge_width=0, title=None, cmap=plt.cm.berlin)
plt.savefig(os.path.join(save_path, 'embedor_moons.pdf'), dpi=1200)

# run tSNE
tsne = TSNE(n_components=2, random_state=0, init='random', verbose=10, perplexity=150, method='exact')
X_tsne = tsne.fit_transform(X)
plot_data_2D(X_tsne, color=return_dict['cluster'], cmap=plt.cm.berlin)
plt.savefig(os.path.join(save_path, 'tsne_moons.pdf'), dpi=1200)

# tsne with delta metric
tsne_delta_metric = TSNE(n_components=2, random_state=0, init='random', verbose=10, metric='precomputed', perplexity=150, method='exact')
X_tsne = tsne.fit_transform(embedor.apsp)
plot_graph_2D(X_tsne, embedor.G, node_color=return_dict['cluster'][embedor.G.nodes()], edge_width=0, title=None, cmap=plt.cm.berlin)
plt.savefig(os.path.join(save_path, 'tsne_delta_moons.pdf'), dpi=1200)

# double swiss roll
from src.data import *
return_dict = swiss_roll(n_points=2500, noise=0.05, noise_thresh=None, double=True)
X = return_dict['data']

from src.data import *
return_dict = swiss_roll(n_points=2500, noise=0.05, noise_thresh=None, double=True)
X = return_dict['data']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_proj_type('ortho')
ax.scatter3D(X[:, 1], X[:, 2], X[:, 0], c=return_dict['cluster'], cmap=plt.cm.berlin, s=1)
# get rid of the grid and axes
ax.grid(False)
ax.set_axis_off()
ax.view_init(elev=0, azim=7.5)
plt.savefig(os.path.join(save_path, 'double_swiss_roll.pdf'), dpi=1200)

# run embedor
embedor = EmbedOR()
X_emb = embedor.fit_transform(X)
plt.figure(figsize=(10, 10))
plot_graph_2D(X_emb, embedor.G, node_color=return_dict['cluster'][embedor.G.nodes()], edge_width=0, title=None, cmap=plt.cm.berlin)
plt.savefig(os.path.join(save_path, 'embedor_double_swiss_roll.pdf'), dpi=1200)

# run tSNE
tsne = TSNE(n_components=2, random_state=0, init='random', verbose=10, perplexity=150, method='exact')
X_tsne = tsne.fit_transform(X)
plot_data_2D(X_tsne, color=return_dict['cluster'], cmap=plt.cm.berlin)
plt.savefig(os.path.join(save_path, 'tsne_double_swiss_roll.pdf'), dpi=1200)

# tSNE with delta metric
tsne_delta_metric = TSNE(n_components=2, random_state=0, init='random', verbose=10, metric='precomputed', perplexity=150, method='exact')
X_tsne = tsne.fit_transform(embedor.apsp)
plot_graph_2D(X_tsne, embedor.G, node_color=return_dict['cluster'][embedor.G.nodes()], edge_width=0, title=None, cmap=plt.cm.berlin)
plt.savefig(os.path.join(save_path, 'tsne_delta_double_swiss_roll.pdf'), dpi=1200)