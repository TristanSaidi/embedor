from src.data.data import *
from src.plotting import *
from src.utils.graph_utils import *
from src.utils.embeddings import *
import numpy as np
from src.utils.layout import *
from sklearn.manifold import TSNE, SpectralEmbedding


class EmbedOR(object):
    def __init__(
            self, 
            exp_params = {}, 
            dim=2,
            verbose=False,
            fast=False,
            seed=10,
            k_scale=15,
            metric='orc'
        ):

        """ 
        Initialize the EmbedOR algorithm.
        Parameters
        ----------
        exp_params : dict
            The experimental parameters. Includes 'mode', 'n_neighbors' or 'epsilon'.
        dim : int, optional
            The dimensionality of the embedding (if any).
        """
        self.dim = dim
        self.exp_params = exp_params
        self.k = self.exp_params.get('n_neighbors', 15)
        self.alpha = self.exp_params.get('alpha', 3)
        self.weighted = self.exp_params.get('weighted', True)
        self.k_scale = k_scale
        self.metric = metric
        self.exp_params = {
            'mode': 'nbrs',
            'n_neighbors': self.k,
            'alpha': self.alpha,
        }
        self.verbose = verbose
        self.fast = fast
        assert not (self.fast and self.metric == 'euclidean'), "Fast mode is not compatible with euclidean metric."
        self.seed = seed
        self.X = None

    def fit_transform(self, X=None):
        self.fit(X)
        self._init_embedding()
        print("Running Stochastic Neighbor Embedding...")
        self._layout(
            affinities=self.all_affinities,
            repulsions=self.all_repulsions
        )
        return self.embedding

    def fit_transform_tsne(self, X=None, init="random"):
        self.fit(X)
        self._init_embedding()
        self.tsne = TSNE(
            n_components=self.dim,
            random_state=self.seed,
            verbose=self.verbose,
            n_iter=300,
            metric='precomputed',
            init=self.embedding if init != "random" else "random",
        )
        self.embedding = self.tsne.fit_transform(self.all_energies)
        return self.embedding

    def fit(self, X=None):
        self.X = X
        print("Building nearest neighbor graph...")
        self._build_nnG() # self.G, self.orcs, self.A are now available
        print("Computing energies...")
        self._compute_energies()
        self._compute_affinities()
        self._update_G() # add edge attribute 'affinity'


    def _update_G(self):
        self.affinities = []
        self.energies = []
        for i, (u,v) in enumerate(self.G.edges):
            idx_u = u
            idx_v = v
            self.G[u][v]['affinity'] = self.all_affinities[idx_u, idx_v]
            self.affinities.append(self.all_affinities[idx_u, idx_v])
            self.energies.append(self.all_energies[idx_u, idx_v])

    def _build_nnG(self):
        """
        Build the nearest neighbor graph and compute ORC for each edge.
        """
        if self.X is None:
            raise ValueError("Data must be provided to build the nearest neighbor graph.")
        # compute diameter
        from sklearn.metrics import pairwise_distances
        self.diameter = np.max(pairwise_distances(self.X))
        # compute nearest neighbor graph
        return_dict = get_nn_graph(self.X, self.exp_params)
        G = return_dict['G']
        # compute ORC
        return_dict = compute_orc(G, nbrhood_size=1) # compute ORC using 1-hop neighborhood
        self.G = return_dict['G']
        self.orcs = return_dict['orcs']
        self.A = nx.to_numpy_array(self.G, weight='weight', nodelist=list(range(len(self.G.nodes()))))
        self.edge_mask = np.where(self.A > 0, 1, 0)


    def _compute_energies(self, max_val=np.inf):
        # compute energy for each edge
        energies = []
        max_energy = 0        
        for u, v in self.G.edges():
            orc = self.G[u][v]['ricciCurvature']
            
            c = 1/(np.log(3) - np.log(2))
            energy = (-c*np.log(orc + 2) + c*np.log(2) + 1) ** self.alpha + 1 # energy(+1) = 0, energy(-2) = infty,
            max_energy = max(energy, max_energy)
            energy = np.clip(energy, 0, max_val) # clip energy to max
            if self.weighted:
                energy = energy * self.G[u][v]['weight'] # scale energy by weight
            self.G[u][v]['energy'] = energy
            energies.append(energy)
        
        print(f"Max energy (unclipped): {max_energy}")
        self.A_energy = nx.to_numpy_array(self.G, weight='energy', nodelist=list(range(len(self.G.nodes()))))
        assert np.allclose(self.A_energy, self.A_energy.T), "Energy matrix must be symmetric."
        
        assert np.all(np.where(self.A_energy > 0, 1, 0) == self.edge_mask), "invalid entries"
        assert np.all(self.A_energy >= 0), "invalid entries"

        self.apsp_energy = scipy.sparse.csgraph.shortest_path(self.A_energy, unweighted=False, directed=False)
        assert np.allclose(self.apsp_energy, self.apsp_energy.T), "APSP matrix must be symmetric."
        max_val = np.max(self.apsp_energy)
        # print(f"Max APSP energy: {np.max(self.apsp_energy)}")
        if not self.fast:
            self.apsp_euclidean = scipy.sparse.csgraph.shortest_path(self.A, unweighted=False, directed=False)
            assert np.allclose(self.apsp_euclidean, self.apsp_euclidean.T), "APSP matrix must be symmetric."

    def _compute_affinities(self):
        self.all_energies = self.apsp_energy.copy()
        from scipy.spatial.distance import squareform
        assert np.allclose(self.apsp_energy, self.apsp_energy.T), "APSP matrix must be symmetric."
        
        if self.metric == "orc":
            self.all_affinities = squareform(joint_probabilities(self.apsp_energy, desired_perplexity=self.k_scale*self.k, verbose=5))
        else:
            self.all_affinities = squareform(joint_probabilities(self.apsp_euclidean, desired_perplexity=self.k_scale*self.k, verbose=5))
        self.all_repulsions = 1 - self.all_affinities
        # fill diagonal with 0
        np.fill_diagonal(self.all_affinities, 0)
        np.fill_diagonal(self.all_repulsions, 0)

    def _init_embedding(self):
        # spectral initialization
        # self.spectral_init = nx.spectral_layout(self.G, weight="affinity", dim=self.dim, scale=1)
        self.A_affinity = nx.to_numpy_array(self.G, weight='affinity', nodelist=list(range(len(self.G.nodes()))))
        self.spectral_init = SpectralEmbedding(
            n_components=self.dim,
            affinity='precomputed',
            random_state=self.seed,
        ).fit_transform(self.A_affinity)

        self.embedding = self.spectral_init.copy()
        # scale the embedding to [-0.5, 0.5] x [-0.5, 0.5]
        self.embedding = (self.embedding - np.min(self.embedding, axis=0)) / (
            np.max(self.embedding, axis=0) - np.min(self.embedding, axis=0)
        ) * 1 - 0.5
        self.spectral_init = self.embedding.copy()

    def _layout(self, affinities, repulsions):

        n_epochs = 300
        # how many epochs to SKIP for each sample
        self.epochs_per_pair_positive = make_epochs_per_pair(affinities, n_epochs=n_epochs)
        self.epochs_per_pair_negative = make_epochs_per_pair(repulsions, n_epochs=n_epochs)
        # compute gamma
        N = self.X.shape[0]
        npairs = (N**2 -N)/2
        Z = (np.sum(affinities) - np.trace(affinities))/2
        self.gamma = (npairs - Z)/(Z*npairs*50)
        print(f"Gamma: {self.gamma}")

        self.embedding = optimize_layout_euclidean(
            self.embedding, 
            n_epochs=n_epochs,
            epochs_per_positive_sample=self.epochs_per_pair_positive,
            epochs_per_negative_sample=self.epochs_per_pair_negative,
            gamma=self.gamma,
            initial_alpha=0.25,
            verbose=self.verbose,
        )

    def plot_energies(self):
        plt.figure()
        plt.hist(self.energies, bins=100)
        plt.title("Energy Distribution")
        plt.xlabel("Energy")
        plt.ylabel("Count")
        plt.show()

    def plot_affinities(self):
        plt.figure()
        plt.hist(self.affinities, bins=100)
        plt.title("Affinity Distribution")
        plt.xlabel("Affinity")
        plt.ylabel("Count")
        plt.show()

    def plot_spectral_init(self):
        spectral_init = np.array([self.spectral_init[node] for node in self.G.nodes()])
        emb = np.array([self.embedding[node] for node in self.G.nodes()])
        plt.scatter(spectral_init[:, 0], spectral_init[:, 1], c='r', s=10)
        plt.scatter(emb[:, 0], emb[:, 1], c='b', s=10)
        plt.legend(["Spectral Init", "Final Embedding"])

    def plot_apsp_energy(self):
        plt.figure()
        plt.hist(self.apsp_energy.flatten(), bins=100)
        plt.title("APSP Energy Distribution")
        plt.xlabel("APSP Energy")
        plt.ylabel("Count")
        plt.show()