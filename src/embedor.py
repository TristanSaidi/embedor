from src.data.data import *
from src.plotting import *
from src.utils.graph_utils import *
from src.utils.embeddings import *
import numpy as np
from src.utils.layout import *


class EmbedOR(object):
    def __init__(
            self, 
            exp_params = {}, 
            dim=2,
            verbose=False,
            seed=10
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
        self.sigma = self.exp_params.get('sigma', 30)
        self.alpha = self.exp_params.get('alpha', 4)
        self.exp_params = {
            'mode': 'nbrs',
            'n_neighbors': self.k,
            'sigma': self.sigma,
            'alpha': self.alpha,
        }

        self.verbose = verbose
        self.seed = seed
        self.X = None

    def fit_transform(self, X=None):
        self.fit(X)
        print("Running Stochastic Neighbor Embedding...")
        self._layout(
            affinities=self.all_affinities,
            repulsions=self.all_repulsions
        )
        return self.embedding

    def fit(self, X=None):
        self.X = X
        print("Building nearest neighbor graph...")
        self._build_nnG() # self.G, self.orcs, self.A are now available
        print("Computing energies and affinities...")
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


    def _compute_energies(self):
        # compute energy for each edge
        energies = []        
        for u, v in self.G.edges():
            orc = self.G[u][v]['ricciCurvature']
            
            c = 1/(np.log(3) - np.log(2))
            energy = (-c*np.log(orc + 2) + c*np.log(2) + 1) ** self.alpha + 1 # energy(+1) = 0, energy(-2) = infty,
            energy = energy * self.G[u][v]['weight'] # scale energy by weight
            self.G[u][v]['energy'] = energy
            energies.append(energy)
        
        self.A_energy = nx.to_numpy_array(self.G, weight='energy', nodelist=list(range(len(self.G.nodes()))))
        assert np.allclose(self.A_energy, self.A_energy.T), "Energy matrix must be symmetric."
        
        assert np.all(np.where(self.A_energy > 0, 1, 0) == self.edge_mask), "invalid entries"
        assert np.all(self.A_energy >= 0), "invalid entries"

        self.apsp_energy = scipy.sparse.csgraph.shortest_path(self.A_energy, unweighted=False, directed=False)
        assert np.allclose(self.apsp_energy, self.apsp_energy.T), "APSP matrix must be symmetric."
        self.apsp_energy /= self.diameter
        print(f"Max APSP energy: {np.max(self.apsp_energy)}")

    def _compute_affinities(self):
        # compute affinities
        def energy_to_affinity(energy):
            affinity = np.exp(-self.sigma * (energy)**2)
            return affinity
        
        # compute affinities [without masking]
        self.all_energies = self.apsp_energy.copy()
        self.all_affinities = energy_to_affinity(self.all_energies)
        self.all_repulsions = 1 - self.all_affinities
        # set diagonal to 0, 1 respectively
        np.fill_diagonal(self.all_affinities, 1)
        np.fill_diagonal(self.all_repulsions, 0)
        # sanity checks
        assert np.allclose(self.all_affinities, self.all_affinities.T), "Affinity matrix must be symmetric."
        assert np.allclose(self.all_repulsions, self.all_repulsions.T), "Repulsion matrix must be symmetric."

    def _layout(self, affinities, repulsions):
        # spectral initialization
        self.spectral_init = nx.spectral_layout(self.G, weight="affinity", dim=self.dim, scale=1)
        self.embedding = np.array([self.spectral_init[node] for node in range(len(self.G.nodes()))])
        
        # convert to array
        from sklearn.utils import check_random_state
        # We add a little noise to avoid local minima for optimization to come
        self.embedding = noisy_scale_coords(
            self.embedding, check_random_state(self.seed), max_coord=10, noise=0.0001
        )

        # how many epochs to SKIP for each sample
        self.epochs_per_pair_positive = make_epochs_per_pair(affinities, n_epochs=500)
        self.epochs_per_pair_negative = make_epochs_per_pair(repulsions, n_epochs=500)

        self.embedding = (
            1.0
            * (self.embedding - np.min(self.embedding, 0))
            / (np.max(self.embedding, 0) - np.min(self.embedding, 0))
        ).astype(np.float32, order="C")

        # number of pairs
        npairs = (self.X.shape[0]**2 - self.X.shape[0])/2
        Z = np.sum(affinities)/2
        self.gamma = ((npairs - Z) / Z) * (self.k / self.X.shape[0]**2)
        print(f"Computed gamma: {self.gamma}")

        self.embedding = optimize_layout_euclidean(
            self.embedding, 
            n_epochs=100,
            epochs_per_positive_sample=self.epochs_per_pair_positive,
            epochs_per_negative_sample=self.epochs_per_pair_negative,
            gamma=self.gamma,
            initial_alpha=1,
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