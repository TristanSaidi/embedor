from src.data.data import *
from src.orcml import *
from src.plotting import *
from src.utils.graph_utils import *
from src.utils.embeddings import *
import numpy as np
from src.utils.layout import *

class ORCFA(object):
    def __init__(
            self, 
            exp_params, 
            dim=2,
            verbose=False,
            uniform=False,
            seed=10
        ):

        """ 
        Initialize the ORCFA algorithm.
        Parameters
        ----------
        exp_params : dict
            The experimental parameters. Includes 'mode', 'n_neighbors' or 'epsilon'.
        dim : int, optional
            The dimensionality of the embedding (if any).
        """
        self.dim = dim
        self.exp_params = exp_params
        self.k = self.exp_params['n_neighbors']
        self.sigma = self.exp_params['sigma']
        if self.exp_params['mode'] == 'eps':
            raise NotImplementedError("ORCFA does not support epsilon neighborhoods.")
        self.verbose = verbose
        self.seed = seed
        self.uniform = uniform
        self.X = None

    def fit_transform(self, X=None):
        self.X = X
        print("Building nearest neighbor graph...")
        self._build_nnG() # self.G, self.orcs, self.A are now available
        print("Computing energies and affinities...")
        self._compute_energies()
        self._compute_affinities()
        self._update_G() # add edge attribute 'affinity'
        print("Running force-directed layout...")
        self._force_directed_layout()
        return self.embedding

    def _update_G(self):
        self.affinities = []
        self.energies = []
        self.affinities_unsigned = []
        node_indices = list(self.G.nodes())
        for i, (u,v) in enumerate(self.G.edges):
            idx_u = u
            idx_v = v
            self.G[u][v]['signed_affinity'] = self.edge_affinities[idx_u, idx_v]
            self.G[u][v]['unsigned_affinity'] = self.edge_affinities_unsigned[idx_u, idx_v]
            self.affinities.append(self.edge_affinities[idx_u, idx_v])
            self.affinities_unsigned.append(self.edge_affinities_unsigned[idx_u, idx_v])
            self.energies.append(self.edge_energy[idx_u, idx_v])

    def _build_nnG(self):
        """
        Build the nearest neighbor graph and compute ORC for each edge.
        """
        if self.X is None:
            raise ValueError("Data must be provided to build the nearest neighbor graph.")
        return_dict = get_nn_graph(self.X, self.exp_params)
        G = return_dict['G']
        # compute ORC
        return_dict = compute_orc(G, nbrhood_size=1) # compute ORC using 1-hop neighborhood
        self.G = return_dict['G']
        self.orcs = return_dict['orcs']
        self.A = nx.to_numpy_array(self.G, weight='weight', nodelist=list(range(len(self.G.nodes()))))
        self.edge_mask = np.where(self.A > 0, 1, 0)


    def _compute_energies(self, max_energy=np.inf):
        # compute energy for each edge
        energies = []        
        for u, v in self.G.edges():
            orc = self.G[u][v]['ricciCurvature']
            energy = min(
                -np.log(orc + 2) + np.log(3) + 1, # energy(+1) = 0, energy(-2) = infty,
                max_energy
            )
            self.G[u][v]['energy'] = energy
            energies.append(energy)
        
        self.A_energy = nx.to_numpy_array(self.G, weight='energy', nodelist=list(range(len(self.G.nodes()))))

        assert np.all(np.where(self.A_energy > 0, 1, 0) == self.edge_mask), "invalid entries"
        assert np.all(self.A_energy >= 0), "invalid entries"

        self.apsp_energy = scipy.sparse.csgraph.shortest_path(self.A_energy, unweighted=False, directed=False)
        assert np.all(self.apsp_energy * self.edge_mask <= self.A_energy)
        self.apsp_euc = scipy.sparse.csgraph.shortest_path(self.A, unweighted=True, directed=False)

        # if disconnected, set inf to 1e10
        self.apsp_energy[np.isinf(self.apsp_energy)] = 1e10
        
        assert np.allclose(self.apsp_energy, self.apsp_energy.T), "APSP matrix must be symmetric."
        assert np.allclose(self.apsp_euc, self.apsp_euc.T), "APSP matrix must be symmetric."

    def _compute_affinities(self):
        # compute affinities
        def energy_to_affinity(energy):
            affinity = 2 * (np.exp(-((energy-1)/self.sigma)**2) - 0.5)
            return affinity
        
        # compute affinities [without masking]
        self.all_energies = self.apsp_energy.copy()
        self.all_affinities = energy_to_affinity(self.all_energies)
        self.all_affinities_unsigned = 0.5 * (self.all_affinities + 1) 
        self.all_repulsions_unsigned = 1 - self.all_affinities_unsigned

        # compute affinities [with masking]
        self.edge_energy = self.edge_mask * self.apsp_energy
        self.edge_affinities = energy_to_affinity(self.edge_energy)
        self.edge_affinities_unsigned = 0.5 * (self.edge_affinities + 1) 
        self.edge_repulsions_unsigned = 1 - self.edge_affinities_unsigned

        # compute isomap affinities [without masking]
        self.all_euc = self.apsp_euc.copy()
        self.all_euc_affinities = energy_to_affinity(self.all_euc)
        self.all_euc_affinities_unsigned = 0.5 * (self.all_euc_affinities + 1)
        self.all_euc_repulsions_unsigned = 1 - self.all_euc_affinities_unsigned
        np.fill_diagonal(self.all_euc_affinities, 0)
        np.fill_diagonal(self.all_euc_affinities_unsigned, 0)

        self.edge_affinities *= self.edge_mask
        self.edge_affinities_unsigned *= self.edge_mask
        # set diagonal to 0
        np.fill_diagonal(self.edge_affinities, 0)
        np.fill_diagonal(self.edge_affinities_unsigned, 0)
        np.fill_diagonal(self.edge_repulsions_unsigned, 0)
        np.fill_diagonal(self.all_affinities, 0)
        np.fill_diagonal(self.all_affinities_unsigned, 0)
        np.fill_diagonal(self.all_repulsions_unsigned, 0)

        assert np.allclose(self.edge_affinities, self.edge_affinities.T), "Affinity matrix must be symmetric."
        assert np.allclose(self.edge_affinities_unsigned, self.edge_affinities_unsigned.T), "Unsigned affinity matrix must be symmetric."
        assert np.allclose(self.all_affinities, self.all_affinities.T), "Affinity matrix must be symmetric."
        assert np.allclose(self.all_affinities_unsigned, self.all_affinities_unsigned.T), "Unsigned affinity matrix must be symmetric."
        assert np.allclose(self.edge_repulsions_unsigned, self.edge_repulsions_unsigned.T), "Repulsion matrix must be symmetric."
        assert np.allclose(self.all_repulsions_unsigned, self.all_repulsions_unsigned.T), "Repulsion matrix must be symmetric."
        assert np.allclose(self.all_euc_affinities_unsigned, self.all_euc_affinities_unsigned.T), "Unsigned affinity matrix must be symmetric."
        assert np.allclose(self.all_euc_repulsions_unsigned, self.all_euc_repulsions_unsigned.T), "Repulsion matrix must be symmetric."

    def _force_directed_layout(self):
        # spectral initialization
        self.spectral_init = nx.spectral_layout(self.G, weight="unsigned_affinity", dim=self.dim, scale=1)
        self.embedding = np.array([self.spectral_init[node] for node in range(len(self.G.nodes()))])
        
        # convert to array
        from sklearn.utils import check_random_state
        # We add a little noise to avoid local minima for optimization to come
        self.embedding = noisy_scale_coords(
            self.embedding, check_random_state(self.seed), max_coord=10, noise=0.0001
        )

        # how many epochs to SKIP for each sample
        self.epochs_per_sample = make_epochs_per_sample(np.array(self.affinities_unsigned), n_epochs=500)

        from src.utils.layout import make_epochs_per_pair
        affinities = self.all_affinities_unsigned
        repulsions = self.all_repulsions_unsigned

        self.epochs_per_pair_positive = make_epochs_per_pair(affinities, n_epochs=500)
        self.epochs_per_pair_negative = make_epochs_per_pair(repulsions, n_epochs=500)

        self.embedding = (
            10.0
            * (self.embedding - np.min(self.embedding, 0))
            / (np.max(self.embedding, 0) - np.min(self.embedding, 0))
        ).astype(np.float32, order="C")

        # number of pairs
        npairs = (self.X.shape[0]**2 - self.X.shape[0])/2
        Z = np.sum(affinities)/2
        self.gamma = ((npairs - Z) / Z) * (self.k / self.X.shape[0]**2)

        self.embedding = optimize_layout_euclidean(
            self.embedding, 
            n_epochs=100,
            epochs_per_positive_sample=self.epochs_per_pair_positive,
            epochs_per_negative_sample=self.epochs_per_pair_negative,
            gamma=self.gamma,
            initial_alpha=0.25,
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


def noisy_scale_coords(coords, random_state, max_coord=10.0, noise=0.0001):
    expansion = max_coord / np.abs(coords).max()
    coords = (coords * expansion).astype(np.float32)
    return coords + random_state.normal(scale=noise, size=coords.shape).astype(
        np.float32
    )