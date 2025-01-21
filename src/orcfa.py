from src.data.data import *
from src.orcml import *
from src.plotting import *
from src.utils.graph_utils import *
from src.utils.embeddings import *
import numpy as np
from src.utils.layout import spring_layout, forceatlas2_layout

class ORCFA(object):
    def __init__(
            self, 
            exp_params, 
            dim=2,
            verbose=False,
            uniform=False,
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
        # dim must be int if init != 'ambient'
        self.dim = dim
        self.exp_params = exp_params
        self.k = self.exp_params['n_neighbors']
        self.sigma = self.exp_params['sigma']
        if self.exp_params['mode'] == 'eps':
            raise NotImplementedError("ORCFA does not support epsilon neighborhoods.")
        self.verbose = verbose
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
        node_indices = list(self.G.nodes())
        for i, (u,v) in enumerate(self.G.edges):
            idx_u = node_indices.index(u)
            idx_v = node_indices.index(v)
            assert self.edge_mask[idx_u, idx_v] == 1, "index error"
            self.G[u][v]['signed_affinity'] = self.edge_affinities[idx_u, idx_v]
            self.G[u][v]['unsigned_affinity'] = self.edge_affinities_unsigned[idx_u, idx_v]
            self.affinities.append(self.edge_affinities[idx_u, idx_v])
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
        self.A = nx.to_numpy_array(self.G, weight='weight')
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
        
        self.A_energy = nx.to_numpy_array(self.G, weight='energy')

        assert np.all(np.where(self.A_energy > 0, 1, 0) == self.edge_mask), "invalid entries"
        assert np.all(self.A_energy >= 0), "invalid entries"

        self.apsp_energy = scipy.sparse.csgraph.shortest_path(self.A_energy, unweighted=False, directed=False)
        assert np.all(self.apsp_energy * self.edge_mask <= self.A_energy)
        # if disconnected, set inf to 1e10
        self.apsp_energy[np.isinf(self.apsp_energy)] = 1e10
        
        assert np.allclose(self.apsp_energy, self.apsp_energy.T), "APSP matrix must be symmetric."

    def _compute_affinities(self):
        # compute affinities
        self.edge_energy = self.edge_mask * self.apsp_energy
        def energy_to_affinity(energy):
            affinity = 2 * (np.exp(-((energy-1)/self.sigma)**2) - 0.5)
            return affinity
        self.edge_affinities = energy_to_affinity(self.edge_energy)
        self.edge_affinities_unsigned = self.edge_affinities + 1 # min affinity = -1

        self.edge_affinities *= self.edge_mask
        self.edge_affinities_unsigned *= self.edge_mask
        # set diagonal to 0
        np.fill_diagonal(self.edge_affinities, 0)
        np.fill_diagonal(self.edge_affinities_unsigned, 0)

        assert np.allclose(self.edge_affinities, self.edge_affinities.T), "Affinity matrix must be symmetric."
        assert np.allclose(self.edge_affinities_unsigned, self.edge_affinities_unsigned.T), "Unsigned affinity matrix must be symmetric."

    def _force_directed_layout(self, method='forceatlas2', init='random'):
        # spectral initialization
        if init == 'spectral':
            self.spectral_init = nx.spectral_layout(self.G, weight="unsigned_affinity", dim=self.dim, scale=1)
        elif init == 'random':
            self.spectral_init = nx.random_layout(self.G, dim=self.dim)
        # convert to dict
        if method == 'forceatlas2':
            self.embedding = forceatlas2_layout(
                self.G, 
                pos=self.spectral_init, 
                weight='signed_affinity', 
                dim=self.dim,
                max_iter=100
            )

        elif method=='spring':
            self.embedding = spring_layout(
                self.G, 
                scale=1,
                k=1e-8,
                pos=self.spectral_init, 
                weight='signed_affinity', 
                dim=self.dim,
                iterations=50
            )

        elif method == 'kamada_kawai':
            # apsp energy indices misaligned
            G_indices = list(self.G.nodes())
            inverse_indices = [G_indices.index(i) for i in range(len(G_indices))]
            self.embedding = nx.kamada_kawai_layout(
                self.G, 
                pos=self.spectral_init,
                dist=self.apsp_energy[inverse_indices][:, inverse_indices],
                weight=None, # weight has no effect when dist is provided 
                dim=self.dim
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