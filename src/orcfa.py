from src.data.data import *
from src.orcml import *
from src.plotting import *
from src.utils.graph_utils import *
from src.utils.embeddings import *
import numpy as np

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

    def fit_transform(self, X):
        self.X = X
        self._build_nnG() # self.G, self.orcs, self.A are now available
        self._compute_energies()
        self._compute_affinities()
        self._update_G() # add edge attribute 'affinity'
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
            self.G[u][v]['affinity'] = self.edge_affinities[idx_u, idx_v]
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


    def _compute_energies(self, max_energy=1e10):
        # compute energy for each edge
        energies = []        
        for u, v in self.G.edges():
            orc = self.G[u][v]['ricciCurvature']
            energy = min(
                (-np.log(orc + 2) + np.log(3)), # energy(+1) = 0, energy(-2) = infty,
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
            affinity = 2 * (np.exp(-(energy/self.sigma)**2) - 0.5)
            return affinity
        self.edge_affinities = energy_to_affinity(self.edge_energy)
        self.edge_affinities *= self.edge_mask
        # set diagonal to 0
        np.fill_diagonal(self.edge_affinities, 0)
        assert np.allclose(self.edge_affinities, self.edge_affinities.T), "Affinity matrix must be symmetric."

    def _force_directed_layout(self):
        # spectral initialization
        spectral_init = nx.spectral_layout(self.G, weight="weight")
        node_mass = {node: 1 for node in self.G.nodes()}
        self.embedding = nx.forceatlas2_layout(
            self.G, 
            pos=spectral_init, 
            node_mass=node_mass,
            weight='affinity', 
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