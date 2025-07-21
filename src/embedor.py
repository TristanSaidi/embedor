# check if matplotlib is already imported
import matplotlib.pyplot as plt
# # from src.data.data import *
from src.utils.graph_utils import *
# # from src.utils.embeddings import *
import numpy as np
from src.utils.layout import *
from umap.spectral import spectral_layout
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix

import scipy
import networkit as nk


ENERGY_PARAMS = {
    'k_max': 1,
    'k_min': -2,
    'k_crit': 0
}

class EmbedOR(object):
    def __init__(
            self, 
            exp_params = {}, 
            dim=2,
            verbose=False,
            seed=10,
            edge_weight='orc',
            subsample=False,
            subsample_factor=0.05
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
        self.p = self.exp_params.get('p', 3)
        self.epochs = self.exp_params.get('epochs', 300)
        self.weighted = self.exp_params.get('weighted', True)
        self.perplexity = self.exp_params.get('perplexity', 150)
        self.edge_weight = edge_weight
        self.exp_params = {
            'mode': 'nbrs',
            'n_neighbors': self.k,
            'p': self.p,
        }
        self.verbose = verbose
        self.seed = seed
        self.X = None
        self.fitted = False
        self.subsample = subsample
        self.subsample_factor = subsample_factor

    def fit_transform(self, X=None):
        if not self.fitted:
            self.fit(X)
        if self.subsample:
            print("Subsampling interactions...")
            self._subsample_interactions()
        print("Initializing embedding...")
        self._init_embedding()
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
        print("Computing distances...")
        self._compute_distances()
        print("Computing affinities...")
        self._compute_affinities()
        print("Updating the graph attributes...")
        self._update_G() # add edge attribute 'affinity'
        self.fitted = True


    def _update_G(self):
        self.affinities = []
        self.distances = []
        for i, (u,v) in enumerate(self.G.edges):
            idx_u = u
            idx_v = v
            self.G[u][v]['affinity'] = self.all_affinities[idx_u, idx_v]
            self.affinities.append(self.all_affinities[idx_u, idx_v])
            self.distances.append(self.apsp[idx_u, idx_v])

    def _build_nnG(self):
        """
        Build the nearest neighbor graph and compute ORC for each edge.
        """
        if self.X is None:
            raise ValueError("Data must be provided to build the nearest neighbor graph.")
        
        # compute nearest neighbor graph
        return_dict = get_nn_graph(self.X, self.exp_params)
        G = return_dict['G']
        
        # compute ORC
        return_dict = compute_orc(G, nbrhood_size=1) # compute ORC using 1-hop neighborhood
        self.orcs = return_dict['orcs']

        self.G = return_dict['G']
        self.A = nx.to_numpy_array(self.G, weight='weight', nodelist=list(range(len(self.G.nodes()))))
        # get knn indices
        A_ut = self.A * np.triu(np.ones(self.A.shape), k=1)
        self.knn_indices =  A_ut.nonzero()
        self.all_indices = np.triu(np.ones(self.A.shape), k=1).nonzero()
        self.all_indices = np.stack(self.all_indices, axis=0)
        del A_ut
        # convert A to sparse matrix
        self.A = csr_matrix(self.A)

    def _compute_distances(self, max_val=np.inf):
        # compute energy for each edge
        # time_start = time.time()

        if self.edge_weight != "euclidean":
            k_max = ENERGY_PARAMS['k_max']
            k_min = ENERGY_PARAMS['k_min']
            k_crit = ENERGY_PARAMS['k_crit']
            energies = []

            for idx, (u, v) in enumerate(self.G.edges()):
                orc = self.orcs[idx]
                c = 1/np.log((k_max-k_min)/(k_crit-k_min))                
                energy = (-c * np.log(orc - k_min) + c * np.log(k_crit - k_min) + 1) ** self.p + 1 # energy(k_max) = 1, energy(k_min) = infty, energy(k_crit) = 2                max_energy = max(energy, max_energy)
                energy = np.clip(energy, 0, max_val) # clip energy to max
                if self.weighted:
                    energy = energy * self.G[u][v]['weight'] # scale energy by weight (i.e. Euclidean distance)
                self.G[u][v]['energy'] = energy
                energies.append(energy)
            self.G_nk = nk.nxadapter.nx2nk(self.G, weightAttr='energy')                    

        else:
            self.G_nk = nk.nxadapter.nx2nk(self.G, weightAttr='weight')

        self.apsp = nk.distance.APSP(self.G_nk).run().getDistances()
        self.apsp = np.array(self.apsp)
        indices = list(self.G.nodes())
        inverse_indices = [indices.index(i) for i in range(len(indices))]
        self.apsp = self.apsp[inverse_indices, :][:, inverse_indices]
        assert np.allclose(self.apsp, self.apsp.T), "APSP matrix must be symedge_weight."

    def _compute_affinities(self):
        from scipy.spatial.distance import squareform     
        self.all_affinities = squareform(joint_probabilities(self.apsp, desired_perplexity=self.perplexity, verbose=0))

        # symmetrize affinities
        self.all_affinities = (self.all_affinities + self.all_affinities.T) / 2
        self.all_repulsions = 1 - self.all_affinities
        # fill diagonal with 0
        np.fill_diagonal(self.all_affinities, 0)
        np.fill_diagonal(self.all_repulsions, 0)


    def _init_embedding(self):
        # spectral initialization
        A_affinity_sparse = nx.to_scipy_sparse_array(self.G, weight='affinity', nodelist=list(range(len(self.G.nodes()))))
        self.spectral_init = spectral_layout(
            data=None,
            graph=A_affinity_sparse,
            dim=self.dim,
            random_state=self.seed,
        )

        self.embedding = self.spectral_init.copy()
        # scale the embedding to [-0.5, 0.5] x [-0.5, 0.5]
        self.embedding = (self.embedding - np.min(self.embedding, axis=0)) / (
            np.max(self.embedding, axis=0) - np.min(self.embedding, axis=0)
        ) * 1 - 0.5
        self.spectral_init = self.embedding.copy()

    def _layout(self, affinities, repulsions):
        if self.subsample:
            affinities = affinities[self.subsample_indices[0], self.subsample_indices[1]]
            repulsions = repulsions[self.subsample_indices[0], self.subsample_indices[1]]
            n_pairs = self.subsample_indices.shape[1]
            N = self.X.shape[0]
            Z = np.sum(affinities)
            self.gamma = (n_pairs - Z)/(Z*n_pairs)
        else:
            # compute gamma
            N = self.X.shape[0]
            npairs = (N**2 -N)/2
            Z = (np.sum(affinities) - np.trace(affinities))/2
            self.gamma = (npairs - Z)/(Z*N**2)
            self.subsample_indices = None
        # how many epochs to SKIP for each sample
        self.epochs_per_pair_positive = make_epochs_per_pair(affinities, n_epochs=self.epochs)
        self.epochs_per_pair_negative = make_epochs_per_pair(repulsions, n_epochs=self.epochs)
        
        self.embedding = optimize_layout_euclidean(
            self.subsample_indices,
            self.embedding, 
            n_epochs=self.epochs,
            epochs_per_positive_sample=self.epochs_per_pair_positive,
            epochs_per_negative_sample=self.epochs_per_pair_negative,
            gamma=self.gamma,
            initial_alpha=0.25,
            verbose=False,
        )

    def _subsample_interactions(self):
        """
        Subsample the interactions.
        """
        if self.subsample_factor == 1:
            self.subsample_indices = self.all_indices
            return
        # now randomly sample from all of remaining O(n^2) pairs
        total_pairs = self.all_indices.shape[1]  # total number of pairs in the upper triangular part of the matrix
        n_samples = int(total_pairs * self.subsample_factor)
        random_pairs = np.random.choice(total_pairs, n_samples, replace=False)
        # get the indices of the sampled pairs
        # subsume
        self.subsample_indices = self.all_indices[:, random_pairs]
        # add knn indices to the subsample
        knn_indices = np.array(self.knn_indices)
        self.subsample_indices = np.concatenate((self.subsample_indices, knn_indices), axis=1)
        # make sure we have unique pairs
        self.subsample_indices = np.unique(self.subsample_indices, axis=1)

    def plot_distances(self):
        plt.figure()
        plt.hist(self.distances, bins=100)
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

    def plot_apsp(self):
        plt.figure()
        plt.hist(self.apsp.flatten(), bins=100)
        plt.title("APSP Energy Distribution")
        plt.xlabel("APSP Energy")
        plt.ylabel("Count")
        plt.show()