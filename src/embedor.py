# check if matplotlib is already imported
import matplotlib.pyplot as plt
# # from src.data.data import *
from src.utils.graph_utils import compute_frc, compute_orc, get_nn_graph
# # from src.utils.embeddings import *
import numpy as np
from src.utils.layout import *
from umap.spectral import spectral_layout
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from src.plotting import plot_graph_2D

import scipy
import networkit as nk
import time

ENERGY_PARAMS = {
    'orc': {
        'k_max': 1,
        'k_min': -2,
        'k_crit': 0
    },
    'frc': {
        'k_max': 25,
        'k_min': -35,
        'k_crit': -5
    }
}

class EmbedOR(object):
    def __init__(
            self, 
            nng_params = {
                'mode': 'nbrs',
                'n_neighbors': 15,
            }, 
            dim=2,
            p = 3,
            epochs=300,
            perplexity=150,
            verbose=False,
            seed=10,
            edge_weight='orc',
            subsample=False,
            subsample_factor=0.05,
            n_landmarks=None,
            landmark_selection='random',
            approx_affinities=False
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
        self.nng_params = nng_params
        assert 'mode' in self.nng_params, "Nearest neighbor graph parameter 'mode' not provided"
        assert 'n_neighbors' in self.nng_params or 'epsilon' in self.nng_params, "Nearest neighbor graph parameter 'k' or 'epsilon' not provided"
        self.nn_mode = nng_params.get('mode', 'nbrs')
        self.k = self.nng_params.get('n_neighbors', None)
        self.epsilon = self.nng_params.get('epsilon', None)
        self.p = p
        self.epochs = epochs
        self.perplexity = perplexity
        self.edge_weight = edge_weight
        # obtain energy parameters
        if edge_weight in ENERGY_PARAMS:
            energy_params = ENERGY_PARAMS[edge_weight]
            self.k_max = energy_params['k_max']
            self.k_min = energy_params['k_min']
            self.k_crit = energy_params['k_crit']
        self.exp_params = {
            'mode': self.nn_mode,
            'n_neighbors': self.k,
            'epsilon': self.epsilon,
            'p': self.p,
        }
        self.verbose = verbose
        self.seed = seed
        self.X = None
        self.fitted = False
        self.subsample = subsample
        self.subsample_factor = subsample_factor
        self.n_landmarks = n_landmarks
        self.landmark_selection = landmark_selection
        self.approx_affinities = approx_affinities

    def fit_transform(self, X=None):
        if not self.fitted:
            self.fit(X)
        if self.subsample:
            self._subsample_interactions()
        self._init_embedding()
        self._layout(
            affinities=self.all_affinities,
            repulsions=self.all_repulsions
        )
        return self.embedding

    def fit(self, X=None, A=None):
        if X is None and A is None:
            raise ValueError("Either data X or adjacency matrix A must be provided.")
        if X is not None:
            self.X = X
            self._build_nnG() # self.G, self.curvatures, self.A are now available
        else:
            self.G = nx.from_numpy_array(A)
        self._compute_curvatures() # self.curvatures are now available
        self._compute_distances()
        self._compute_affinities()
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
        time_start = time.time()
        return_dict = get_nn_graph(self.X, self.nng_params)
        self.G = return_dict['G']
        time_end = time.time()
        print(f"Time taken to build nearest neighbor graph: {time_end - time_start:.2f} seconds")
    
    def _compute_curvatures(self):
        # compute ORC
        time_start = time.time()
        if self.edge_weight == "orc":
            return_dict = compute_orc(self.G, nbrhood_size=1) # compute ORC using 1-hop neighborhood
            self.curvatures = return_dict['orcs']
            self.G = return_dict['G']

        elif self.edge_weight == "frc":
            return_dict = compute_frc(self.G)
            self.curvatures = return_dict['frcs']
            self.k_min = min(self.k_min, min(self.curvatures)-1) # -1 to avoid log(0)
            self.k_max = max(self.k_max, max(self.curvatures))
            self.G = return_dict['G']

        time_end = time.time()
        print(f"Time taken to compute {self.edge_weight.upper()} for each edge: {time_end - time_start:.2f} seconds")

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
        
        time_start = time.time()
        if self.edge_weight != "euclidean":
            k_max = self.k_max
            k_min = self.k_min
            k_crit = self.k_crit
            self.energies = []

            for idx, (u, v) in enumerate(self.G.edges()):
                orc = self.curvatures[idx]
                c = 1/np.log((k_max-k_min)/(k_crit-k_min))                
                energy = (-c * np.log(orc - k_min) + c * np.log(k_crit - k_min) + 1) ** self.p + 1 # energy(k_max) = 1, energy(k_min) = infty, energy(k_crit) = 2                max_energy = max(energy, max_energy)
                energy = np.clip(energy, 0, max_val) # clip energy to max
                energy = energy * self.G[u][v]['weight'] # scale energy by weight (i.e. Euclidean distance)
                self.G[u][v]['energy'] = energy
                self.energies.append(energy)
            self.G_nk = nk.nxadapter.nx2nk(self.G, weightAttr='energy')                    
        else:
            self.G_nk = nk.nxadapter.nx2nk(self.G, weightAttr='weight')
        time_end = time.time()
        print(f"Time taken to compute edge energies: {time_end - time_start:.2f} seconds")

        time_start = time.time()
        if self.n_landmarks is not None: # landmark APSP
            self._landmark_apsp()
        else: # exact APSP
            self.apsp = nk.distance.APSP(self.G_nk).run().getDistances()
            self.apsp = np.array(self.apsp)
        time_end = time.time()
        print(f"Time taken to compute APSP: {time_end - time_start:.2f} seconds")

        indices = list(self.G.nodes())
        inverse_indices = [indices.index(i) for i in range(len(indices))]
        self.apsp = self.apsp[inverse_indices, :][:, inverse_indices]
        assert np.allclose(self.apsp, self.apsp.T), "APSP matrix must be symmetric."

    def _landmark_apsp(self):
        # algorithm from Potamias et. al. https://www.francescobonchi.com/paper_7.pdf
        if self.landmark_selection == 'random':
            # select random landmarks
            self.landmark_indices = np.random.choice(self.G.number_of_nodes(), self.n_landmarks, replace=False)
        else: 
            betweenness = nk.centrality.ApproxBetweenness(self.G_nk).run().ranking() # returns list of tuples (node, betweenness)
            landmark_indices_tuple = betweenness[:self.n_landmarks]  # take the top n_landmark
            self.landmark_indices = [node for node, _ in landmark_indices_tuple]
        nk_obj = nk.distance.SPSP(self.G_nk, self.landmark_indices).run()
        X_emb = np.array(nk_obj.run().getDistances()).T
        # L = scipy.spatial.distance_matrix(X_emb, X_emb, p=np.inf) # lower bound estimator
        L = pairwise_distances(X_emb, metric='chebyshev') # lower bound estimator
        apsp = L
        # fill diag with 0
        np.fill_diagonal(apsp, 0)
        self.apsp = apsp

    def _compute_affinities(self):
        time_start = time.time()
        from sklearn.neighbors import kneighbors_graph
        if self.approx_affinities:
            # get mask for 5 * perplexity k-nearest neighbors
            n_neighbors = min(5 * self.perplexity, self.X.shape[0] - 1)  # ensure we don't exceed the number of points
            A_perp = kneighbors_graph(self.apsp, n_neighbors=n_neighbors, mode='connectivity', metric='precomputed')
            row, col = A_perp.nonzero()
            apsp_perp = self.apsp[row, col]
            apsp_perp = scipy.sparse.csr_matrix((apsp_perp, (row, col)), shape=A_perp.shape)

            self.all_affinities = squareform(joint_probabilities_nn(apsp_perp, desired_perplexity=self.perplexity, verbose=0))
        else:
            # compute joint probabilities from the APSP matrix
            self.all_affinities = squareform(joint_probabilities(self.apsp, desired_perplexity=self.perplexity, verbose=0))
        # symmetrize affinities
        self.all_affinities = (self.all_affinities + self.all_affinities.T) / 2
        self.all_repulsions = 1 - self.all_affinities
        # fill diagonal with 0
        np.fill_diagonal(self.all_affinities, 0)
        np.fill_diagonal(self.all_repulsions, 0)
        time_end = time.time()
        print(f"Time taken to compute affinities: {time_end - time_start:.2f} seconds")


    def _init_embedding(self):
        # spectral initialization
        time_start = time.time()
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
        time_end = time.time()
        print(f"Time taken to initialize embedding: {time_end - time_start:.2f} seconds")

    def _layout(self, affinities, repulsions):
        time_start = time.time()
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
        time_end = time.time()
        print(f"Time taken to optimize layout: {time_end - time_start:.2f} seconds")

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

    def plot_low_energy_graph(self, edge_pctile=33):
        self.G_low_energy = self.G.copy()
        threshold = np.percentile(self.energies, edge_pctile)
        for idx,(u, v) in enumerate(self.G_low_energy.edges()):
            if self.energies[idx] > threshold:
                self.G_low_energy.remove_edge(u, v)
        # plot the graph
        plot_graph_2D(self.embedding, self.G_low_energy, node_color=None, edge_width=0.1, node_size=0.0, edge_color='green')

    def plot_full_graph(self):
        plot_graph_2D(self.embedding, self.G, node_color=None, edge_width=0.1, node_size=0.0, edge_color='green')