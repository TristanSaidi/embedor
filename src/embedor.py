import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import networkit as nk
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from umap.spectral import spectral_layout
from src.utils.graph_utils import compute_frc, compute_orc, compute_src, get_nn_graph
from src.utils.layout import *
from src.plotting import plot_graph_2D
import scipy
import time
import torch

ENERGY_PARAMS = {
    'orc': {'k_max': 1, 'k_min': -2, 'k_crit': 0},
    'frc': {'k_max': 25, 'k_min': -35, 'k_crit': -5},
    'src': {'k_max': 1, 'k_min': -2, 'k_crit': 0},
}

class EmbedOR(object):
    def __init__(
        self,
        nng_params={'mode': 'nbrs', 'n_neighbors': 15},
        dim=2,
        p=3,
        epochs=300,
        perplexity=150,
        verbose=False,
        seed=10,
        edge_weight='orc',
        subsample=False,
        subsample_factor=0.05,
        n_landmarks=None,
        landmark_selection='random',
        approx_affinities=False,
        layout="torch"   
    ):
        self.dim = dim
        self.nng_params = nng_params
        self.nn_mode = nng_params.get('mode', 'nbrs')
        self.k = nng_params.get('n_neighbors', None)
        self.epsilon = nng_params.get('epsilon', None)
        self.p = p
        self.epochs = epochs
        self.perplexity = perplexity
        self.edge_weight = edge_weight
        self.subsample = subsample
        self.subsample_factor = subsample_factor
        self.n_landmarks = n_landmarks
        self.landmark_selection = landmark_selection
        self.approx_affinities = approx_affinities
        self.verbose = verbose
        self.seed = seed
        self.X = None
        self.fitted = False
        self.layout = layout  # "torch" or "numpy"

        # Energy params
        if edge_weight in ENERGY_PARAMS:
            params = ENERGY_PARAMS[edge_weight]
            self.k_max = params['k_max']
            self.k_min = params['k_min']
            self.k_crit = params['k_crit']

    def fit(self, X=None, A=None):
        if X is None and A is None:
            raise ValueError("Either data X or adjacency matrix A must be provided.")
        if X is not None:
            self.X = X
            self._build_nnG()
        else:
            self.G = nx.from_numpy_array(A)

        self._compute_curvatures()
        self._compute_distances()
        self._compute_affinities()
        self._update_G()
        self.fitted = True
        return self

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

    # -----------------------------------
    # Graph construction & energies
    # -----------------------------------
    def _build_nnG(self):
        if self.X is None:
            raise ValueError("Data must be provided to build the nearest neighbor graph.")
        start = time.time()
        return_dict = get_nn_graph(self.X, self.nng_params)
        self.G = return_dict['G']
        end = time.time()
        print(f"Time taken to build nearest neighbor graph: {end-start:.2f}s")

    def _compute_curvatures(self):
        start = time.time()
        if self.edge_weight == "orc":
            result = compute_orc(self.G, nbrhood_size=1)
            self.curvatures = result['orcs']
            self.G = result['G']
        elif self.edge_weight == "frc":
            result = compute_frc(self.G)
            self.curvatures = result['frcs']
            self.k_min = min(self.k_min, min(self.curvatures)-1)
            self.k_max = max(self.k_max, max(self.curvatures))
            self.G = result['G']
        elif self.edge_weight == "src":                    
            result = compute_src(self.G, method="src-spt")
            self.curvatures = result['orcs']
            self.G = result['G']
        end = time.time()
        print(f"Time taken to compute {self.edge_weight.upper()}: {end-start:.2f}s")

        # adjacency matrix
        self.A = nx.to_numpy_array(self.G, weight='weight', nodelist=list(range(len(self.G.nodes()))))
        A_ut = self.A * np.triu(np.ones(self.A.shape), k=1)
        self.knn_indices = A_ut.nonzero()
        self.all_indices = np.stack(np.triu(np.ones(self.A.shape), k=1).nonzero(), axis=0)
        del A_ut
        self.A = csr_matrix(self.A)

    def _compute_distances(self, max_val=np.inf):
        start = time.time()
        self.energies = []
        if self.edge_weight != "euclidean":
            k_max, k_min, k_crit = self.k_max, self.k_min, self.k_crit
            for idx, (u, v) in enumerate(self.G.edges()):
                # clip
                orc = self.curvatures[idx]
                orc = np.clip(orc, k_min + 1e-6, k_max)  # prevent log(<=0)
                c = 1 / np.log((k_max-k_min)/(k_crit-k_min))
                energy = (-c*np.log(orc - k_min) + c*np.log(k_crit - k_min) + 1)**self.p + 1

                # orc = self.curvatures[idx]
                # c = 1 / np.log((k_max-k_min)/(k_crit-k_min))
                # energy = (-c*np.log(orc - k_min) + c*np.log(k_crit - k_min) + 1)**self.p + 1
                energy = np.clip(energy, 0, max_val) * self.G[u][v]['weight']
                self.G[u][v]['energy'] = energy
                self.energies.append(energy)
            self.G_nk = nk.nxadapter.nx2nk(self.G, weightAttr='energy')
        else:
            self.G_nk = nk.nxadapter.nx2nk(self.G, weightAttr='weight')
        end = time.time()
        print(f"Time taken to compute edge energies: {end-start:.2f}s")

        start = time.time()
        if self.n_landmarks is not None:
            self._landmark_apsp()
        else:
            self.apsp = np.array(nk.distance.APSP(self.G_nk).run().getDistances())
        end = time.time()
        print(f"Time taken to compute APSP: {end-start:.2f}s")

        indices = list(self.G.nodes())
        inverse_indices = [indices.index(i) for i in range(len(indices))]
        self.apsp = self.apsp[inverse_indices, :][:, inverse_indices]
        assert np.allclose(self.apsp, self.apsp.T), "APSP must be symmetric."

    def _landmark_apsp(self):
        if self.landmark_selection == 'random':
            self.landmark_indices = np.random.choice(self.G.number_of_nodes(), self.n_landmarks, replace=False)
        else:
            betweenness = nk.centrality.ApproxBetweenness(self.G_nk).run().ranking()
            self.landmark_indices = [node for node,_ in betweenness[:self.n_landmarks]]
        nk_obj = nk.distance.SPSP(self.G_nk, self.landmark_indices).run()
        X_emb = np.array(nk_obj.run().getDistances()).T
        L = pairwise_distances(X_emb, metric='chebyshev')
        np.fill_diagonal(L, 0)
        self.apsp = L

    def _compute_affinities(self):
        start = time.time()
        from sklearn.neighbors import kneighbors_graph
        if self.approx_affinities:
            n_neighbors = min(5*self.perplexity, self.X.shape[0]-1)
            A_perp = kneighbors_graph(self.apsp, n_neighbors=n_neighbors, mode='connectivity', metric='precomputed')
            row, col = A_perp.nonzero()
            apsp_perp = csr_matrix((self.apsp[row, col], (row, col)), shape=A_perp.shape)
            self.all_affinities = squareform(joint_probabilities_nn(apsp_perp, desired_perplexity=self.perplexity, verbose = False))
        else:
            self.all_affinities = squareform(joint_probabilities(self.apsp, desired_perplexity=self.perplexity, verbose = False))
        self.all_affinities = (self.all_affinities + self.all_affinities.T)/2
        self.all_repulsions = 1 - self.all_affinities
        np.fill_diagonal(self.all_affinities, 0)
        np.fill_diagonal(self.all_repulsions, 0)
        end = time.time()
        print(f"Time taken to compute affinities: {end-start:.2f}s")

    def _init_embedding(self):
        start = time.time()
        A_sparse = nx.to_scipy_sparse_array(self.G, weight='affinity', nodelist=list(range(len(self.G.nodes()))))
        self.spectral_init = spectral_layout(data=None, graph=A_sparse, dim=self.dim, random_state=self.seed)
        self.embedding = (self.spectral_init - np.min(self.spectral_init, axis=0)) / (
            np.max(self.spectral_init, axis=0) - np.min(self.spectral_init, axis=0)
        ) - 0.5
        self.spectral_init = self.embedding.copy()
        end = time.time()
        print(f"Time taken to initialize embedding: {end-start:.2f}s")

    def _layout_numpy(self, affinities, repulsions):
        start = time.time()
        if self.subsample:
            affinities = affinities[self.subsample_indices[0], self.subsample_indices[1]]
            repulsions = repulsions[self.subsample_indices[0], self.subsample_indices[1]]
            n_pairs = self.subsample_indices.shape[1]
            N = self.X.shape[0]
            Z = np.sum(affinities)
            self.gamma = (n_pairs - Z)/(Z*n_pairs)
        else:
            N = self.X.shape[0]
            npairs = (N**2 - N)/2
            Z = (np.sum(affinities) - np.trace(affinities))/2
            self.gamma = (npairs - Z)/(Z*N**2)
            self.subsample_indices = None
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
            verbose=False
        )
        end = time.time()
        print(f"Time taken to optimize layout (numpy): {end-start:.2f}s")

    def _layout_torch(self, affinities, repulsions, device=None, batch_size=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.subsample:
            indices = self.subsample_indices
            aff = affinities[indices[0], indices[1]]
            rep = repulsions[indices[0], indices[1]]
            n_pairs = indices.shape[1]
            N = self.X.shape[0]
            Z = np.sum(aff)
            self.gamma = (n_pairs - Z)/(Z*n_pairs)
        else:
            indices = self.all_indices
            aff = affinities[indices[0], indices[1]]
            rep = repulsions[indices[0], indices[1]]
            N = self.X.shape[0]
            npairs = (N**2 - N)//2
            Z = np.sum(aff)
            self.gamma = (npairs - Z)/(Z*N**2)
            self.subsample_indices = None

        epochs_pos = make_epochs_per_pair(aff, n_epochs=self.epochs)
        epochs_neg = make_epochs_per_pair(rep, n_epochs=self.epochs)

        X_embed = torch.tensor(self.embedding, dtype=torch.float32, device=device, requires_grad=True)
        indices_t = torch.tensor(indices, dtype=torch.long, device=device)
        aff_t = torch.tensor(aff, dtype=torch.float32, device=device)
        rep_t = torch.tensor(rep, dtype=torch.float32, device=device)

        optimizer = torch.optim.SGD([X_embed], lr=0.1)
        n_edges = indices_t.shape[1]
        batch_size = batch_size or n_edges

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            total_loss = 0.0
            for start_idx in range(0, n_edges, batch_size):
                end_idx = min(start_idx + batch_size, n_edges)
                batch_idx = slice(start_idx, end_idx)
                emb_i = X_embed[indices_t[0, batch_idx]]
                emb_j = X_embed[indices_t[1, batch_idx]]
                d2 = ((emb_i - emb_j)**2).sum(dim=1)
                f_ij = torch.clamp(1.0/(1.0+d2), min=1e-4)
                loss_pos = (-torch.log(f_ij) * aff_t[batch_idx]).sum()
                loss_neg = (-torch.log(1.0 + 0.001 - f_ij) * rep_t[batch_idx]).sum()
                total_loss += loss_pos + self.gamma * loss_neg
            total_loss.backward()
            optimizer.step()
        self.embedding = X_embed.detach().cpu().numpy()
        print("Optimized layout (torch).")

    def _layout(self, affinities, repulsions, **kwargs):
        if self.layout == "torch":
            return self._layout_torch(affinities, repulsions, **kwargs)
        elif self.layout in ("numpy", "legacy"):
            return self._layout_numpy(affinities, repulsions)
        else:
            raise ValueError(f"Unknown layout={self.layout}")

    def _subsample_interactions(self):
        if self.subsample_factor == 1:
            self.subsample_indices = self.all_indices
            return
        total_pairs = self.all_indices.shape[1]
        n_samples = int(total_pairs * self.subsample_factor)
        random_pairs = np.random.choice(total_pairs, n_samples, replace=False)
        self.subsample_indices = self.all_indices[:, random_pairs]
        knn_indices = np.array(self.knn_indices)
        self.subsample_indices = np.concatenate((self.subsample_indices, knn_indices), axis=1)
        self.subsample_indices = np.unique(self.subsample_indices, axis=1)

    def _update_G(self):
        self.affinities, self.distances = [], []
        for i,(u,v) in enumerate(self.G.edges):
            self.G[u][v]['affinity'] = self.all_affinities[u,v]
            self.affinities.append(self.all_affinities[u,v])
            self.distances.append(self.apsp[u,v])

    def plot_low_energy_graph(self, edge_pctile=33):
        self.G_low_energy = self.G.copy()
        threshold = np.percentile(self.energies, edge_pctile)
        for idx,(u,v) in enumerate(self.G_low_energy.edges):
            if self.energies[idx] > threshold:
                self.G_low_energy.remove_edge(u,v)
        plot_graph_2D(self.embedding, self.G_low_energy, node_color=None, edge_width=0.1, node_size=0.0, edge_color='green')

    def plot_full_graph(self):
        plot_graph_2D(self.embedding, self.G, node_color=None, edge_width=0.1, node_size=0.0, edge_color='green')
