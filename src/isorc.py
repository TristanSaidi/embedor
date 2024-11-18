import numpy as np
from src.orcml import *
from src.utils.graph_utils import *
from src.plotting import *
from sklearn import manifold
from tqdm import tqdm
import torch


class ISORC(object):

    def __init__(
            self, 
            orcmanl=None, 
            exp_params=default_exp_params, 
            verbose=False,
            init='spectral',
            dim=2,
            repulsion_scale=10.0
        ):
        """ 
        Initialize the ISORC algorithm.
        Parameters
        ----------
        orcmanl : ORCManL
            The ORCManL object.
        exp_params : dict
            The experimental parameters. Includes 'mode', 'n_neighbors', 'epsilon', 'lda', 'delta'.
        verbose : bool, optional
            Whether to print verbose output for ISORC algorithm.
        init : str, optional
            The initialization method for the embedding (if any).
        dim : int, optional
            The dimensionality of the embedding (if any).
        """
        # dim must be int if init != 'ambient'
        assert isinstance(dim, int) or init == 'ambient', "dim must be an integer if init != 'ambient'"
        if orcmanl is None:
            self.orcmanl = ORCManL(exp_params=exp_params, verbose=verbose)
        else:
            self.orcmanl = orcmanl
        # get number of cc's in the pruned graph
        self.n_cc = nx.number_connected_components(self.orcmanl.G_pruned)
        self.exp_params = self.orcmanl.exp_params
        self.verbose = verbose
        self.repulsion_scale = repulsion_scale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_structs()
        self._ann_summary(self.orcmanl.G_ann) # get the annotation summary
        # embedding map
        self.emb_map = {
            'spectral': self._spectral
        }
        self._init_emb(init, dim)

    def _setup_structs(self):
        """
        Setup data structures for the ISORC algorithm.
        """
        # grab graph from ORCManL
        self.G = self.orcmanl.G_pruned
        self.A = self.orcmanl.A_pruned

        self.index_map = np.array(np.argsort(self.G.nodes))
        # inverse index map
        self.inv_index_map = np.zeros(len(self.index_map), dtype=int)
        for i, idx in enumerate(self.index_map):
            self.inv_index_map[idx] = i
        # adjust indexing of A
        self.A = self.A[self.index_map][:, self.index_map]
        # create adjacency mask
        self.adj_mask = torch.tensor(self.A != 0)
        self.adj_mask = self.adj_mask.to(self.device).requires_grad_(False)


    def _init_emb(self, init, dim):
        """
        Initialize the embedding.
        Parameters
        ----------
        init : str
            The initialization method.
        dim : int
            The dimensionality of the embedding.
        """
        if init == 'ambient':
            # data
            self.X = torch.tensor(
                self.orcmanl.X
            ).to(self.device).requires_grad_(True)
        else:
            self.X = self.emb_map[init](dim)
            self.X = torch.tensor(self.X).to(self.device)
        self.dim = dim if dim is not None else self.X.shape[1]
        self.init = init
        self.X_opt = None

    def _spectral(self, n_components):
        """
        Compute the spectral embedding of a graph.
        Parameters
        ----------
        A : array-like, shape (n_samples, n_samples)
            The adjacency matrix of the graph.
        n_components : int
            The number of components to keep.
        Returns
        -------
        Y : array-like, shape (n_samples, n_components)
            The spectral embedding of the graph.
        """
        Y = manifold.spectral_embedding(n_components=n_components, adjacency=self.orcmanl.A) # unpruned adjacency
        # scale eigenvectors to have similar scale as original data
        pw_dist_Y = scipy.spatial.distance.pdist(Y)
        max_pw_dist_Y = np.max(pw_dist_Y)
        scale = self.max_non_shortcut_dist.cpu().numpy() / max_pw_dist_Y
        # if more than one connected component, scale by repulsion scale
        if self.n_cc > 1:
            print(f"Scaling by repulsion scale: {self.repulsion_scale}")
            scale *= self.repulsion_scale
        Y *= scale
        return Y


    def _ann_summary(self, G):
        """
        Get information regarding orcmanl annotations.
        """
        shortcut_indices = []
        non_shortcut_indices = []

        # iterate through edges of G
        for u, v in G.edges:
            # get the isorc annotation of the edge: shortcut
            sc = G[u][v]['shortcut']
            if sc:
                shortcut_indices.append([u, v])
            else:
                non_shortcut_indices.append([u, v])

        # create shortcut indices matrix
        self.shortcut_indices = torch.tensor(shortcut_indices).to(self.device)
        self.non_shortcut_indices = torch.tensor(non_shortcut_indices).to(self.device)

        self.geo_dist = scipy.sparse.csgraph.shortest_path(self.A, directed=False)
        self.geo_dist = torch.tensor(self.geo_dist).to(self.device)
        # create a mask of noninf values
        self.intracluster_mask = self.geo_dist != np.inf
        self.intracluster_mask = self.intracluster_mask.to(self.device).requires_grad_(False)

        # max non-shortcut distance
        self.max_non_shortcut_dist = torch.max(self.geo_dist[self.geo_dist != np.inf]).detach()
        print(f"Max non-shortcut distance: {self.max_non_shortcut_dist}")

        # target distances are geo distances, and max non-shortcut distance for shortcut edges where geo distance is infinite
        self.target_dist = self.geo_dist.clone().requires_grad_(False)
        for u, v in shortcut_indices:
            if self.target_dist[u, v] == np.inf:
                self.target_dist[u, v] = self.repulsion_scale * self.max_non_shortcut_dist
        # create a mask of noninf values
        self.noninf_mask = self.target_dist != np.inf
        self.noninf_mask = self.noninf_mask.to(self.device).requires_grad_(False)
        # fill in inf values with 0 now for loss computation
        self.target_dist[self.target_dist == np.inf] = 0
        # ORC -> weight
        self.W = torch.tensor(
            weight_fn(
                self.G
            )
        ).to(self.device).requires_grad_(False)
        # set shortcut weights to 1 if they exist
        if self.shortcut_indices.shape[0] > 0:
            self.W[self.shortcut_indices[:, 0], self.shortcut_indices[:, 1]] = 1

    def fit_graph(
                self,
                n_iter=1000,
                beta=1e-3,
                weight_orc=True,
                p=10,
                lr=0.1,
                patience=5,
        ):
            """
            Fit the ISORC algorithm.
            Parameters
            ----------
            n_iter : int, optional
                The number of iterations.
            lr : float, optional
                The learning rate.
            """
            # optimize X so that pairwise distances are close to the equilibrium distances
            self.X_opt = self.X.clone().detach().requires_grad_(True)
            self.W_p = self.W ** p
            # optimizer and lr scheduler
            optimizer = torch.optim.Adam([self.X_opt], lr=lr)
            # animation
            losses = []
            with tqdm(total=n_iter) as pbar:
                for i in range(n_iter):
                    optimizer.zero_grad()
                    pdist = torch.cdist(self.X_opt, self.X_opt, p=2)
                    diff = self.target_dist - pdist
                    # clamp elements specified by self.shortcut_indices to zero from below, as we dont want to penalize excess distance between shortcut pairs
                    squared_diff = torch.abs(diff) ** 2
                    if weight_orc:
                        squared_diff = squared_diff * self.W_p # element-wise multiplication
                    # only consider target pairs with noninf geo distances
                    squared_diff = torch.masked_select(squared_diff, self.adj_mask)
                    loss_geo = torch.sum(squared_diff)
                    # compute the covariance matrix of the data
                    cov = torch.cov(self.X_opt)
                    # compute the spread loss as the trace of the covariance matrix
                    loss_spread = -1 * torch.trace(cov) / cov.shape[0]
                    loss = loss_geo + beta * loss_spread
                    loss.backward()
                    
                    optimizer.step()
                    pbar.set_postfix({'loss': loss.item(), 'loss_geo': loss_geo.item(), 'loss_spread': loss_spread.item()})
                    pbar.update(1)
                    losses.append(loss.item())
                    if len(losses) > patience:
                        if loss.item() > np.max(losses[-patience:]):
                            print(f"Early stopping at iteration {i}")
                            break
            return self.X_opt.detach().cpu().numpy(), losses