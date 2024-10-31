import numpy as np
from src.orcml import *
from src.utils.graph_utils import *
from src.plotting import *
import sklearn.metrics as metrics
from tqdm import tqdm
import torch


class ISORC(object):

    def __init__(self, orcmanl=None, exp_params=default_exp_params, verbose=False):
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
        """
        if orcmanl is None:
            self.orcmanl = ORCManL(exp_params=exp_params, verbose=verbose)
        else:
            self.orcmanl = orcmanl
        self.exp_params = self.orcmanl.exp_params
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_structs()
        self._ann_summary(self.orcmanl.G_ann) # get the annotation summary

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
        # data
        self.X = torch.tensor(
            self.orcmanl.X
        ).to(self.device).requires_grad_(True)
        self.X_opt = None


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
        shortcut_indices = torch.tensor(shortcut_indices).to(self.device)
        non_shortcut_indices = torch.tensor(non_shortcut_indices).to(self.device)

        self.geo_dist = scipy.sparse.csgraph.shortest_path(self.A, directed=False)
        self.geo_dist = torch.tensor(self.geo_dist).to(self.device)

        # max non-shortcut distance
        self.max_non_shortcut_dist = torch.max(self.geo_dist[self.geo_dist != np.inf]).detach()
        print(f"Max non-shortcut distance: {self.max_non_shortcut_dist}")

        # target distances are geo distances, and max non-shortcut distance for shortcut edges where geo distance is infinite
        self.target_dist = self.geo_dist.clone().requires_grad_(False)
        for u, v in shortcut_indices:
            if self.target_dist[u, v] == np.inf:
                self.target_dist[u, v] = self.max_non_shortcut_dist
        # create a mask of noninf values
        self.noninf_mask = self.target_dist != np.inf
        self.noninf_mask = self.noninf_mask.to(self.device).requires_grad_(False)
        # fill in inf values with 0 now for loss computation
        self.target_dist[self.target_dist == np.inf] = 0

    def fit(
            self,
            n_iter=1000,
            beta=0,
            lr=0.1,
            frames=10
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
        self.X_opt = torch.tensor(
            self.orcmanl.X
        ).float().to(self.device).requires_grad_(True)
        # optimizer and lr scheduler
        optimizer = torch.optim.Adam([self.X_opt], lr=lr)
        # animation
        steps_per_frame = n_iter // frames
        frames = [self.X_opt.clone().detach().cpu().numpy()]
        with tqdm(total=n_iter) as pbar:
            for i in range(n_iter):
                optimizer.zero_grad()
                pdist = torch.cdist(self.X_opt, self.X_opt, p=2)
                fro_norm = torch.norm(pdist)
                squared_diff = (pdist - self.target_dist) ** 2
                # only consider target pairs with noninf geo distances
                squared_diff = torch.masked_select(squared_diff, self.noninf_mask)
                loss_geo = torch.sum(squared_diff)
                loss_spread = - beta * fro_norm
                loss = loss_geo + loss_spread
                loss.backward()

                optimizer.step()
                if i % steps_per_frame == 0:
                    frames.append(self.X_opt.clone().detach().cpu().numpy())
                pbar.set_postfix({'loss': loss.item(), 'loss_geo': loss_geo.item(), 'loss_spread': loss_spread.item()})
                pbar.update(1)
        return self.X_opt.detach().cpu().numpy(), frames

        