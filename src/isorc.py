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
        # self._eq_matrix() # compute the equilibrium matrix


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
        Parameters
        ----------
        G : networkx.Graph
            The annotated graph.
        Returns
        -------
        sc_ind : np.ndarray
            The shortcut indicator matrix.
        D_G_prime : np.ndarray
            The G_prime distance matrix.
        """
        # create two |V| x |V| matrices
        n = len(G.nodes)
        sc_ind = np.zeros((n, n))
        D_G_prime = np.zeros((n, n))

        # iterate through edges of G
        for u, v in G.edges:
            # get the isorc annotation of the edge: shortcut
            sc = G[u][v]['shortcut']
            sc_ind[u, v] = sc
            sc_ind[v, u] = sc
            # get the isorc annotation of the edge: distance
            d = G[u][v]['G_prime_dist']
            D_G_prime[u, v] = d
            D_G_prime[v, u] = d    

        self.D_G_prime = D_G_prime
        self.sc_ind = sc_ind # shortcut indicator mask matrix
        self.non_sc_mask = ~sc_ind.astype(bool) # non-shortcut indicator mask matrix
        # create torch versions
        self.D_G_prime_torch = torch.tensor(D_G_prime).to(self.device)
        self.sc_ind_torch = torch.tensor(sc_ind).to(self.device)
        self.non_sc_mask_torch = torch.tensor(self.non_sc_mask).to(self.device)
        # create a connected component mask by computing pairwise distances with adjacency matrix
        self.geo_dist = scipy.sparse.csgraph.shortest_path(self.A, directed=False)
        # set infinities to zero, all other values to 1
        self.connected_comp_mask = torch.tensor(self.geo_dist < np.inf).to(self.device)
        self.geo_dist = torch.tensor(self.geo_dist).to(self.device)
        # replace infinities with 0
        self.masked_geo_dist = torch.where(self.geo_dist == np.inf, torch.tensor(0.0).to(self.device), self.geo_dist)
        # get max value of nonshortcut edges
        self.max_val = torch.max(self.geo_dist[self.geo_dist < np.inf])

    def _eq_matrix(self, dist_scale=1.0):
        """
        Compute the spring equilibrium matrix for the isorc embedding.
        Parameters
        ----------
        dist_scale : float, optional
            The distance scaling factor.
        """
        # create the equilibrium matrix as element-wise maximum of A and sc_ind * D_G_prime
        E = np.maximum(self.A.cpu().numpy(), self.sc_ind * self.D_G_prime * dist_scale)
        self.E = torch.tensor(E).to(self.device)

    def fit(
            self,
            n_iter=1000,
            beta=0.0,
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
        ).to(self.device).requires_grad_(True)
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
                masked_pdist = pdist * self.connected_comp_mask
                # squared displacement for same cc pairs
                D_sq_same_cc = (masked_pdist - self.masked_geo_dist)**2
                # squared displacement for shortcut edges
                print(self.max_val)
                D_sq_sc = (masked_pdist - self.max_val)**2 * self.sc_ind_torch
                # clamp D_sq_sc so max value is max value of nonshortcut edges
                D_sq_sc = torch.clamp(D_sq_sc, max=self.max_val)
                # compute loss
                loss_sc = torch.sum(D_sq_sc)
                loss_same_cc = torch.sum(D_sq_same_cc)
                loss_spread = - beta * fro_norm
                loss = loss_sc + loss_spread
                loss.backward()
                optimizer.step()
                if i % steps_per_frame == 0:
                    frames.append(self.X_opt.clone().detach().cpu().numpy())
                pbar.set_postfix({'loss': loss.item(), 'loss_sc': loss_sc.item(), 'loss_same_cc': loss_same_cc.item(), 'loss_spread': loss_spread.item()})
                pbar.update(1)
        return self.X_opt.detach().cpu().numpy(), frames

        