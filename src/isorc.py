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
        self._eq_matrix() # compute the equilibrium matrix


    def _setup_structs(self):
        """
        Setup data structures for the ISORC algorithm.
        """
        # grab graph from ORCManL
        self.G = self.orcmanl.G_ann
        self.A = torch.tensor(self.orcmanl.A).to(self.device).requires_grad_(False)
        # convert to unwieghted adjacency matrix
        self.A_uw = torch.tensor(self.orcmanl.A > 0).to(self.device)
        self.index_map = np.array(np.argsort(self.G.nodes))
        # data
        self.X = torch.tensor(
            np.array(
                [self.G.nodes[i]['pos'] for i in self.G.nodes]
            )[self.index_map]
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

        # replace  np.inf d_G_prime entries with 10*max finite entry
        max_finite = np.max(D_G_prime[np.isfinite(D_G_prime)])
        D_G_prime[np.isinf(D_G_prime)] = 10 * max_finite
        self.D_G_prime = D_G_prime
        self.sc_ind = sc_ind # shortcut indicator mask matrix
        self.non_sc_mask = ~sc_ind.astype(bool) # non-shortcut indicator mask matrix
        # create torch versions
        self.sc_ind_torch = torch.tensor(sc_ind).to(self.device)
        self.non_sc_mask_torch = torch.tensor(self.non_sc_mask).to(self.device)

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
            beta=1.0,
            alpha=0.1,
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
            np.array(
                [self.G.nodes[i]['pos'] for i in self.G.nodes]
            )[self.index_map]
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
                masked_pdist = pdist * self.A_uw
                # squared displacement
                D_sq = (masked_pdist - self.E)**2
                # clamp D_sq so max value is max value of nonshortcut edges
                max_val = 5 * torch.max(D_sq[self.non_sc_mask_torch])
                D_sq = torch.clamp(D_sq, max=max_val)
                # compute loss
                loss = torch.sum(D_sq) - beta * fro_norm
                loss.backward()
                optimizer.step()
                if i % steps_per_frame == 0:
                    frames.append(self.X_opt.clone().detach().cpu().numpy())
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
        return self.X_opt.detach().cpu().numpy(), frames

        