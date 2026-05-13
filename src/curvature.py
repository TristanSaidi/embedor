import random
import networkx as nx
import numpy as np
from typing import List, Tuple
from collections import deque, Counter
from itertools import combinations


def _neighbors_arrays(G: nx.Graph, 
                        nodes: List[int]) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    For each node in `nodes`, return array of neighbor indices (in nodes-list indexing) and degree.
    - Inputs:
        - G: nx.Graph
        - nodes: list of node IDs in G
    - Returns:
        - neigh_idx_list: list of np.ndarray, each array contains neighbor indices for the node
        - deg: np.ndarray of degrees for each node
    """
    idx = {n: i for i, n in enumerate(nodes)}
    neigh_idx_list: List[np.ndarray] = []
    deg = np.zeros(len(nodes), dtype=np.int32)
    for i, u in enumerate(nodes):
        nbrs = list(G.neighbors(u))
        arr = np.array([idx[v] for v in nbrs], dtype=np.int32)
        neigh_idx_list.append(arr)
        deg[i] = int(arr.shape[0])
    return neigh_idx_list, deg

def _edge_index_arrays(G: nx.Graph, 
                        nodes: List[int], 
                        length_attr: str = "length") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return edge index arrays u_idx, v_idx and lengths L aligned with G.edges().
    - Inputs:
        - G: nx.Graph
        - nodes: list of node IDs in G
        - length_attr: str, edge attribute name for lengths
    - Returns:
        - u_idx: np.ndarray of source node indices (in nodes-list indexing)
        - v_idx: np.ndarray of target node indices (in nodes-list indexing)
        - L: np.ndarray of edge lengths
    """
    idx = {n: i for i, n in enumerate(nodes)}
    E = G.number_of_edges()
    u_idx = np.empty(E, dtype=np.int32)
    v_idx = np.empty(E, dtype=np.int32)
    L = np.empty(E, dtype=np.float64)
    for k, (u, v, d) in enumerate(G.edges(data=True)):
        u_idx[k] = int(idx[u])
        v_idx[k] = int(idx[v])
        L[k] = float(d.get(length_attr, 1.0))
    return u_idx, v_idx, L

def _build_tree_parent(G: nx.Graph, 
                       root: int, 
                       method: str, 
                       length_attr: str):
    if method == "src-spt":
        return _build_spt_parent(G, root, length_attr)
    elif method == "src-mst":
        return _build_mst_parent(G, root, length_attr)
    elif method == "src-randtree":
        return _build_randtree_parent(G, root, length_attr)
    else:
        raise ValueError(f"Unknown SRC tree method: {method}")

def _uniform_spanning_tree_wilson(G: nx.Graph, seed: int | None = None) -> nx.Graph:
    """
    Sample a uniform spanning tree using Wilson's algorithm.
    Reference: https://en.wikipedia.org/wiki/Wilson%27s_algorithm
    - Inputs:
        - G: nx.Graph
        - seed: Optional[int], random seed for reproducibility
    - Returns:
        - T: nx.Graph, a uniform spanning tree of G    
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return nx.Graph()

    # pick a root for the Wilson process (not the SRC root; just algorithm root)
    in_tree = set([nodes[0]])
    parent = {nodes[0]: None}

    # helper: random neighbor
    def rand_nbr(u):
        nbrs = list(G.neighbors(u))
        if not nbrs:
            raise ValueError("Graph has isolated node; spanning tree impossible.")
        return rng.choice(nbrs)

    remaining = [v for v in nodes if v not in in_tree]
    while remaining:
        start = rng.choice(remaining)

        # loop-erased random walk: path as ordered dict-like
        walk = [start]
        pos = {start: 0}

        cur = start
        while cur not in in_tree:
            nxt = rand_nbr(cur)
            if nxt in pos:
                # erase loop
                loop_start = pos[nxt]
                walk = walk[: loop_start + 1]
                pos = {node: i for i, node in enumerate(walk)}
            else:
                walk.append(nxt)
                pos[nxt] = len(walk) - 1
            cur = walk[-1]

        # add the loop-erased path to the tree
        for i in range(len(walk) - 1):
            u, v = walk[i], walk[i + 1]
            parent[u] = v  # u points toward existing tree
            in_tree.add(u)

        # refresh remaining
        remaining = [v for v in nodes if v not in in_tree]

    # build nx.Graph tree edges
    T = nx.Graph()
    T.add_nodes_from(nodes)
    for u, p in parent.items():
        if p is None:
            continue
        T.add_edge(u, p)

    return T

def _build_randtree_parent(
    G: nx.Graph,
    root: int,
    length_attr: str = "length",
    seed: int | None = None,
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Build a RANDOM extracted spanning tree (uniform spanning tree),
    then root it at `root` (BFS) to produce parent/plen.
    """
    print("Using RANDOM spanning tree for SRC tree construction.")

    if G.number_of_nodes() == 0:
        return [], np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    # 1) sample a uniform spanning tree
    T = _uniform_spanning_tree_wilson(G, seed=seed)

    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    if root not in idx:
        root = nodes[0]

    parent = np.full(n, -1, dtype=np.int32)
    plen = np.zeros(n, dtype=np.float64)

    parent[idx[root]] = -1
    plen[idx[root]] = 0.0

    # 2) BFS to set parent pointers on the sampled tree
    q = deque([root])
    visited = set([root])
    dist_tree = {root: 0.0}

    while q:
        u = q.popleft()
        for v in T.neighbors(u):
            if v in visited:
                continue
            visited.add(v)
            parent[idx[v]] = idx[u]

            # IMPORTANT: tree edge length inherited from original graph G
            Luv = float(G[u][v].get(length_attr, 1.0))
            plen[idx[v]] = Luv
            dist_tree[v] = dist_tree[u] + Luv
            q.append(v)

    order = sorted(nodes, key=lambda x: float(dist_tree.get(x, np.inf)))
    return order, parent, plen

def _build_spt_parent(G: nx.Graph,
                    root: int,
                    length_attr: str = "length",
                    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Build a shortest-path tree (SPT) rooted at `root`.

    Returns:
        order: list of nodes in a topological-like order (root-first)
        parent: parent[idx(node)] = idx(parent_node) or -1
        plen: plen[idx(node)] = length(parent->node) on the chosen predecessor
    """
    print('Using SPT for SRC tree construction.')
    # dijkstra predecessors/distances on current edge lengths
    pred, dist = nx.dijkstra_predecessor_and_distance(G, root, weight=length_attr)

    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    parent = np.full(n, -1, dtype=np.int32)
    plen = np.zeros(n, dtype=np.float64)

    r = idx[root]
    parent[r] = -1
    plen[r] = 0.0

    # choose one predecessor (tie-break by smallest node id after relabeling)
    for v in nodes:
        if v == root:
            continue
        if v not in pred or len(pred[v]) == 0:
            # disconnected; keep parent=-1
            continue

        dv = dist.get(v, np.inf)
        cand = [p for p in pred[v] if dist.get(p, np.inf) < dv - 1e-12]

        if not cand:
            pnode = min(pred[v], key=lambda p: dist.get(p, np.inf))
            if dist.get(pnode, np.inf) < dv - 1e-12:
                parent[idx[v]] = idx[pnode]
                plen[idx[v]] = float(G[pnode][v].get(length_attr, 1.0))
            else:
                continue
        else:
            pnode = min(cand)
            parent[idx[v]] = idx[pnode]
            plen[idx[v]] = float(G[pnode][v].get(length_attr, 1.0))

    order = sorted(nodes, key=lambda x: float(dist.get(x, np.inf)))
    return order, parent, plen

def _build_mst_parent(
    G: nx.Graph,
    root: int,
    length_attr: str = "length",
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Build a Minimum Spanning Tree (MST) and then root it at `root`.

    Returns:
        order: list of nodes in a root-first order (increasing tree-distance from root)
        parent: parent[idx(node)] = idx(parent_node) or -1
        plen: plen[idx(node)] = length(parent->node) on the MST edge
    """
    print('Using MST for SRC tree construction.')
    # ---- 1) Build MST as a new graph ----
    T = nx.minimum_spanning_tree(G, weight=length_attr)

    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    parent = np.full(n, -1, dtype=np.int32)
    plen = np.zeros(n, dtype=np.float64)

    if root not in idx:
        root = nodes[0]

    r = idx[root]
    parent[r] = -1
    plen[r] = 0.0

    # ---- 2) Root the MST at root by BFS/DFS ----
    # BFS over the tree edges to assign parent pointers
    q = deque([root])
    visited = set([root])

    # tree-distance from root (sum of plen along tree path)
    dist_tree = {root: 0.0}

    while q:
        u = q.popleft()
        for v in T.neighbors(u):
            if v in visited:
                continue
            visited.add(v)

            # set parent (in index space)
            parent[idx[v]] = idx[u]

            # edge length along MST
            Luv = float(T[u][v].get(length_attr, 1.0))
            plen[idx[v]] = Luv

            dist_tree[v] = dist_tree[u] + Luv
            q.append(v)

    # ---- 3) order: sort nodes by tree-distance (root-first) ----
    order = sorted(nodes, key=lambda x: float(dist_tree.get(x, np.inf)))

    return order, parent, plen

def _children_index(parent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a compact representation:
        child_nodes: list of child node indices (excluding root/disconnected)
        child_to_eidx: map node_idx -> edge index (0..m-1) or -1
    """
    n = parent.shape[0]
    child_nodes = np.where(parent >= 0)[0].astype(np.int32)
    m = child_nodes.shape[0]
    child_to_eidx = np.full(n, -1, dtype=np.int32)
    child_to_eidx[child_nodes] = np.arange(m, dtype=np.int32)
    return child_nodes, child_to_eidx

def _compute_dist_to_root(parent: np.ndarray, plen: np.ndarray, order: List[int]) -> np.ndarray:
    """
    dist_root[v] on the tree (sum of plen along root->v).
    """
    n = parent.shape[0]
    dist_root = np.zeros(n, dtype=np.float64)

    for v in order:
        pv = parent[v]
        if pv >= 0:
            dist_root[v] = dist_root[pv] + plen[v]
        else:
            dist_root[v] = 0.0
    return dist_root

def _mu_vector_for_node(i: int,
                        neigh: np.ndarray,
                        deg: int,
                        delta: float,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    For node i (index space):
    support indices (including self and neighbors)
    weights (sum to 1)

    μ_i(i)=delta, neighbors get (1-delta)/deg(i)
    """
    if deg <= 0:
        # isolated: all mass stays at self
        return np.array([i], dtype=np.int32), np.array([1.0], dtype=np.float64)

    w_self = float(delta)
    w_nbr = (1.0 - float(delta)) / float(deg)
    supp = np.concatenate([np.array([i], dtype=np.int32), neigh.astype(np.int32)])
    w = np.concatenate([np.array([w_self], dtype=np.float64), np.full(deg, w_nbr, dtype=np.float64)])
    return supp, w

def _tree_distance_uv(u: int,
                      v: int,
                      parent: np.ndarray,
                      plen: np.ndarray,
                    ) -> float:
    """
    Compute tree distance D_p(u,v) on the rooted tree defined by parent/plen.
    This corresponds to S_p(delta_u, delta_v) for p=1.

    parent[i] : parent index of i, or -1 for root
    plen[i]   : length of edge (parent[i] -> i)
    """
    du = 0.0
    dv = 0.0

    uu = int(u)
    vv = int(v)

    # record ancestors of u with distances to root
    seen = {}
    cur = uu
    acc = 0.0
    while cur >= 0:
        seen[cur] = acc
        p = parent[cur]
        if p < 0:
            break
        acc += float(plen[cur])
        cur = int(p)

    # walk v upward until LCA
    cur = vv
    acc_v = 0.0
    while cur not in seen:
        p = parent[cur]
        if p < 0:
            break
        acc_v += float(plen[cur])
        cur = int(p)

    # cur is LCA (or root fallback)
    acc_u = seen.get(cur, 0.0)

    return acc_u + acc_v

def _calculate_src_curvature(parent, plen, method, step, n,
                             neigh_idx_list, deg, delta,
                             L, u_idx, v_idx, p):

    child_nodes, child_to_eidx = _children_index(parent)
    m = int(child_nodes.shape[0])

    # tree-edge lengths λ_e (indexed by child node / eidx)
    lambdas = np.zeros(m, dtype=np.float64)
    for ei, c in enumerate(child_nodes):
        lc = float(plen[int(c)])
        lambdas[ei] = lc if (np.isfinite(lc) and lc > 0) else 1.0

    # --- cap lambdas for stability ---
    if method == "src-mst":
        cap = np.quantile(lambdas, 0.95)
        lambdas = np.minimum(lambdas, cap)
        print(f"[SRC step {step+1:02d}] MST lambda cap: {cap:.4f}", flush=True)

    # --- build per-node flow vectors f_i in R^m ---
    # f_i[e] = total μ_i mass whose root-path crosses tree-edge e
    flow_vecs = np.zeros((n, m), dtype=np.float64)

    for i in range(n):
        supp, w = _mu_vector_for_node(i, neigh_idx_list[i], int(deg[i]), delta)

        # push each support mass to root along parent pointers
        # (uses child_to_eidx to map node->tree-edge index)
        for s, mass in zip(supp.tolist(), w.tolist()):
            if mass == 0.0:
                continue
            max_hops = n + 5
            hops = 0
            cur = int(s)
            while True:
                hops += 1
                if hops > max_hops:
                    raise RuntimeError("Exceeded max hops in tree traversal; possible cycle.")
                eidx = int(child_to_eidx[cur])
                if eidx < 0:
                    break
                flow_vecs[i, eidx] += float(mass)
                cur = int(parent[cur])
                if cur < 0:
                    break

    # --- curvature κ on original edges ---
    E = int(L.shape[0])
    kappas = np.zeros(E, dtype=np.float64)
    dps = np.zeros(E, dtype=np.float64)         # store d^{(k)}(u,v)
    sps = np.zeros(E, dtype=np.float64)        # store Sp(u,v)
    for e in range(E):
        ui = int(u_idx[e]); vi = int(v_idx[e])
        diff = np.abs(flow_vecs[ui] - flow_vecs[vi])
        if p != 1.0:
            Sp_p = float((diff ** p) @ lambdas)
            Sp = float(Sp_p ** (1.0 / p))
        else:
            Sp = float(diff @ lambdas)
        # D = float(L[e])
        Dp = _tree_distance_uv(ui, vi, parent, plen)
        if (not np.isfinite(Dp)) or Dp <= 0:
            Dp = 1.0
        sps[e] = Sp
        dps[e] = Dp
        kappas[e] = 1.0 - (Sp / Dp)
    
    return kappas, sps, dps, E

def tree_edges_from_parent(nodes, parent):
    """
    parent is index-based array.
    nodes is list of node IDs aligned to parent index.
    return set of undirected edges as tuple (min(u,v), max(u,v))
    """
    edges = set()
    for i, p in enumerate(parent):
        if p < 0:
            continue
        u = nodes[i]
        v = nodes[int(p)]
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
    return edges

def _kappas_for_root(G, 
                     root, 
                     nodes, 
                     length_attr="length",
                     delta=0.5, 
                     p=1.0, 
                     method="src-spt"):
    """
    Compute SRC curvature kappas for a given root (single iteration, fixed lengths).
    Returns:
      kappas: (E,) array aligned with _edge_index_arrays ordering
      parent: parent array (index-based)
    """
    neigh_idx_list, deg = _neighbors_arrays(G, nodes)
    u_idx, v_idx, L = _edge_index_arrays(G, nodes, length_attr=length_attr)

    # build SPT parent
    order, parent, plen = _build_spt_parent(G, root, length_attr=length_attr)

    kappas, sps, dps, E = _calculate_src_curvature(
        parent=parent,
        plen=plen,
        method=method,
        step=0,
        n=len(nodes),
        neigh_idx_list=neigh_idx_list,
        deg=deg,
        delta=delta,
        L=L,
        u_idx=u_idx,
        v_idx=v_idx,
        p=p
    )

    return kappas, parent


def measure_root_sensitivity_kappa_vs_tree(
    G,
    length_attr="length",
    n_roots=20,
    tau=0.8,
    delta=0.5,
    p=1.0,
    seed=0,
):
    """
    Measure:
      - RHS: tree delta ratio (symmetric difference of SPT edges)
      - LHS: curvature delta (mean abs difference in kappas)
    over random roots.

    Returns summary stats to include in CSV.
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())

    # sample roots
    if n_roots >= len(nodes):
        roots = nodes
    else:
        roots = rng.sample(nodes, n_roots)

    # precompute kappas & tree edges for each root
    kappas_map = {}
    tree_edges_map = {}

    for r in roots:
        kappas, parent = _kappas_for_root(
            G=G, 
            root=r, 
            nodes=nodes,
            length_attr=length_attr,
            delta=delta,
            p=p,
            method="src-spt",
        )
        kappas_map[r] = kappas
        tree_edges_map[r] = tree_edges_from_parent(nodes, parent)

    denom_tree = max(G.number_of_nodes() - 1, 1)

    # collect pairwise stats
    kappa_l1_list = []
    tree_delta_ratio_list = []

    for r1, r2 in combinations(roots, 2):
        k1 = kappas_map[r1]
        k2 = kappas_map[r2]
        kappa_l1 = float(np.mean(np.abs(k1 - k2)))   # LHS (mean over edges)

        A = tree_edges_map[r1]
        B = tree_edges_map[r2]
        delta_edges = len(A.symmetric_difference(B))
        delta_ratio = float(delta_edges / denom_tree)  # RHS

        kappa_l1_list.append(kappa_l1)
        tree_delta_ratio_list.append(delta_ratio)

    # summary
    kappa_l1_mean = float(np.mean(kappa_l1_list)) if kappa_l1_list else 0.0
    kappa_l1_max  = float(np.max(kappa_l1_list)) if kappa_l1_list else 0.0

    rhs_mean = float(np.mean(tree_delta_ratio_list)) if tree_delta_ratio_list else 0.0
    rhs_max  = float(np.max(tree_delta_ratio_list)) if tree_delta_ratio_list else 0.0

    eps = 1e-12
    ratio_mean = kappa_l1_mean / (rhs_mean + eps)
    ratio_max  = kappa_l1_max  / (rhs_max + eps)

    return {
        "root_sens_n_roots": int(len(roots)),
        "kappa_l1_mean": kappa_l1_mean,
        "kappa_l1_max": kappa_l1_max,
        "tree_delta_ratio_mean": rhs_mean,
        "tree_delta_ratio_max": rhs_max,
        "kappa_over_tree_mean": float(ratio_mean),
        "kappa_over_tree_max": float(ratio_max),
    }

def measure_spt_backbone(G, length_attr="length", n_roots=20, tau=0.8, seed=0):
    rng = random.Random(seed)

    # IMPORTANT: use nodes from each _build_spt_parent call
    all_nodes = list(G.nodes())

    roots = all_nodes if n_roots >= len(all_nodes) else rng.sample(all_nodes, n_roots)

    tree_edges = []
    for r in roots:
        order, parent, plen = _build_spt_parent(G, r, length_attr=length_attr)

        # use the same node ordering as _build_spt_parent internally
        nodes_local = list(G.nodes())

        edges = tree_edges_from_parent(nodes_local, parent)
        tree_edges.append(edges)

    deltas = []
    for A, B in combinations(tree_edges, 2):
        deltas.append(len(A.symmetric_difference(B)))

    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    max_delta  = float(np.max(deltas)) if deltas else 0.0

    cnt = Counter()
    for E in tree_edges:
        cnt.update(E)

    m = len(tree_edges)
    backbone = {e for e, c in cnt.items() if c / m >= tau}

    n = G.number_of_nodes()
    denom = max(n - 1, 1)

    return {
        "n_roots": int(m),
        "tau": float(tau),
        "mean_delta_edges": mean_delta,
        "max_delta_edges": max_delta,
        "mean_delta_ratio": mean_delta / denom,
        "max_delta_ratio": max_delta / denom,
        "backbone_size": int(len(backbone)),
        "backbone_ratio": float(len(backbone) / denom),
    }