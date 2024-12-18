import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from multiprocessing import Pool

def process_row(args):
    """Compute a single row of the APSP matrix."""
    i, predecessors, A_euc_distance, n = args
    row_distances = np.zeros(n)
    for j in range(n):
        if i == j:
            row_distances[j] = 0
            continue
        total_weight_A = 0
        current = j
        while current != i:
            prev = predecessors[i, current]
            if prev == -9999:  # Path does not exist
                total_weight_A = np.inf
                break
            assert A_euc_distance[prev, current] != 0, f"Zero distance between {prev} and {current}"
            total_weight_A += A_euc_distance[prev, current]
            current = prev
        row_distances[j] = total_weight_A
    return i, row_distances

def compute_apsp_with_dual_weights_multiprocessing(G, weight_A, weight_B):
    """
    Compute the all-pairs shortest path matrix where:
    - Paths are shortest with respect to 'weight_B'.
    - Distances are accumulated with respect to 'weight_A'.
    
    This version parallelizes path reconstruction and weight accumulation using multiprocessing.
    """

    # Step 1: Get adjacency matrices for both weights
    A_kernel_distance = nx.to_numpy_array(G, weight=weight_B)
    A_euc_distance = nx.to_numpy_array(G, weight=weight_A)

    # assert symmetric matrices
    assert np.allclose(A_kernel_distance, A_kernel_distance.T)
    assert np.allclose(A_euc_distance, A_euc_distance.T)

    # Step 2: Compute shortest paths and predecessors with respect to 'weight_B'
    apsp_weight_B, predecessors = shortest_path(
        A_kernel_distance, directed=False, unweighted=False, return_predecessors=True
    )

    # Step 3: Use multiprocessing to process rows in parallel
    n = A_kernel_distance.shape[0]
    apsp_weight_A = np.zeros((n, n))

    # args = [(i, predecessors, A_euc_distance, n) for i in range(n)]

    # with Pool() as pool:
    #     results = pool.map(process_row, args)

    # # Step 4: Populate the APSP matrix
    # for i, row_distances in results:
    #     apsp_weight_A[i] = row_distances

    # # Step 5: Return the symmetrized APSP matrices
    # apsp_weight_A = np.maximum(apsp_weight_A, apsp_weight_A.T)
    # print(np.argwhere(apsp_weight_A != apsp_weight_A.T))
    # print(np.argwhere(apsp_weight_A != apsp_weight_A.T).shape)
    # print(apsp_weight_A)

    # apsp_weight_B = np.minimum(apsp_weight_B, apsp_weight_B.T)

    # assert np.allclose(apsp_weight_A, apsp_weight_A.T)
    # assert np.allclose(apsp_weight_B, apsp_weight_B.T)

    return apsp_weight_A, apsp_weight_B