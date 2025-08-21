from src.data.data import *
from src.embedor import *
from src.plotting import *
import matplotlib.pyplot as plt
import umap
from src.utils.umap_exact import UMAP_exact
import numpy as np
from sklearn.manifold import TSNE
import time

REPO_ROOT = os.getenv('PYTHONPATH')
save_path = f'{REPO_ROOT}/outputs/runtime'
os.makedirs(save_path, exist_ok=True)
# add datetime
save_path = f'{save_path}/{time.strftime("%Y%m%d-%H%M%S")}'
os.makedirs(save_path, exist_ok=True)

embedor_times = []
embedor_times_landmark = []
umap_times = []
umap_times_approx = []
tsne_times = []
tsne_times_approx = []

cutoff_embedor = False
cutoff_embedor_landmark = False
cutoff_umap = False
cutoff_umap_approx = False
cutoff_tsne = False
cutoff_tsne_approx = False
cutoff_time = 1e3

embedor_npt_idx = 0
embedor_landmark_npt_idx = 0
umap_npt_idx = 0
umap_approx_npt_idx = 0
tsne_npt_idx = 0
tsne_approx_npt_idx = 0

n_points_array = [1000, 2000, 5000, 10000, 15000, 25000]

noise = 0.1
noise_thresh = None

for idx, n_points in enumerate(n_points_array):
    print(f"Running for {n_points} points...")
    return_dict = concentric_circles(n_points=n_points, factor=0.4, noise=noise, noise_thresh=noise_thresh)

    if not cutoff_embedor:
        time_start = time.time()
        embedor = EmbedOR()
        _ = embedor.fit_transform(return_dict['data'])
        time_end = time.time()
        embedor_time = time_end - time_start
        embedor_times.append(embedor_time)
        if embedor_time > cutoff_time:
            cutoff_embedor = True
            print(f"EmbedOR took too long: {embedor_time:.2f} seconds, stopping further tests.")
        embedor_npt_idx +=1

    if not cutoff_embedor_landmark:
        time_start = time.time()
        embedor = EmbedOR(landmark_selection='random', n_landmarks=50, subsample=True, subsample_factor=0.2, approx_affinities=True, edge_weight='frc')
        _ = embedor.fit_transform(return_dict['data'])
        time_end = time.time()
        embedor_time_landmark = time_end - time_start
        embedor_times_landmark.append(embedor_time_landmark)
        if embedor_time_landmark > cutoff_time:
            cutoff_embedor_landmark = True
            print(f"EmbedOR with landmark selection took too long: {embedor_time_landmark:.2f} seconds, stopping further tests.")
        embedor_landmark_npt_idx += 1

    if not cutoff_umap:
        time_start = time.time()
        umap_model = UMAP_exact(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', verbose=True, n_epochs=300)
        _ = umap_model.fit_transform(return_dict['data'])
        time_end = time.time()
        umap_time = time_end - time_start
        umap_times.append(umap_time)
        if umap_time > cutoff_time:
            cutoff_umap = True
            print(f"UMAP took too long: {umap_time:.2f} seconds, stopping further tests.")
        umap_npt_idx += 1

    if not cutoff_umap_approx:
        time_start = time.time()
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', verbose=True, n_epochs=300)
        _ = umap_model.fit_transform(return_dict['data'])
        time_end = time.time()
        umap_time_approx = time_end - time_start
        umap_times_approx.append(umap_time_approx)
        if umap_time_approx > cutoff_time:
            cutoff_umap_approx = True
            print(f"UMAP (Approx) took too long: {umap_time_approx:.2f} seconds, stopping further tests.")
        umap_approx_npt_idx += 1

    if not cutoff_tsne:
        time_start = time.time()
        tsne_model = TSNE(method='exact', n_jobs=-1, verbose=1)
        _ = tsne_model.fit_transform(return_dict['data'])
        time_end = time.time()
        tsne_time = time_end - time_start
        tsne_times.append(tsne_time)
        if tsne_time > cutoff_time:
            cutoff_tsne = True
            print(f"t-SNE (Exact) took too long: {tsne_time:.2f} seconds, stopping further tests.")
        tsne_npt_idx += 1

    if not cutoff_tsne_approx:
        time_start = time.time()
        tsne_model = TSNE()
        _ = tsne_model.fit_transform(return_dict['data'])
        time_end = time.time()
        tsne_time_approx = time_end - time_start
        tsne_times_approx.append(tsne_time_approx)
        if tsne_time_approx > cutoff_time:
            cutoff_tsne_approx = True
            print(f"t-SNE (Approx) took too long: {tsne_time_approx:.2f} seconds, stopping further tests.")
        tsne_approx_npt_idx += 1

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(n_points_array[:embedor_npt_idx], embedor_times, label='EmbedOR', marker='o')
    plt.plot(n_points_array[:embedor_landmark_npt_idx], embedor_times_landmark, label='EmbedOR (approx)', marker='o')
    plt.plot(n_points_array[:umap_npt_idx], umap_times, label='UMAP', marker='o')
    plt.plot(n_points_array[:umap_approx_npt_idx], umap_times_approx, label='UMAP (Approx)', marker='o')
    plt.plot(n_points_array[:tsne_npt_idx], tsne_times, label='t-SNE (Exact)', marker='o')
    plt.plot(n_points_array[:tsne_approx_npt_idx], tsne_times_approx, label='t-SNE (Approx)', marker='o')
    plt.xlabel('Number of Points')
    plt.ylabel('Time (seconds)')
    # y log scale
    plt.yscale('log')
    # add legend
    plt.legend()
    plt.savefig(f'{save_path}/runtime_{n_points}_points.png', dpi=1200)