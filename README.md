# EmbedOR
Official Implementation of the paper:  [*"EmbedOR: Provable Cluster-Preserving Visualizations with Curvature-Based Stochastic Neighbor Embeddings"*](https://arxiv.org/pdf/2509.03703)

## Getting Started

### Installation

Clone the repository:

``` bash
git clone https://github.com/kathyzxu/embedOR.git
cd embedOR
```

Create and activate a virtual environment:

``` bash
python3 -m venv env
source env/bin/activate
export PYTHONPATH="$PWD"
```

Upgrade pip and install dependencies:

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` tracks the third-party packages imported from `src/`, including the optional biological-data and visualization helpers in `src/data/data.py` and `src/vis.py`.

### Quick Start

The repository currently exposes a Python API rather than a checked-in CLI script. A minimal end-to-end example is:

``` python
from src.data.data import moons
from src.embedor import EmbedOR
from src.plotting import plot_data_2D

dataset = moons(n_points=1000, noise=0.05)
X = dataset["data"]
labels = dataset["cluster"]

embedding = EmbedOR(
	nng_params={"mode": "nbrs", "n_neighbors": 15},
	edge_weight="orc",
	layout="torch",
	seed=42,
).fit_transform(X)

plot_data_2D(embedding, color=labels, title="EmbedOR embedding")
```

### Project Layout

- `src/embedor.py` contains the main `EmbedOR` estimator.
- `src/data/data.py` contains synthetic data generators and dataset loaders.
- `src/plotting.py` contains lightweight plotting helpers for embeddings and graphs.
- `src/vis.py` contains additional visualization and RNA-velocity utilities.
- `src/utils/` contains graph construction, curvature, layout, and timing helpers.

If you want a scriptable workflow, add a small runner that imports `EmbedOR` from `src.embedor` and passes either a NumPy array or a precomputed adjacency matrix into `fit`/`fit_transform`.
