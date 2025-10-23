import numpy as np

# Default CartPole discretization
BIN_COUNTS = (6, 6, 6, 6)
bounds = np.array([
    [-4.8, 4.8],
    [-3.0, 3.0],
    [-0.418, 0.418],
    [-3.5, 3.5],
], dtype=float)


def make_edges(bounds_arr=bounds, bin_counts=BIN_COUNTS):
    return [np.linspace(lo, hi, n + 1) for (lo, hi), n in zip(bounds_arr, bin_counts)]


EDGES = make_edges(bounds, BIN_COUNTS)


def discretize(obs, edges=EDGES, bin_counts=BIN_COUNTS) -> int:
    """Discretizes a continuous observation into a single integer."""
    idxs = []
    for i, e in enumerate(edges):
        n = len(e) - 1
        val = float(np.clip(obs[i], e[0], e[-1]))
        b = int(np.digitize(val, e) - 1)
        b = max(0, min(n - 1, b))
        idxs.append(b)

    if not idxs:
        return 0

    # N-dimensional to 1-dimensional mapping
    # This is equivalent to np.ravel_multi_index(tuple(idxs), bin_counts)
    state = idxs[0]
    for i in range(1, len(idxs)):
        state = state * bin_counts[i] + idxs[i]
    return state


def state_space_size(bin_counts=BIN_COUNTS) -> int:
    return int(np.prod(bin_counts))

