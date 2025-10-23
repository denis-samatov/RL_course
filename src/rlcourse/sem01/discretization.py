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
    """Создает границы бинов для дискретизации непрерывного пространства.

    Args:
        bounds_arr (np.ndarray): Массив с границами для каждой размерности
            состояния. Формат: [[min1, max1], [min2, max2], ...].
        bin_counts (tuple): Количество бинов для каждой размерности.

    Returns:
        list: Список массивов, где каждый массив содержит границы бинов
        для соответствующей размерности.
    """
    return [np.linspace(lo, hi, n + 1) for (lo, hi), n in zip(bounds_arr, bin_counts)]


EDGES = make_edges(bounds, BIN_COUNTS)


def discretize(obs, edges=EDGES, bin_counts=BIN_COUNTS) -> int:
    """Преобразует непрерывное наблюдение в дискретное состояние.

    Args:
        obs (np.ndarray): Непрерывное наблюдение (состояние).
        edges (list): Границы бинов для каждой размерности.
        bin_counts (tuple): Количество бинов для каждой размерности.

    Returns:
        int: Дискретный индекс состояния.
    """
    idxs = []
    for i, e in enumerate(edges):
        n = len(e) - 1
        val = float(np.clip(obs[i], e[0], e[-1]))
        b = int(np.digitize(val, e) - 1)
        b = max(0, min(n - 1, b))
        idxs.append(b)
    b0, b1, b2, b3 = idxs
    return ((b0 * bin_counts[1] + b1) * bin_counts[2] + b2) * bin_counts[3] + b3


def state_space_size(bin_counts=BIN_COUNTS) -> int:
    """Вычисляет общий размер дискретного пространства состояний.

    Args:
        bin_counts (tuple): Количество бинов для каждой размерности.

    Returns:
        int: Общее количество дискретных состояний.
    """
    return int(np.prod(bin_counts))

