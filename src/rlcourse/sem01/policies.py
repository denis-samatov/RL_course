import numpy as np


def epsilon_greedy(Q: np.ndarray, s: int, eps: float) -> int:
    """Выбирает действие с использованием ε-жадной стратегии.

    Args:
        Q (np.ndarray): Q-таблица (размер: [S, A]).
        s (int): Текущее состояние.
        eps (float): Вероятность выбора случайного действия (ε).

    Returns:
        int: Выбранное действие.
    """
    if np.random.rand() < eps:
        return int(np.random.randint(Q.shape[1]))
    return int(np.argmax(Q[s]))


def softmax_action(Q: np.ndarray, s: int, tau: float) -> int:
    """Выбирает действие с использованием softmax (Boltzmann) стратегии.

    Args:
        Q (np.ndarray): Q-таблица (размер: [S, A]).
        s (int): Текущее состояние.
        tau (float): Температурный параметр. Высокий tau -> более случайный
            выбор, низкий tau -> более жадный.

    Returns:
        int: Выбранное действие.
    """
    z = (Q[s] - np.max(Q[s])) / float(tau)
    probs = np.exp(z)
    probs /= probs.sum()
    return int(np.random.choice(len(Q[s]), p=probs))


def random_policy_from_env(env):
    """Создает случайную политику на основе пространства действий среды.

    Args:
        env: Среда Gymnasium.

    Returns:
        callable: Функция политики, которая принимает состояние и возвращает
        случайное действие.
    """
    return lambda s: env.action_space.sample()

