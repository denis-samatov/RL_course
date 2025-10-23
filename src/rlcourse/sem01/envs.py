import random
import numpy as np
import gymnasium as gym

SEED = 42


def make_cartpole(seed: int = SEED, **kwargs):
    """Создает и инициализирует среду CartPole-v1.

    Args:
        seed (int): Зерно для генераторов случайных чисел.
        **kwargs: Дополнительные аргументы для `gym.make`.

    Returns:
        gym.Env: Инициализированная среда.
    """
    env = gym.make('CartPole-v1', **kwargs)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return env


def make_frozenlake(seed: int = SEED, **kwargs):
    """Создает и инициализирует среду FrozenLake-v1.

    Args:
        seed (int): Зерно для генераторов случайных чисел.
        **kwargs: Дополнительные аргументы для `gym.make`.

    Returns:
        gym.Env: Инициализированная среда.
    """
    env = gym.make('FrozenLake-v1', **kwargs)
    np.random.seed(seed)
    random.seed(seed)
    return env

