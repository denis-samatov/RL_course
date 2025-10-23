import numpy as np


def epsilon_greedy(Q: np.ndarray, s: int, eps: float) -> int:
    if np.random.rand() < eps:
        return int(np.random.randint(Q.shape[1]))
    return int(np.argmax(Q[s]))


def softmax_action(Q: np.ndarray, s: int, tau: float) -> int:
    z = (Q[s] - np.max(Q[s])) / float(tau)
    probs = np.exp(z)
    probs /= probs.sum()
    return int(np.random.choice(len(Q[s]), p=probs))


def random_policy_from_env(env):
    return lambda s: env.action_space.sample()

