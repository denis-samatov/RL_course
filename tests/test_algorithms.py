import numpy as np
import pytest

from rlcourse.sem01.algorithms import q_learning

class MockEnv:
    def __init__(self, n_states, n_actions):
        self.observation_space = self
        self.action_space = self
        self.n = n_states
        self.n_actions = n_actions
        self.reset_count = 0
        self.step_count = 0

    def seed(self, seed):
        pass

    def reset(self, seed=None):
        self.reset_count += 1
        self.step_count = 0
        return 0, {}

    def step(self, action):
        self.step_count += 1
        if self.step_count > 10:
            return self.step_count, 1, True, False, {}
        return self.step_count, 1, False, False, {}


def test_q_learning_return():
    env = MockEnv(20, 5)
    _, returns, _ = q_learning(env, episodes=1, gamma=0.99)
    assert np.isclose(returns[0], np.sum([0.99 ** i for i in range(11)])), "Return should be discounted"
