"""
Two-state MDP demo (analytics + practice).

MDP definition:
  S = {0=s1, 1=s2}, A = {0,1}
  - a=0: s1->s2 with r=+1; s2->s2 with r=0 (absorbing)
  - a=1: s1->s1 with r=0;  s2->s2 with r=0 (absorbing)

Optimal: in s1 take a=0, in s2 arbitrary. Then V*(s1)=1, V*(s2)=0 for any gamma in [0,1).

This script:
  1) Computes analytic V*, Q* and checks basic identities.
  2) Simulates episodes with the optimal greedy policy and compares average return to V*(s1).
  3) Demonstrates Bellman value iteration convergence and residual decay.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


S1, S2 = 0, 1
A0, A1 = 0, 1


@dataclass
class TwoStateMDP:
    gamma: float = 0.99

    def step(self, s: int, a: int):
        if s == S1:
            if a == A0:  # s1 --a0--> s2, r=+1
                return S2, 1.0, True, False
            else:        # s1 --a1--> s1, r=0
                return S1, 0.0, False, False
        else:            # s2 absorbing, r=0
            return S2, 0.0, True, False

    def V_star(self):
        # From definition: V*(s2)=0; V*(s1)=max{1 + gamma*V*(s2), 0 + gamma*V*(s1)} = 1
        return np.array([1.0, 0.0], dtype=float)

    def Q_star(self):
        V = self.V_star()
        # Q*(s1,0)=1+gamma*V(s2)=1; Q*(s1,1)=0+gamma*V(s1)=0; at s2 all zeros (absorbing)
        Q = np.zeros((2, 2), dtype=float)
        Q[S1, A0] = 1.0
        Q[S1, A1] = 0.0
        Q[S2, A0] = 0.0
        Q[S2, A1] = 0.0
        return Q


def simulate(env: TwoStateMDP, policy, episodes=1000, seed=42):
    rs = np.random.RandomState(seed)
    totals = []
    for _ in range(episodes):
        # start always at s1 for clarity
        s = S1
        done = trunc = False
        G = 0.0
        t = 0
        while not (done or trunc):
            a = policy(s)
            s_next, r, done, trunc = env.step(s, a)
            G += (env.gamma ** t) * r
            s = s_next
            t += 1
            # small safety to avoid infinite loops at gamma close to 1 when a=1 at s1
            if t > 500:
                trunc = True
        totals.append(G)
    return float(np.mean(totals)), float(np.std(totals))


def greedy_star_policy(s):
    return A0 if s == S1 else A0


def value_iteration(env: TwoStateMDP, iters=10):
    V = np.zeros(2, dtype=float)
    residuals = []
    for _ in range(iters):
        V_new = np.zeros_like(V)
        # s1: max over a
        v_a0 = 1.0 + env.gamma * 0.0   # to s2
        v_a1 = 0.0 + env.gamma * V[S1] # stay in s1
        V_new[S1] = max(v_a0, v_a1)
        # s2: absorbing 0
        V_new[S2] = 0.0
        residuals.append(float(np.max(np.abs(V_new - V))))
        V = V_new
    return V, np.array(residuals, dtype=float)


def main():
    env = TwoStateMDP(gamma=0.99)
    V_star = env.V_star()
    Q_star = env.Q_star()

    # 1) analytics
    assert np.allclose(V_star, [1.0, 0.0])
    assert np.isclose(Q_star[S1, A0], 1.0)
    assert np.isclose(Q_star[S1, A1], 0.0)

    # 2) simulation under optimal greedy
    mean, std = simulate(env, greedy_star_policy, episodes=500)
    print(f"Simulated return (start s1): mean={mean:.3f} ± {std:.3f}")
    assert abs(mean - V_star[S1]) < 0.1, "Simulation should match analytic V*(s1)≈1"

    # 3) residual decay under value iteration
    V_vi, residuals = value_iteration(env, iters=8)
    print("Value iteration residuals:", residuals.tolist())
    print("V after VI:", V_vi.tolist())
    assert residuals[-1] <= residuals[0] + 1e-9, "Residual should not grow"
    print("✓ Two-state MDP demo OK")


if __name__ == "__main__":
    main()

