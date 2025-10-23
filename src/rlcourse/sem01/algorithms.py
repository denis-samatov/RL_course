import numpy as np

from .discretization import make_edges, discretize as cp_discretize, state_space_size, bounds as CP_BOUNDS

SEED = 42


def smooth(x, k=5):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode='valid')


def moving_avg(x, k=100):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode='valid')


def run_random(env, episodes=20, gamma=0.99, seed=SEED,
               log_first=0, log_steps=0, return_details=False):
    env.action_space.seed(seed)
    rs = np.random.RandomState(seed)
    returns = []
    details = [] if return_details else None
    for ep in range(episodes):
        s, _ = env.reset(seed=int(rs.randint(0, 1_000_000)))
        done = trunc = False
        G = 0.0
        t = 0
        ep_trans = []
        while not (done or trunc):
            a = env.action_space.sample()
            s_next, r, done, trunc, _ = env.step(a)
            G += (gamma ** t) * r
            if return_details and ep < log_first and t < log_steps:
                ep_trans.append((s.copy(), int(a), float(r), s_next.copy()))
            s = s_next
            t += 1
        returns.append(G)
        if return_details:
            details.append(ep_trans)
    returns = np.asarray(returns, dtype=float)
    if return_details:
        return returns, details
    return returns


def eval_value_mc(env, policy, episodes=200, seed=SEED, gamma=0.99):
    rs = np.random.RandomState(seed)
    totals = []
    for ep in range(episodes):
        s, _ = env.reset(seed=int(rs.randint(0, 1_000_000)))
        done = trunc = False
        G = 0.0
        t = 0
        while not (done or trunc):
            a = policy(s)
            s, r, done, trunc, _ = env.step(a)
            G += (gamma ** t) * r
            t += 1
        totals.append(G)
    totals = np.asarray(totals, dtype=float)
    mean, std = float(np.mean(totals)), float(np.std(totals))
    ci95 = 1.96 * std / np.sqrt(len(totals))
    return mean, std, ci95, totals


def q_learning(env, episodes=2000, alpha=0.7, gamma=0.99, epsilon=0.2, seed=SEED):
    rs = np.random.RandomState(seed)
    S, A = env.observation_space.n, env.action_space.n
    Q = np.zeros((S, A), dtype=np.float32)
    returns, delta_norm = [], []
    max_abs_td = 0.0
    for ep in range(episodes):
        s, _ = env.reset(seed=int(rs.randint(0, 1_000_000)))
        done = trunc = False
        G = 0.0
        t = 0
        while not (done or trunc):
            if rs.rand() < epsilon:
                a = int(rs.randint(A))
            else:
                a = int(np.argmax(Q[s]))
            s_next, r, done, trunc, _ = env.step(a)
            td_target = r + (0.0 if (done or trunc) else gamma * np.max(Q[s_next]))
            td_error = td_target - Q[s, a]
            Q[s, a] += alpha * td_error
            max_abs_td = max(max_abs_td, float(abs(td_error)))
            G += (gamma ** t) * r
            s = s_next
            t += 1
        returns.append(G)
        if (ep + 1) % 50 == 0:
            delta_norm.append(max_abs_td)
            max_abs_td = 0.0
    return Q, np.array(returns, dtype=float), np.array(delta_norm, dtype=float)


def eval_greedy(env, Q, episodes=200, seed=SEED):
    rs = np.random.RandomState(seed)
    wins = 0
    for ep in range(episodes):
        s, _ = env.reset(seed=int(rs.randint(0, 1_000_000)))
        done = trunc = False
        while not (done or trunc):
            a = int(np.argmax(Q[s]))
            s, r, done, trunc, _ = env.step(a)
            if done and r > 0:
                wins += 1
    return wins / episodes


def td0_value_learning(env, episodes=300, alpha=0.1, gamma=0.99, seed=SEED,
                       bin_counts=(6, 6, 6, 6), bounds=CP_BOUNDS):
    rs = np.random.RandomState(seed)
    edges = make_edges(bounds, bin_counts)
    state_space = int(np.prod(bin_counts))
    V = np.zeros(state_space, dtype=np.float32)
    mse_hist = []
    for ep in range(episodes):
        s_cont, _ = env.reset(seed=int(rs.randint(0, 1_000_000)))
        s = cp_discretize(s_cont, edges=edges, bin_counts=bin_counts)
        done = trunc = False
        deltas = []
        while not (done or trunc):
            a = env.action_space.sample()
            s_next_cont, r, done, trunc, _ = env.step(a)
            s_next = cp_discretize(s_next_cont, edges=edges, bin_counts=bin_counts)
            target = r + (0.0 if (done or trunc) else gamma * V[s_next])
            d = target - V[s]
            V[s] += alpha * d
            deltas.append(float(d))
            s = s_next
        if deltas:
            mse_hist.append(float(np.mean(np.square(deltas))))
    return V, np.asarray(mse_hist, dtype=float)
