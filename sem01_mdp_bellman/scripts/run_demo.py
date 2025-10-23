"""
Seminar 1 demo runner (refactored to use rlcourse.sem01 modules).

Usage examples:
  python run_demo.py --env cartpole --mode random
  python run_demo.py --env cartpole --mode mc
  python run_demo.py --env cartpole --mode td0
  python run_demo.py --env frozenlake --mode ql
"""

import argparse
import numpy as np
import os, sys

# Add ../../src to sys.path for local package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from rlcourse.sem01.envs import make_cartpole, make_frozenlake, SEED
from rlcourse.sem01.algorithms import (
    run_random,
    eval_value_mc,
    q_learning,
    td0_value_learning,
)
from rlcourse.sem01.discretization import discretize


def main():
    parser = argparse.ArgumentParser(description="Seminar 1 demos")
    parser.add_argument("--env", choices=["cartpole", "frozenlake"], default="cartpole")
    parser.add_argument("--mode", choices=["random", "mc", "td0", "ql"], default="random")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    print('SEED:', SEED)
    print('numpy:', np.__version__)

    if args.env == "cartpole":
        env = make_cartpole()
        if args.mode == "random":
            returns = run_random(env, episodes=args.episodes)
            print('Returns per episode:', [float(x) for x in returns])
            print(f'Mean return over {args.episodes} episodes: {np.mean(returns):.2f}')
        elif args.mode == "mc":
            mean, std, _, _ = eval_value_mc(env, lambda s: env.action_space.sample(), episodes=max(100, args.episodes))
            print(f"MC estimate of V^pi (start-dist): mean={mean:.2f} ± {std:.2f}")
        elif args.mode == "td0":
            V, _ = td0_value_learning(env, episodes=max(200, args.episodes))
            vals = []
            for _ in range(10):
                obs, _ = env.reset()
                vals.append(V[discretize(obs)])
            print(f'TD(0) average V over start states (approx): {np.mean(vals):.2f}')
        else:
            raise SystemExit("Mode 'ql' requires --env frozenlake")
        env.close()

    else:  # frozenlake
        env = make_frozenlake()
        if args.mode != "ql":
            print("For FrozenLake, run with --mode ql (Q-learning) for a meaningful demo.")
        Q, returns, _ = q_learning(env, episodes=max(1000, args.episodes))
        print(f"Avg return over last 100 eps: {np.mean(returns[-100:]):.3f}")


if __name__ == "__main__":
    main()
