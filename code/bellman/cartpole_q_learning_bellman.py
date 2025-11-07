"""CartPole-v1 tabular Q-learning with Bellman optimality updates.

This script discretises the continuous CartPole observations into a finite grid and
trains a tabular epsilon-greedy Q-learning agent. After training it reports metrics,
saves the learned Q-table and can optionally record evaluation videos.
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="`np.bool8` is a deprecated alias for `np.bool_.`"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pygame.pkgdata"
)


@dataclass
class QLearningConfig:
    """Configuration bundle that controls training, evaluation, and discretisation behaviour."""
    num_episodes: int = 4000
    max_steps_per_episode: int = 500
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 2000
    seed: int = 42
    # Discretization bins per state dimension: [x, x_dot, theta, theta_dot]
    bins: Tuple[int, int, int, int] = (8, 8, 16, 16)
    model_output_path: str = "cartpole_q_table.npy"


class Discretizer:
    """Map continuous CartPole observations onto a finite lattice of bins."""

    def __init__(self, bins: Tuple[int, int, int, int]):
        """Pre-compute clipping ranges and bin edges for every state dimension.

        Args:
            bins: Number of bins for (x, x_dot, theta, theta_dot).
        
        Raises:
            ValueError: If any bin count is less than 1.
        """
        # Validate bin counts
        if any(b < 1 for b in bins):
            raise ValueError(
                f"All bin counts must be >= 1, got bins={bins}. "
                "Each dimension needs at least one bin for discretization."
            )
        
        # Reasonable clipping ranges for CartPole-v1
        self.bins = bins
        # cart position
        self.x_range = (-4.8, 4.8)
        # cart velocity (not bounded in env spec) – clip to a sensible range
        self.x_dot_range = (-3.0, 3.0)
        # pole angle (radians) ~ +/- 24 degrees
        self.theta_range = (-0.418, 0.418)
        # pole angular velocity – clip to a sensible range
        self.theta_dot_range = (-3.5, 3.5)

        self.bin_edges = [
            np.linspace(self.x_range[0], self.x_range[1], bins[0] - 1),
            np.linspace(self.x_dot_range[0], self.x_dot_range[1], bins[1] - 1),
            np.linspace(self.theta_range[0], self.theta_range[1], bins[2] - 1),
            np.linspace(self.theta_dot_range[0], self.theta_dot_range[1], bins[3] - 1),
        ]

    def clip_state(self, state: np.ndarray) -> np.ndarray:
        """Clip raw observations to stabilise downstream binning.

        Args:
            state: Continuous observation [x, x_dot, theta, theta_dot].

        Returns:
            Clipped observation with values limited to admissible ranges.
        """
        x, x_dot, theta, theta_dot = state
        x = np.clip(x, *self.x_range)
        x_dot = np.clip(x_dot, *self.x_dot_range)
        theta = np.clip(theta, *self.theta_range)
        theta_dot = np.clip(theta_dot, *self.theta_dot_range)
        return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

    def discretize(self, state: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert a continuous observation into a tuple of bin indices.

        Args:
            state: Continuous observation from the environment.

        Returns:
            Tuple of per-dimension bin indices after clipping and digitisation.
        """
        clipped = self.clip_state(state)
        idxs = [int(np.digitize(clipped[i], self.bin_edges[i])) for i in range(4)]
        # ensure indices are within [0, bins_i - 1]
        idxs = [min(max(0, idx), self.bins[i] - 1) for i, idx in enumerate(idxs)]
        return tuple(idxs)

    def flat_index(self, indices: Tuple[int, int, int, int]) -> int:
        """Flatten multi-dimensional bin indices into a single integer index.

        Args:
            indices: Tuple of bin indices (i_x, i_xdot, i_theta, i_thetadot).

        Returns:
            Linearised index compatible with the tabular Q-table layout.
        """
        b0, b1, b2, b3 = self.bins
        i0, i1, i2, i3 = indices
        return ((i0 * b1 + i1) * b2 + i2) * b3 + i3

    @property
    def num_states(self) -> int:
        """Total number of discrete states produced by the discretizer.

        Returns:
            Number of unique discrete states across all dimensions.
        """
        return int(np.prod(self.bins))


class QLearningAgent:
    """Tabular epsilon-greedy Q-learning agent for discretised CartPole observations."""

    def __init__(self, env: gym.Env, config: QLearningConfig):
        """Initialise buffers, RNG, and helper structures required for learning.

        Args:
            env: Gym environment exposing the CartPole dynamics.
            config: Hyperparameters and discretisation settings.
        """
        self.env = env
        self.config = config
        self.discretizer = Discretizer(config.bins)
        self.num_actions = env.action_space.n
        self.q_table = np.zeros((self.discretizer.num_states, self.num_actions), dtype=np.float32)
        self.rng = np.random.default_rng(config.seed)

    def _epsilon(self, episode: int) -> float:
        """Return exploration rate for the given episode using linear decay.

        Args:
            episode: Zero-based training episode index.

        Returns:
            Exploration probability epsilon for epsilon-greedy action selection.
        """
        if episode >= self.config.epsilon_decay_episodes:
            return self.config.epsilon_end
        frac = episode / max(1, self.config.epsilon_decay_episodes)
        return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

    def _state_to_index(self, obs: np.ndarray) -> int:
        """Map a continuous observation to its discrete state index.

        Args:
            obs: Continuous observation returned by the environment.

        Returns:
            Integer index into the flattened Q-table.
        """
        idxs = self.discretizer.discretize(obs)
        return self.discretizer.flat_index(idxs)

    def act(self, state_idx: int, epsilon: float) -> int:
        """Choose an action via epsilon-greedy policy with respect to the current Q-table.

        Args:
            state_idx: Discrete index representing the current state.
            epsilon: Exploration probability for the decision.

        Returns:
            Selected action index from the discrete action space.
        """
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.num_actions))
        return int(np.argmax(self.q_table[state_idx]))
    
    def act_greedy(self, state_idx: int) -> int:
        """Choose the greedy action (no exploration) for evaluation.
        
        Args:
            state_idx: Discrete index representing the current state.
            
        Returns:
            Action with highest Q-value for the given state.
        """
        return int(np.argmax(self.q_table[state_idx]))

    def bellman_update(self, s_idx: int, a: int, r: float, s_next_idx: int, terminated: bool) -> None:
        """Apply a single Q-learning Bellman update for state-action pair (s, a).
        
        Args:
            s_idx: Current state index.
            a: Action taken.
            r: Reward received.
            s_next_idx: Next state index.
            terminated: True if episode ended naturally (not by time limit).
                       Bootstrap should be kept when episode was merely truncated.
        """
        best_next = 0.0 if terminated else float(np.max(self.q_table[s_next_idx]))
        td_target = r + self.config.discount_factor * best_next
        td_error = td_target - float(self.q_table[s_idx, a])
        self.q_table[s_idx, a] += self.config.learning_rate * td_error

    def train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run Q-learning for the configured number of episodes."""
        episode_returns = np.zeros(self.config.num_episodes, dtype=np.float32)
        episode_lengths = np.zeros(self.config.num_episodes, dtype=np.int32)

        for ep in tqdm(range(self.config.num_episodes), desc="Training episodes", unit="ep"):
            obs, _ = self.env.reset(seed=self.config.seed + ep)
            s_idx = self._state_to_index(obs)
            epsilon = self._epsilon(ep)
            total_reward = 0.0
            done = False

            for t in range(self.config.max_steps_per_episode):
                action = self.act(s_idx, epsilon)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                s_next_idx = self._state_to_index(next_obs)

                # Only zero bootstrap on actual termination, not time-limit truncation
                self.bellman_update(s_idx, action, float(reward), s_next_idx, terminated)

                total_reward += float(reward)
                s_idx = s_next_idx
                done = terminated or truncated

                if done:
                    episode_returns[ep] = total_reward
                    episode_lengths[ep] = t + 1
                    break

            if not done:
                episode_returns[ep] = total_reward
                episode_lengths[ep] = self.config.max_steps_per_episode
        return episode_returns, episode_lengths

    def evaluate(self, num_episodes: int = 10, seed_offset: int = 10_000) -> Tuple[float, float]:
        """Evaluate a greedy policy and return mean episode return and length.
        
        Args:
            num_episodes: Number of episodes to evaluate.
            seed_offset: Offset added to base seed for reproducible evaluation.
            
        Returns:
            Tuple of (mean_return, mean_length) over evaluation episodes.
        """
        returns = []
        lengths = []
        for ep in range(num_episodes):
            eval_seed = self.config.seed + seed_offset + ep
            total_reward, episode_length = self._run_greedy_episode(self.env, eval_seed)
            returns.append(total_reward)
            lengths.append(episode_length)
            print(f"  Eval episode {ep+1}/{num_episodes} (seed={eval_seed}): return={total_reward:.1f}, length={episode_length}")
        return float(np.mean(returns)), float(np.mean(lengths))
    
    def _run_greedy_episode(self, env: gym.Env, seed: int) -> Tuple[float, int]:
        """Roll out a single greedy episode (shared helper for evaluation and video recording).
        
        Args:
            env: Gym environment to run the episode in.
            seed: Random seed for this episode.
            
        Returns:
            Tuple of (total_reward, episode_length).
        """
        obs, _ = env.reset(seed=seed)
        s_idx = self._state_to_index(obs)
        total_reward = 0.0
        for t in range(self.config.max_steps_per_episode):
            action = self.act_greedy(s_idx)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            s_idx = self._state_to_index(next_obs)
            if terminated or truncated:
                return total_reward, t + 1
        return total_reward, self.config.max_steps_per_episode

    def save(self, path) -> None:
        """Persist the learned Q-table to disk, creating directories if required.
        
        Args:
            path: File path (str or PathLike) where Q-table will be saved.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), self.q_table)

    def load(self, path) -> None:
        """Load a previously saved Q-table after validating its structure.
        
        Args:
            path: File path (str or PathLike) to load Q-table from.
            
        Raises:
            ValueError: If loaded Q-table shape doesn't match expected dimensions.
        """
        arr = np.load(str(path), allow_pickle=False)
        expected_shape = (self.discretizer.num_states, self.num_actions)
        if arr.shape != expected_shape:
            raise ValueError(f"Q-table shape mismatch: got {arr.shape}, expected {expected_shape}")
        self.q_table = arr.astype(np.float32, copy=False)





def train_and_evaluate(args: argparse.Namespace) -> None:
    """Train the agent, report metrics, save artefacts, and optionally record evaluation videos."""
    env = gym.make("CartPole-v1")

    try:
        config = QLearningConfig(
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            learning_rate=args.lr,
            discount_factor=args.gamma,
            epsilon_start=args.eps_start,
            epsilon_end=args.eps_end,
            epsilon_decay_episodes=args.eps_decay_episodes,
            seed=args.seed,
            bins=(args.bins_x, args.bins_xdot, args.bins_theta, args.bins_thetadot),
            model_output_path=args.output,
        )

        agent = QLearningAgent(env, config)
        returns, lengths = agent.train()

        mean_last_100 = float(np.mean(returns[-100:])) if len(returns) >= 100 else float(np.mean(returns))
        eval_return, eval_length = agent.evaluate(num_episodes=args.eval_episodes)

        agent.save(config.model_output_path)

        print(f"Training complete. Mean return (last 100): {mean_last_100:.2f}")
        print(f"Evaluation over {args.eval_episodes} episodes -> mean return: {eval_return:.2f}, mean length: {eval_length:.1f}")
        print(f"Q-table saved to: {os.path.abspath(config.model_output_path)}")

        # Optional: record evaluation videos
        if args.video_dir:
            base_dir = Path(args.video_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            name_prefix = f"qlearn_eval_after_{config.num_episodes}"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = base_dir / f"{name_prefix}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)

            video_env = gym.make("CartPole-v1", render_mode="rgb_array")
            video_env = RecordVideo(
                video_env,
                video_folder=str(run_dir),
                episode_trigger=lambda ep_id: True,
                name_prefix=name_prefix,
            )
            try:
                print(
                    f"Recording {args.video_episodes} evaluation episode(s) to: {run_dir.resolve()}"
                )
                rec_returns = []
                rec_lengths = []
                for ep in range(args.video_episodes):
                    video_seed = config.seed + 20_000 + ep
                    ret, ln = agent._run_greedy_episode(video_env, video_seed)
                    rec_returns.append(ret)
                    rec_lengths.append(ln)
                    print(f"  Video episode {ep+1}/{args.video_episodes} (seed={video_seed}): return={ret:.1f}, length={ln}")
                if rec_returns:
                    print(
                        f"Recorded episodes -> mean return: {float(np.mean(rec_returns)):.2f}, "
                        f"mean length: {float(np.mean(rec_lengths)):.1f}"
                    )
            finally:
                video_env.close()
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments controlling training, evaluation, and output options."""
    parser = argparse.ArgumentParser(description="CartPole-v1 Q-learning with Bellman optimality updates (discretized)")
    parser.add_argument("--episodes", type=int, default=4000, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Starting epsilon for epsilon-greedy")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon after decay")
    parser.add_argument("--eps-decay-episodes", type=int, default=2000, help="Episodes over which to linearly decay epsilon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bins-x", type=int, default=8, help="Number of bins for cart position x")
    parser.add_argument("--bins-xdot", type=int, default=8, help="Number of bins for cart velocity x_dot")
    parser.add_argument("--bins-theta", type=int, default=16, help="Number of bins for pole angle theta")
    parser.add_argument("--bins-thetadot", type=int, default=16, help="Number of bins for pole angular velocity theta_dot")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes after training")
    parser.add_argument("--output", type=str, default="cartpole_q_table.npy", help="Path to save the learned Q-table")
    parser.add_argument("--video-dir", type=str, default="", help="Directory to save evaluation videos (empty string disables video recording)")
    parser.add_argument("--video-episodes", type=int, default=1, help="How many evaluation episodes to record to video")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args)
