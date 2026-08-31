"""
Deep Q-Network (DQN) for LunarLander-v2.

A classic DQN implementation with:
- an Experience Replay Buffer
- a Target Network
- ε-greedy exploration
- gradient clipping
"""

import argparse
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Named tuple for storing transitions
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminated'))


@dataclass
class DQNConfig:
    """Configuration for DQN training."""
    total_steps: int = 50000
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997  # Exponential decay
    buffer_size: int = 50000
    batch_size: int = 64
    target_update: int = 1000  # Target network update frequency
    warmup_steps: int = 2000  # Steps before training starts
    hidden_dims: Tuple[int, ...] = (128, 128)
    gradient_clip: float = 10.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    """Experience Replay Buffer for DQN."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition):
        """Adds a transition to the buffer."""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Transition:
        """Samples a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)

        # Convert to tensors
        states = torch.FloatTensor(np.array([t.state for t in batch]))
        actions = torch.LongTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
        terminated = torch.FloatTensor([t.terminated for t in batch])

        return Transition(states, actions, rewards, next_states, terminated)

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network for approximating the Q-function."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (128, 128)):
        super().__init__()

        # Initialize the layers
        dims = (observation_dim,) + hidden_dims
        layers = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            lin = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.extend([lin, nn.ReLU()])

        self.backbone = nn.Sequential(*layers)

        # Output layer for Q-values
        self.head = nn.Linear(hidden_dims[-1], action_dim)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.backbone(x.float())
        return self.head(features)


class DQNAgent:
    """A DQN agent for training and interacting with the environment."""

    def __init__(self, env: gym.Env, config: DQNConfig):
        self.env = env
        self.config = config

        # Networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Check that the action space is discrete
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("LunarLander-v2 has a discrete action space. This script expects gym.spaces.Discrete.")

        self.policy_net = DQN(obs_dim, action_dim, config.hidden_dims).to(config.device)
        self.target_net = DQN(obs_dim, action_dim, config.hidden_dims).to(config.device)

        # Initialize the target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)

        # Replay buffer
        self.buffer = ReplayBuffer(config.buffer_size)

        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.epsilons = []

    def epsilon_schedule(self, step: int) -> float:
        """Exponential decay schedule for epsilon."""
        epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start * (self.config.epsilon_decay ** step)
        )
        return epsilon

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Chooses an action via the ε-greedy policy."""
        if random.random() < epsilon:
            return self.env.action_space.sample()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def compute_td_target(self, batch: Transition) -> torch.Tensor:
        """Computes the TD target: r + γ * max_a Q_target(s', a)."""
        with torch.no_grad():
            next_q_values = self.target_net(batch.next_state.to(self.config.device))
            max_next_q = next_q_values.max(dim=1)[0]

            # Only bootstrap if not terminated
            td_target = batch.reward.to(self.config.device) + \
                       self.config.gamma * (1.0 - batch.terminated.to(self.config.device)) * max_next_q

        return td_target

    def update(self, batch: Transition) -> float:
        """Performs a single training step."""
        # Current Q-values for the chosen actions
        state_action_values = self.policy_net(batch.state.to(self.config.device)).gather(
            1, batch.action.unsqueeze(1).to(self.config.device)
        ).squeeze(1)

        # TD target
        td_target = self.compute_td_target(batch)

        # Loss (Smooth L1)
        loss = F.smooth_l1_loss(state_action_values, td_target)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        return loss.item()

    def train(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """The main training loop."""
        self.policy_net.train()

        state, _ = self.env.reset(seed=self.config.seed)
        state = np.asarray(state, dtype=np.float32)

        episode_reward = 0.0
        episode_length = 0

        pbar = tqdm(range(1, self.config.total_steps + 1), desc="DQN Training", unit="step")

        for step in pbar:
            # Epsilon for the current step
            epsilon = self.epsilon_schedule(step - 1)

            # Choose an action
            action = self.select_action(state, epsilon)

            # Step in the environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = np.asarray(next_state, dtype=np.float32)
            done = terminated or truncated

            # Store into the replay buffer
            self.buffer.push(Transition(
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                terminated=float(terminated)  # Only terminated, not truncated
            ))

            # Train (after warmup)
            if step >= self.config.warmup_steps and len(self.buffer) >= self.config.batch_size:
                batch = self.buffer.sample(self.config.batch_size)
                loss = self.update(batch)
                self.losses.append(loss)

            # Update the target network
            if step % self.config.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Update metrics
            episode_reward += float(reward)
            episode_length += 1
            state = next_state

            # End of episode
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.epsilons.append(epsilon)

                # Update the progress bar
                if len(self.episode_rewards) >= 10:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    pbar.set_postfix({
                        'reward': f'{episode_reward:.1f}',
                        'avg_10': f'{avg_reward:.1f}',
                        'epsilon': f'{epsilon:.3f}'
                    })

                # Reset the episode
                state, _ = self.env.reset()
                state = np.asarray(state, dtype=np.float32)
                episode_reward = 0.0
                episode_length = 0

        pbar.close()

        return self.episode_rewards, self.episode_lengths, self.losses, self.epsilons

    @torch.no_grad()
    def evaluate(self, num_episodes: int = 100) -> Tuple[float, float]:
        """Evaluates the trained policy (greedy, no epsilon)."""
        self.policy_net.eval()

        rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset(seed=self.config.seed + 10000 + episode)
            state = np.asarray(state, dtype=np.float32)
            episode_reward = 0.0
            done = False

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
                q_values = self.policy_net(state_tensor)
                action = int(q_values.argmax(dim=1).item())

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += float(reward)
                state = np.asarray(next_state, dtype=np.float32)
                done = terminated or truncated

            rewards.append(episode_reward)

        self.policy_net.train()

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))

        return mean_reward, std_reward

    def save(self, path: str):
        """Saves the trained model."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'config': self.config,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Loads a model."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        print(f"Model loaded from {path}")


def plot_training_curves(
    episode_rewards: List[float],
    episode_lengths: List[float],
    losses: List[float],
    epsilons: List[float],
    save_path: str = "dqn_training.png"
):
    """Visualizes the training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(episode_rewards) >= 20:
        window = min(20, len(episode_rewards))
        rolling_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), rolling_avg,
                       color='red', linewidth=2, label=f'Rolling Avg ({window})')
    axes[0, 0].axhline(y=200, color='green', linestyle='--', label='Solved (200)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.3, color='purple')
    if len(episode_lengths) >= 20:
        window = min(20, len(episode_lengths))
        rolling_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(episode_lengths)), rolling_avg,
                       color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].grid(True, alpha=0.3)

    # Losses
    if losses:
        axes[1, 0].plot(losses, alpha=0.5, color='red')
        if len(losses) >= 100:
            window = min(100, len(losses))
            rolling_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(losses)), rolling_avg,
                           color='darkred', linewidth=2)
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

    # Epsilon decay
    if epsilons:
        axes[1, 1].plot(epsilons, color='green', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].set_title('Epsilon Decay')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="DQN on LunarLander-v2")
    parser.add_argument("--total-steps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.997, help="Epsilon decay rate")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--target-update", type=int, default=1000, help="Target network update frequency")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Warmup steps before training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="dqn_lunarlander.pt", help="Output model path")

    args = parser.parse_args()

    # Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create the environment
    env = gym.make("LunarLander-v2")

    # Configuration
    config = DQNConfig(
        total_steps=args.total_steps,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )

    print("=" * 60)
    print("DQN Training on LunarLander-v2")
    print("=" * 60)
    print(f"Total steps: {config.total_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Buffer size: {config.buffer_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Target update: {config.target_update}")
    print("Environment: LunarLander-v2")
    print("=" * 60)

    # Create the agent
    agent = DQNAgent(env, config)

    # Train
    episode_rewards, episode_lengths, losses, epsilons = agent.train()

    # Evaluate
    print("\nEvaluating trained policy...")
    mean_reward, std_reward = agent.evaluate(num_episodes=100)
    print(f"Evaluation over 100 episodes: {mean_reward:.2f} ± {std_reward:.2f}")

    if mean_reward >= 200:
        print("✓ Environment SOLVED! (Average reward >= 200)")
    else:
        print("✗ Environment not solved yet. Try training longer or tuning hyperparameters.")

    # Save the model
    agent.save(args.output)

    # Visualize
    plot_training_curves(episode_rewards, episode_lengths, losses, epsilons)

    env.close()


if __name__ == "__main__":
    main()
