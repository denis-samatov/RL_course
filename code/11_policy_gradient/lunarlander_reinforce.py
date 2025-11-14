"""LunarLander-v2 REINFORCE (Policy Gradient) implementation.

This script implements the REINFORCE algorithm (Monte Carlo Policy Gradient) for
the LunarLander-v2 environment. It demonstrates policy-based methods that directly
learn a stochastic policy through gradient ascent on expected returns.

Key features:
- Pure policy gradient with baseline (value function)
- Entropy regularization for exploration
- Gradient normalization for stability
- Video recording of best episodes
- Comprehensive metrics tracking
"""

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordVideo
from torch.distributions import Categorical
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")


@dataclass
class REINFORCEConfig:
    """Configuration for REINFORCE training on LunarLander-v2.
    
    Notes:
        - LunarLander-v2 has 8-dim continuous state space and 4 discrete actions
        - Typical episode length: 200-400 steps
        - Solved when average reward > 200 over 100 consecutive episodes
    """
    num_episodes: int = 2000
    max_steps_per_episode: int = 1000
    learning_rate_policy: float = 3e-4
    learning_rate_value: float = 1e-3
    gamma: float = 0.99
    entropy_coef: float = 0.01  # Entropy regularization coefficient
    baseline: bool = True  # Use value function baseline to reduce variance
    normalize_advantages: bool = True
    gradient_clip: float = 0.5
    seed: int = 42
    hidden_sizes: Tuple[int, int] = (128, 128)
    render_mode: str = "rgb_array"
    model_save_path: str = "lunarlander_reinforce.pt"
    video_folder: str = "videos/lunarlander"
    video_episodes: int = 5  # Record top-5 episodes


class PolicyNetwork(nn.Module):
    """Policy network for discrete action spaces.
    
    Outputs a probability distribution over actions via softmax.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_dim)
        
    def forward(self, state: torch.Tensor) -> Categorical:
        """Forward pass returning action distribution.
        
        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)
            
        Returns:
            Categorical distribution over actions
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """Value function network (baseline) for variance reduction."""
    
    def __init__(self, state_dim: int, hidden_sizes: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state value.
        
        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)
            
        Returns:
            Value estimate of shape (batch_size, 1) or (1,)
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class REINFORCEAgent:
    """REINFORCE agent with optional baseline and entropy regularization."""
    
    def __init__(self, config: REINFORCEConfig):
        self.config = config
        
        # Environment setup
        self.env = gym.make("LunarLander-v2", render_mode=config.render_mode)
        self.env.action_space.seed(config.seed)
        self.env.observation_space.seed(config.seed)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Networks
        self.policy = PolicyNetwork(self.state_dim, self.action_dim, config.hidden_sizes)
        self.value = ValueNetwork(self.state_dim, config.hidden_sizes) if config.baseline else None
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate_policy)
        if self.value:
            self.value_optimizer = optim.Adam(self.value.parameters(), lr=config.learning_rate_value)
        
        # Random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action using current policy.
        
        Args:
            state: Current state observation
            
        Returns:
            Tuple of (action, log_prob, entropy)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        dist = self.policy(state_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy
    
    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """Compute discounted returns (Monte Carlo).
        
        Args:
            rewards: List of rewards from episode
            
        Returns:
            Tensor of discounted returns for each timestep
        """
        returns = []
        G = 0.0
        for reward in reversed(rewards):
            G = reward + self.config.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        return returns
    
    def train_episode(self, episode_idx: int = 0) -> Tuple[float, int]:
        """Run one episode and update policy.
        
        Args:
            episode_idx: Episode index for seeding (enables exploration across episodes)
            
        Returns:
            Tuple of (total_reward, episode_length)
        """
        states, log_probs, rewards, entropies = [], [], [], []
        
        # Use episode_idx for seed to ensure different trajectories per episode
        state, _ = self.env.reset(seed=self.config.seed + episode_idx)
        done = False
        episode_reward = 0.0
        
        # Collect trajectory
        for t in range(self.config.max_steps_per_episode):
            action, log_prob, entropy = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            states.append(state)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            
            state = next_state
            episode_reward += reward
            done = terminated or truncated
            
            if done:
                break
        
        # Compute returns
        returns = self.compute_returns(rewards)
        
        # Compute advantages (with baseline if enabled)
        if self.config.baseline:
            # Properly move tensors to device for GPU compatibility
            states_tensor = torch.as_tensor(
                np.array(states), dtype=torch.float32, device=self.device
            )
            values = self.value(states_tensor).squeeze()
            
            # Normalize returns for numerical stability (improves value learning)
            returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)
            advantages = returns_normalized - values.detach()
            
            # Update value function
            value_loss = nn.MSELoss()(values, returns_normalized)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.config.gradient_clip)
            self.value_optimizer.step()
        else:
            # Without baseline, use returns as advantages
            returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)
            advantages = returns_normalized
        
        # Normalize advantages (per-episode normalization for stability)
        if self.config.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy gradient update
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        
        policy_loss = -(log_probs * advantages).mean()
        entropy_loss = -entropies.mean()
        total_loss = policy_loss + self.config.entropy_coef * entropy_loss
        
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.gradient_clip)
        self.policy_optimizer.step()
        
        return episode_reward, t + 1
    
    def train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Train the agent for configured number of episodes.
        
        Returns:
            Tuple of (episode_rewards, episode_lengths)
        """
        episode_rewards = []
        episode_lengths = []
        
        with tqdm(range(self.config.num_episodes), desc="Training REINFORCE", unit="ep") as pbar:
            for episode in pbar:
                reward, length = self.train_episode(episode_idx=episode)
                episode_rewards.append(reward)
                episode_lengths.append(length)
                
                # Update progress bar with rolling statistics
                if episode >= 99:
                    avg_reward = np.mean(episode_rewards[-100:])
                    avg_length = np.mean(episode_lengths[-100:])
                    pbar.set_postfix({
                        'reward': f'{reward:.1f}',
                        'avg_100': f'{avg_reward:.1f}',
                        'length': f'{length}'
                    })
        
        return np.array(episode_rewards), np.array(episode_lengths)
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Tuple[float, float]:
        """Evaluate trained policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes
            
        Returns:
            Tuple of (mean_reward, std_reward)
        """
        eval_env = gym.make("LunarLander-v2", render_mode="human" if render else None)
        rewards = []
        
        for episode in range(num_episodes):
            state, _ = eval_env.reset(seed=self.config.seed + 10000 + episode)
            episode_reward = 0.0
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    dist = self.policy(state_tensor)
                    action = dist.probs.argmax().item()  # Greedy action
                
                state, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
        
        eval_env.close()
        return np.mean(rewards), np.std(rewards)
    
    def save(self, path: str):
        """Save policy and value networks."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict() if self.value else None,
            'config': self.config
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load policy and value networks."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if self.value and checkpoint['value_state_dict']:
            self.value.load_state_dict(checkpoint['value_state_dict'])
        print(f"Model loaded from {path}")


def plot_training_curves(rewards: np.ndarray, lengths: np.ndarray, window: int = 100):
    """Plot training curves with rolling average."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Rewards
    ax1.plot(rewards, alpha=0.3, label='Episode reward')
    if len(rewards) >= window:
        rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), rolling_mean, label=f'{window}-episode average', linewidth=2)
    ax1.axhline(y=200, color='r', linestyle='--', label='Solved threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('REINFORCE on LunarLander-v2: Training Rewards')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Lengths
    ax2.plot(lengths, alpha=0.3, label='Episode length')
    if len(lengths) >= window:
        rolling_mean = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(lengths)), rolling_mean, label=f'{window}-episode average', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('REINFORCE on LunarLander-v2: Episode Lengths')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lunarlander_reinforce_training.png', dpi=150)
    plt.show()


def record_video(agent: REINFORCEAgent, num_episodes: int = 5):
    """Record video of best episodes."""
    video_path = Path(agent.config.video_folder)
    video_path.mkdir(parents=True, exist_ok=True)
    
    # Wrapper to record video
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = RecordVideo(
        env, 
        video_path,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix="reinforce"
    )
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=agent.config.seed + 20000 + episode)
        episode_reward = 0.0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                dist = agent.policy(state_tensor)
                action = dist.probs.argmax().item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        print(f"Video episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()


def main(args):
    """Main training pipeline."""
    config = REINFORCEConfig(
        num_episodes=args.episodes,
        learning_rate_policy=args.lr,
        baseline=args.baseline,
        entropy_coef=args.entropy,
        seed=args.seed
    )
    
    print("=" * 60)
    print("REINFORCE on LunarLander-v2")
    print("=" * 60)
    print(f"Episodes: {config.num_episodes}")
    print(f"Learning rate: {config.learning_rate_policy}")
    print(f"Baseline: {config.baseline}")
    print(f"Entropy coefficient: {config.entropy_coef}")
    print(f"Seed: {config.seed}")
    print("=" * 60)
    
    # Train
    agent = REINFORCEAgent(config)
    rewards, lengths = agent.train()
    
    # Plot results
    plot_training_curves(rewards, lengths)
    
    # Evaluate
    print("\nEvaluating trained policy...")
    mean_reward, std_reward = agent.evaluate(num_episodes=100)
    print(f"Evaluation over 100 episodes: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Check if solved
    if mean_reward >= 200:
        print("✓ Environment SOLVED! (Average reward >= 200)")
    else:
        print(f"✗ Not solved yet. Need {200 - mean_reward:.2f} more reward.")
    
    # Save model
    agent.save(config.model_save_path)
    
    # Record videos
    if args.record_video:
        print("\nRecording videos...")
        record_video(agent, num_episodes=5)
        print(f"Videos saved to {config.video_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REINFORCE on LunarLander-v2")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--no-baseline", action="store_false", dest="baseline", 
                        help="Disable value baseline (baseline enabled by default)")
    parser.set_defaults(baseline=True)
    parser.add_argument("--entropy", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--record-video", action="store_true", help="Record evaluation videos")
    
    args = parser.parse_args()
    main(args)

