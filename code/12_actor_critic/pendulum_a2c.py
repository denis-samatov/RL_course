"""Pendulum-v1 A2C (Advantage Actor-Critic) implementation.

This script implements the A2C algorithm for the Pendulum-v1 environment with
continuous action space. It demonstrates the power of Actor-Critic methods for
continuous control tasks.

Key features:
- Actor-Critic architecture with shared backbone
- Continuous action space with Gaussian policy
- TD-error (advantage) for variance reduction
- Entropy regularization for exploration
- Gradient clipping for stability
- Comprehensive metrics and visualization
"""

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordVideo
from torch.distributions import Normal
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")


@dataclass
class A2CConfig:
    """Configuration for training A2C on Pendulum-v1.

    Notes:
        - Pendulum-v1 has a 3-dimensional continuous state and a
          1-dimensional continuous action [-2, 2].
        - Goal: swing up and balance the inverted pendulum.
        - Reward range: roughly [-1600, 0] per episode.
        - Episode length: fixed, 200 steps.
    """
    num_episodes: int = 1000
    max_steps_per_episode: int = 200
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 1e-3
    gamma: float = 0.99
    entropy_coef: float = 0.001  # Small entropy for continuous control
    value_loss_coef: float = 0.5
    gradient_clip: float = 0.5
    seed: int = 42
    hidden_sizes: Tuple[int, int] = (256, 256)
    log_std_init: float = 0.0  # Initial log std for action distribution
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    render_mode: str = "rgb_array"
    model_save_path: str = "pendulum_a2c.pt"
    video_folder: str = "videos/pendulum"
    video_episodes: int = 5


class ActorCriticNetwork(nn.Module):
    """The Actor-Critic network with a shared backbone, for continuous
    control.

    The actor outputs the mean and log_std of a Gaussian
    distribution. The critic outputs a single value estimate.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        hidden_sizes: Tuple[int, int] = (256, 256),
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        
        # Actor head (continuous actions)
        self.actor_mean = nn.Linear(hidden_sizes[1], action_dim)
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        # Critic head
        self.critic = nn.Linear(hidden_sizes[1], 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """Forward pass, returning the action distribution and value
        estimate.

        Args:
            state: A state tensor of shape (batch_size, state_dim) or
                (state_dim,).

        Returns:
            A tuple (action_distribution, value_estimate).
        """
        features = self.shared(state)
        
        # Actor: Gaussian distribution
        mean = self.actor_mean(features)
        log_std = torch.clamp(self.actor_log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        # Critic: state value
        value = self.critic(features)
        
        return dist, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chooses an action from the policy.

        Args:
            state: The state tensor.
            deterministic: If True, returns the mean action (no
                sampling).

        Returns:
            A tuple (action, log_prob, entropy).
        """
        dist, _ = self.forward(state)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy


class A2CAgent:
    """An A2C agent for continuous control."""
    
    def __init__(self, config: A2CConfig):
        self.config = config
        
        # Environment setup
        self.env = gym.make("Pendulum-v1", render_mode=config.render_mode)
        self.env.action_space.seed(config.seed)
        self.env.observation_space.seed(config.seed)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_high = float(self.env.action_space.high[0])
        
        # Network
        self.model = ActorCriticNetwork(
            self.state_dim,
            self.action_dim,
            config.hidden_sizes,
            config.log_std_init,
            config.log_std_min,
            config.log_std_max
        )
        
        # Separate optimizers for actor and critic (best practice)
        # IMPORTANT: Shared parameters must belong to ONLY ONE optimizer
        # to avoid non-deterministic updates and optimizer state conflicts.
        # We assign shared params to critic_optimizer and let actor loss
        # backpropagate through them before critic's optimizer.step().
        actor_params = list(self.model.actor_mean.parameters()) + [self.model.actor_log_std]
        critic_params = list(self.model.critic.parameters()) + list(self.model.shared.parameters())
        
        self.actor_optimizer = optim.Adam(actor_params, lr=config.learning_rate_actor)
        self.critic_optimizer = optim.Adam(critic_params, lr=config.learning_rate_critic)
        
        # Random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Chooses an action using the current policy.

        Args:
            state: The current state observation.
            deterministic: If True, use the mean action.

        Returns:
            A tuple (clipped_action, log_prob, entropy).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, entropy = self.model.get_action(state_tensor, deterministic)
        
        # Clip action to valid range and convert to numpy
        action_clipped = torch.clamp(action, -self.action_high, self.action_high)
        action_np = action_clipped.detach().numpy().flatten()
        
        return action_np, log_prob, entropy
    
    def train_step(
        self, 
        state: np.ndarray,
        action_log_prob: torch.Tensor,
        action_entropy: torch.Tensor,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Tuple[float, float]:
        """Performs a single A2C update step.

        Args:
            state: The current state.
            action_log_prob: The log-probability of the action taken.
            action_entropy: The entropy of the action distribution.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode ended.

        Returns:
            A tuple (actor_loss, critic_loss).
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])
        
        # Get current state value
        _, value = self.model(state_tensor)
        value = value.squeeze()
        
        # Get next state value (with no_grad for TD target)
        with torch.no_grad():
            _, next_value = self.model(next_state_tensor)
            next_value = next_value.squeeze()
            if done:
                next_value = torch.zeros_like(next_value)
        
        # Compute TD error (advantage)
        td_target = reward_tensor + self.config.gamma * next_value
        td_error = td_target - value
        advantage = td_error.detach()  # Don't backprop through advantage
        
        # Actor loss (policy gradient with entropy bonus)
        actor_loss = -(action_log_prob * advantage + self.config.entropy_coef * action_entropy)
        
        # Critic loss (MSE)
        critic_loss = self.config.value_loss_coef * td_error.pow(2)
        
        # IMPORTANT: Since shared params are in critic_optimizer only,
        # we do both backward() calls first (to accumulate gradients),
        # then both step() calls. This ensures shared params get gradients
        # from both actor and critic losses.
        
        # Zero gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        # Backward passes (accumulate gradients)
        actor_loss.backward(retain_graph=True)  # Retain for critic backward
        critic_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.model.actor_mean.parameters()) + [self.model.actor_log_std],
            self.config.gradient_clip
        )
        torch.nn.utils.clip_grad_norm_(
            list(self.model.critic.parameters()) + list(self.model.shared.parameters()),
            self.config.gradient_clip
        )
        
        # Update parameters
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def train_episode(self) -> Tuple[float, int, float, float]:
        """Runs a single episode with online updates.

        Returns:
            A tuple (total_reward, episode_length,
            mean_actor_loss, mean_critic_loss).
        """
        state, _ = self.env.reset(seed=self.config.seed + np.random.randint(0, 10000))
        episode_reward = 0.0
        actor_losses = []
        critic_losses = []
        
        for t in range(self.config.max_steps_per_episode):
            # Select action
            action, log_prob, entropy = self.select_action(state, deterministic=False)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # A2C update
            actor_loss, critic_loss = self.train_step(
                state, log_prob, entropy, reward, next_state, done
            )
            
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        return episode_reward, t + 1, np.mean(actor_losses), np.mean(critic_losses)
    
    def train(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Trains the agent for the configured number of episodes.

        Returns:
            A tuple (episode_rewards, episode_lengths,
            actor_losses, critic_losses).
        """
        episode_rewards = []
        episode_lengths = []
        actor_losses = []
        critic_losses = []
        
        with tqdm(range(self.config.num_episodes), desc="Training A2C", unit="ep") as pbar:
            for episode in pbar:
                reward, length, actor_loss, critic_loss = self.train_episode()
                episode_rewards.append(reward)
                episode_lengths.append(length)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                
                # Update progress bar
                if episode >= 99:
                    avg_reward = np.mean(episode_rewards[-100:])
                    pbar.set_postfix({
                        'reward': f'{reward:.1f}',
                        'avg_100': f'{avg_reward:.1f}',
                        'actor_loss': f'{actor_loss:.3f}',
                        'critic_loss': f'{critic_loss:.3f}'
                    })
        
        return (
            np.array(episode_rewards),
            np.array(episode_lengths),
            np.array(actor_losses),
            np.array(critic_losses)
        )
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Tuple[float, float]:
        """Evaluates the trained policy.

        Args:
            num_episodes: The number of episodes to evaluate over.
            render: Whether to render the episodes.

        Returns:
            A tuple (mean_reward, reward_std).
        """
        eval_env = gym.make("Pendulum-v1", render_mode="human" if render else None)
        rewards = []
        
        for episode in range(num_episodes):
            state, _ = eval_env.reset(seed=self.config.seed + 10000 + episode)
            episode_reward = 0.0
            done = False
            
            while not done:
                action, _, _ = self.select_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
        
        eval_env.close()
        return np.mean(rewards), np.std(rewards)
    
    def save(self, path: str):
        """Saves the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Loads the model."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {path}")


def plot_training_curves(
    rewards: np.ndarray,
    lengths: np.ndarray,
    actor_losses: np.ndarray,
    critic_losses: np.ndarray,
    window: int = 50
):
    """Plots comprehensive training curves."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards
    ax1.plot(rewards, alpha=0.3, label='Episode reward')
    if len(rewards) >= window:
        rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), rolling_mean, label=f'{window}-episode average', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('A2C on Pendulum-v1: Training Rewards')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Lengths
    ax2.plot(lengths, alpha=0.3, label='Episode length')
    if len(lengths) >= window:
        rolling_mean = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(lengths)), rolling_mean, label=f'{window}-episode average', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('A2C on Pendulum-v1: Episode Lengths')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Actor loss
    ax3.plot(actor_losses, alpha=0.3, label='Actor loss')
    if len(actor_losses) >= window:
        rolling_mean = np.convolve(actor_losses, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(actor_losses)), rolling_mean, label=f'{window}-episode average', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Actor Loss')
    ax3.set_title('A2C on Pendulum-v1: Actor Loss')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Critic loss
    ax4.plot(critic_losses, alpha=0.3, label='Critic loss')
    if len(critic_losses) >= window:
        rolling_mean = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
        ax4.plot(range(window-1, len(critic_losses)), rolling_mean, label=f'{window}-episode average', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Critic Loss')
    ax4.set_title('A2C on Pendulum-v1: Critic Loss')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pendulum_a2c_training.png', dpi=150)
    plt.show()


def record_video(agent: A2CAgent, num_episodes: int = 5):
    """Records videos of evaluation episodes."""
    video_path = Path(agent.config.video_folder)
    video_path.mkdir(parents=True, exist_ok=True)
    
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_path,
        episode_trigger=lambda x: True,
        name_prefix="a2c"
    )
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=agent.config.seed + 20000 + episode)
        episode_reward = 0.0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        print(f"Video episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()


def main(args):
    """The main training pipeline."""
    config = A2CConfig(
        num_episodes=args.episodes,
        learning_rate_actor=args.lr_actor,
        learning_rate_critic=args.lr_critic,
        entropy_coef=args.entropy,
        seed=args.seed
    )
    
    print("=" * 60)
    print("A2C on Pendulum-v1 (Continuous Control)")
    print("=" * 60)
    print(f"Episodes: {config.num_episodes}")
    print(f"Actor LR: {config.learning_rate_actor}")
    print(f"Critic LR: {config.learning_rate_critic}")
    print(f"Entropy coefficient: {config.entropy_coef}")
    print(f"Seed: {config.seed}")
    print("=" * 60)
    
    # Train
    agent = A2CAgent(config)
    rewards, lengths, actor_losses, critic_losses = agent.train()
    
    # Plot results
    plot_training_curves(rewards, lengths, actor_losses, critic_losses)
    
    # Evaluate
    print("\nEvaluating trained policy...")
    mean_reward, std_reward = agent.evaluate(num_episodes=100)
    print(f"Evaluation over 100 episodes: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Pendulum is "solved" when average reward > -200
    if mean_reward >= -200:
        print("✓ Environment SOLVED! (Average reward >= -200)")
    else:
        print(f"✗ Not solved yet. Current average: {mean_reward:.2f}")
    
    # Save model
    agent.save(config.model_save_path)
    
    # Record videos
    if args.record_video:
        print("\nRecording videos...")
        record_video(agent, num_episodes=5)
        print(f"Videos saved to {config.video_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2C on Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--lr-actor", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument("--entropy", type=float, default=0.001, help="Entropy coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--record-video", action="store_true", help="Record evaluation videos")
    
    args = parser.parse_args()
    main(args)

