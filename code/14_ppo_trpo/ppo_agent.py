"""
Полная реализация Proximal Policy Optimization (PPO) агента.

Особенности:
- PPO-Clip с GAE (Generalized Advantage Estimation)
- Поддержка непрерывных действий (Gaussian policy)
- Multiple epochs на собранных траекториях
- Gradient clipping, value clipping, LR annealing
- Vectorized environments для параллельного сбора данных
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from tqdm import tqdm


@dataclass
class PPOConfig:
    """Конфигурация PPO гиперпараметров."""
    
    # Environment
    env_id: str = "BipedalWalker-v3"
    n_envs: int = 8  # Количество параллельных сред
    
    # Training
    n_steps: int = 2048  # Шагов для сбора траекторий
    total_timesteps: int = 2_000_000  # Всего шагов обучения
    batch_size: int = 64  # Mini-batch size
    n_epochs: int = 10  # Эпох оптимизации на одном батче
    
    # PPO hyperparameters
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_range: float = 0.2  # Epsilon для clipping
    clip_range_vf: Optional[float] = None  # Value function clipping (если None, не используется)
    
    # Loss coefficients
    value_coef: float = 0.5  # Вес value loss
    entropy_coef: float = 0.01  # Вес entropy bonus
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Learning rate
    learning_rate: float = 3e-4
    lr_annealing: bool = True  # Линейное уменьшение LR
    
    # Network architecture
    hidden_dim: int = 256
    shared_backbone: bool = True  # Shared или separate networks
    
    # Normalization
    normalize_advantages: bool = True
    normalize_observations: bool = False  # Running mean/std для obs
    
    # Logging
    log_interval: int = 10  # Логировать каждые N обновлений
    save_interval: int = 100  # Сохранять модель каждые N обновлений
    eval_episodes: int = 10  # Эпизодов для evaluation
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic сеть для PPO.
    
    Поддерживает как shared, так и separate архитектуры.
    Для непрерывных действий: Gaussian policy с learnable std.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        shared: bool = True,
    ):
        super().__init__()
        
        self.shared = shared
        
        if shared:
            # Shared backbone
            self.backbone = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            # Actor head
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            # Critic head
            self.critic = nn.Linear(hidden_dim, 1)
        else:
            # Separate Actor
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            
            # Separate Critic
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        
        # Learnable log std для Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            action_mean: Mean для Gaussian policy [batch, action_dim]
            action_std: Std для Gaussian policy [batch, action_dim]
            value: Value estimation [batch, 1]
        """
        if self.shared:
            features = self.backbone(obs)
            action_mean = self.actor_mean(features)
            value = self.critic(features)
        else:
            actor_features = self.actor(obs)
            action_mean = self.actor_mean(actor_features)
            value = self.critic(obs)
        
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        
        return action_mean, action_std, value
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Вычисляет действие, log_prob, entropy, value.
        
        Если action=None, сэмплирует новое действие.
        Иначе вычисляет log_prob для заданного действия.
        
        Returns:
            action: [batch, action_dim]
            log_prob: [batch]
            entropy: [batch]
            value: [batch]
        """
        action_mean, action_std, value = self.forward(obs)
        dist = Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)


class RolloutBuffer:
    """
    Буфер для хранения траекторий и вычисления GAE.
    """
    
    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        
        # Буферы
        self.observations = torch.zeros((n_steps, n_envs, obs_dim), device=device)
        self.actions = torch.zeros((n_steps, n_envs, action_dim), device=device)
        self.log_probs = torch.zeros((n_steps, n_envs), device=device)
        self.rewards = torch.zeros((n_steps, n_envs), device=device)
        self.dones = torch.zeros((n_steps, n_envs), device=device)
        self.values = torch.zeros((n_steps, n_envs), device=device)
        
        # Вычисленные GAE
        self.advantages = torch.zeros((n_steps, n_envs), device=device)
        self.returns = torch.zeros((n_steps, n_envs), device=device)
        
        self.ptr = 0
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
    ):
        """Добавляет один шаг траектории."""
        self.observations[self.ptr] = torch.as_tensor(obs, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, device=self.device)
        self.log_probs[self.ptr] = torch.as_tensor(log_prob, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(done, device=self.device)
        self.values[self.ptr] = torch.as_tensor(value, device=self.device)
        
        self.ptr += 1
    
    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ):
        """
        Вычисляет GAE advantages и returns.
        
        Args:
            last_value: V(s_T+1) для последнего состояния [n_envs]
            gamma: Discount factor
            gae_lambda: GAE lambda
        """
        last_gae = 0.0
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            # TD-error: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
            delta = (
                self.rewards[t]
                + gamma * next_value * (1.0 - self.dones[t])
                - self.values[t]
            )
            
            # GAE: A_t = δ_t + (γλ) δ_{t+1} + (γλ)^2 δ_{t+2} + ...
            last_gae = delta + gamma * gae_lambda * (1.0 - self.dones[t]) * last_gae
            self.advantages[t] = last_gae
        
        # Returns: G_t = A_t + V(s_t)
        self.returns = self.advantages + self.values
    
    def get_batches(self, batch_size: int):
        """
        Генерирует mini-batches для обучения.
        
        Yields:
            Dict с ключами: obs, actions, log_probs, advantages, returns, values
        """
        # Flatten [n_steps, n_envs] → [n_steps * n_envs]
        obs = self.observations.reshape(-1, self.observations.shape[-1])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        log_probs = self.log_probs.reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)
        values = self.values.reshape(-1)
        
        total_size = obs.shape[0]
        indices = torch.randperm(total_size, device=self.device)
        
        for start in range(0, total_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield {
                "obs": obs[batch_indices],
                "actions": actions[batch_indices],
                "log_probs": log_probs[batch_indices],
                "advantages": advantages[batch_indices],
                "returns": returns[batch_indices],
                "values": values[batch_indices],
            }
    
    def reset(self):
        """Сбрасывает указатель буфера."""
        self.ptr = 0


class PPOAgent:
    """
    PPO агент для непрерывных действий.
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        
        # Создаём vectorized environment
        self.envs = gym.vector.SyncVectorEnv(
            [self._make_env(i) for i in range(config.n_envs)]
        )
        
        obs_dim = self.envs.single_observation_space.shape[0]
        action_dim = self.envs.single_action_space.shape[0]
        
        # Сеть
        self.network = ActorCriticNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            shared=config.shared_backbone,
        ).to(config.device)
        
        # Оптимизатор
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            n_steps=config.n_steps,
            n_envs=config.n_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=config.device,
        )
        
        # Метрики
        self.episode_rewards = []
        self.episode_lengths = []
        self.global_step = 0
    
    def _make_env(self, rank: int):
        """Фабрика для создания среды."""
        def _init():
            env = gym.make(self.config.env_id)
            env.reset(seed=self.config.seed + rank)
            return env
        return _init
    
    def collect_rollouts(self) -> Dict[str, float]:
        """
        Собирает n_steps траекторий со всех сред.
        
        Returns:
            Dict с метриками (mean_reward, mean_length)
        """
        obs, _ = self.envs.reset()
        self.buffer.reset()
        
        episode_rewards_temp = []
        episode_lengths_temp = []
        episode_reward_acc = np.zeros(self.config.n_envs)
        episode_length_acc = np.zeros(self.config.n_envs)
        
        for step in range(self.config.n_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.config.device)
            
            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(obs_tensor)
            
            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy()
            
            next_obs, reward, terminated, truncated, info = self.envs.step(action_np)
            done = np.logical_or(terminated, truncated)
            
            # Сохраняем в буфер
            self.buffer.add(obs, action_np, log_prob_np, reward, done, value_np)
            
            # Обновляем метрики
            episode_reward_acc += reward
            episode_length_acc += 1
            
            for i in range(self.config.n_envs):
                if done[i]:
                    episode_rewards_temp.append(episode_reward_acc[i])
                    episode_lengths_temp.append(episode_length_acc[i])
                    episode_reward_acc[i] = 0
                    episode_length_acc[i] = 0
            
            obs = next_obs
            self.global_step += self.config.n_envs
        
        # Вычисляем last_value для GAE
        with torch.no_grad():
            obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.config.device)
            _, _, _, last_value = self.network.get_action_and_value(obs_tensor)
        
        # Вычисляем GAE
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        
        # Метрики
        metrics = {}
        if len(episode_rewards_temp) > 0:
            metrics["mean_reward"] = np.mean(episode_rewards_temp)
            metrics["mean_length"] = np.mean(episode_lengths_temp)
            self.episode_rewards.extend(episode_rewards_temp)
            self.episode_lengths.extend(episode_lengths_temp)
        
        return metrics
    
    def update_policy(self) -> Dict[str, float]:
        """
        Обновляет политику на собранных траекториях.
        
        Returns:
            Dict с метриками (policy_loss, value_loss, entropy, approx_kl)
        """
        metrics_accum = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clipfrac": [],
        }
        
        for epoch in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                # Вычисляем текущие log_probs и values
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    batch["obs"], batch["actions"]
                )
                
                # Normalize advantages
                advantages = batch["advantages"]
                if self.config.normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Probability ratio: r_t = π_new / π_old
                log_ratio = new_log_probs - batch["log_probs"]
                ratio = torch.exp(log_ratio)
                
                # Approx KL divergence (для мониторинга)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.config.clip_range).float().mean()
                
                # PPO-Clip loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.config.clip_range_vf is not None:
                    # Value clipping
                    v_clipped = batch["values"] + torch.clamp(
                        new_values - batch["values"],
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf,
                    )
                    v_loss1 = (new_values - batch["returns"]) ** 2
                    v_loss2 = (v_clipped - batch["returns"]) ** 2
                    value_loss = torch.max(v_loss1, v_loss2).mean()
                else:
                    value_loss = ((new_values - batch["returns"]) ** 2).mean()
                
                # Entropy bonus (для exploration)
                entropy_loss = entropy.mean()
                
                # Полная loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                )
                
                # Обновление
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Собираем метрики
                metrics_accum["policy_loss"].append(policy_loss.item())
                metrics_accum["value_loss"].append(value_loss.item())
                metrics_accum["entropy"].append(entropy_loss.item())
                metrics_accum["approx_kl"].append(approx_kl.item())
                metrics_accum["clipfrac"].append(clipfrac.item())
        
        # Усредняем метрики
        metrics = {k: np.mean(v) for k, v in metrics_accum.items()}
        return metrics
    
    def train(self):
        """Основной цикл обучения."""
        n_updates = self.config.total_timesteps // (self.config.n_steps * self.config.n_envs)
        
        pbar = tqdm(range(n_updates), desc="PPO Training")
        
        for update in pbar:
            # Learning rate annealing
            if self.config.lr_annealing:
                frac = 1.0 - update / n_updates
                lr_now = frac * self.config.learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr_now
            
            # Сбор траекторий
            rollout_metrics = self.collect_rollouts()
            
            # Обновление политики
            update_metrics = self.update_policy()
            
            # Логирование
            if update % self.config.log_interval == 0:
                pbar.set_postfix({
                    "reward": rollout_metrics.get("mean_reward", 0),
                    "policy_loss": update_metrics["policy_loss"],
                    "kl": update_metrics["approx_kl"],
                })
        
        pbar.close()
    
    def save(self, path: str):
        """Сохраняет модель."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Загружает модель."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Демонстрация обучения
    config = PPOConfig(
        env_id="BipedalWalker-v3",
        total_timesteps=1_000_000,
        n_envs=4,
    )
    
    agent = PPOAgent(config)
    agent.train()
    agent.save("ppo_bipedalwalker.pt")

