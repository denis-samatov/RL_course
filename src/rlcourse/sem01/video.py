import os
from pathlib import Path
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

SEED = 42


def default_video_dir() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / 'sem01_mdp_bellman' / 'assets' / 'videos'


def record_episodes(
    env_name: str = "CartPole-v1",
    num_episodes: int = 3,
    video_folder: str | None = None,
    name_prefix: str = "cartpole-random",
    seed: int = SEED,
):
    if video_folder is None:
        video_folder = str(default_video_dir())
    os.makedirs(video_folder, exist_ok=True)
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda episode_id: episode_id < num_episodes,
        name_prefix=name_prefix,
    )
    env.action_space.seed(seed)
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = trunc = False
        while not (done or trunc):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
    env.close()
    return [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
