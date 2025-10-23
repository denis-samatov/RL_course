"""
Скрипт для записи видео выполнения агента в CartPole-v1 (использует модуль rlcourse.sem01.video).
"""
import os, sys

# Add ../../src to sys.path for local package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from rlcourse.sem01.video import record_episodes, SEED, default_video_dir


if __name__ == "__main__":
    print("=== Запись случайной политики ===\n")
    target_dir = default_video_dir()
    files = record_episodes(
        env_name="CartPole-v1",
        num_episodes=3,
        video_folder=str(target_dir),
        name_prefix="cartpole-random",
        seed=SEED,
    )
    print("\n✓ Видео успешно сохранены в:", target_dir)
    for vf in sorted(files):
        print(" -", vf)
