"""Записывает видео нескольких эпизодов со случайной политикой в среде CartPole-v1.

Этот скрипт использует функцию `record_episodes` из модуля `rlcourse.sem01.video`
для создания и сохранения видео.

Основные шаги:
1. Устанавливает директорию для сохранения видео.
2. Вызывает `record_episodes` для запуска симуляции и записи.
3. Выводит пути к сохраненным файлам.

Для запуска из командной строки:
    python sem01_mdp_bellman/scripts/record_video.py
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
