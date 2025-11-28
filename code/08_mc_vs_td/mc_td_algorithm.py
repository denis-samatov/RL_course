"""Monte Carlo vs. Temporal Difference prediction on FrozenLake.

This module mirrors material from the theoretical notes on how Monte Carlo (MC)
and Temporal Difference (TD) methods approximate value functions. It provides a
minimal FrozenLake experiment that compares first-visit MC prediction with TD(0)
prediction under the same policy.
"""


import numpy as np
import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


Policy = Callable[[int], np.ndarray]
"""Политика — функция, отображающая состояние в распределение вероятностей по действиям."""


# ------------------------- utilities -------------------------

def set_seeds(env: gym.Env, seed: int) -> None:
    """Инициализирует генераторы случайных чисел (ГСЧ) среды Gym.

    Args:
        env: Среда, внутренние ГСЧ которой должны быть сброшены.
        seed: Детерминированный seed, передаваемый в среду.

    Notes:
        Эта функция НЕ инициализирует глобальное состояние np.random, чтобы
        избежать вмешательства в ГСЧ вызывающего кода. Все выборки политики
        используют явные экземпляры np.random.default_rng(seed) для
        обеспечения правильной воспроизводимости.
        
        Старые версии Gym могут не реализовывать ``reset(seed=...)``; в этом
        случае вызов молча игнорируется.
    """

    try:
        env.reset(seed=seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
    except TypeError:
        # Legacy Gym versions may not accept reset(seed=...). Ignored.
        pass


def reset(env: gym.Env) -> int:
    """Сбрасывает среду и возвращает индекс начального состояния.

    Args:
        env: Среда для сброса.

    Returns:
        Целое число, представляющее начальное дискретное состояние.
    """

    obs, _ = env.reset()
    return int(obs)


def step(env: gym.Env, action: int) -> Tuple[int, float, bool, bool, Dict[str, np.ndarray]]:
    """Выполняет один шаг в среде, используя API Gym v0.26+.

    Args:
        env: Среда для взаимодействия.
        action: Дискретное действие, выбранное из политики.

    Returns:
        Кортеж, содержащий индекс следующего состояния, скалярную награду,
        флаг завершения (terminated), флаг усечения (truncated) и словарь
        с информацией (info), предоставляемый Gym.
    """

    obs, reward, terminated, truncated, info = env.step(action)
    return int(obs), float(reward), bool(terminated), bool(truncated), info


def moving_average(series: Iterable[float], window: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Вычисляет простое скользящее среднее для предоставленной последовательности.

    Args:
        series: Итерируемая последовательность числовых наблюдений.
        window: Номинальный размер окна; автоматически обрезается, если
            последовательность короче запрашиваемой ширины.

    Returns:
        Кортеж из (smoothed_values, x_indices), где x_indices скорректированы
        так, чтобы отражать центр каждого окна усреднения для правильного
        выравнивания при построении графиков.
    """

    data = np.asarray(list(series), dtype=float)
    if data.size == 0:
        return np.array([]), np.array([])
    width = max(1, min(window, data.size))
    kernel = np.ones(width, dtype=float) / width
    smoothed = np.convolve(data, kernel, mode="valid")
    # Adjust x-indices to center of averaging window
    x_indices = np.arange(width // 2, width // 2 + len(smoothed))
    return smoothed, x_indices


def uniform_policy(num_actions: int) -> Policy:
    """Создает политику, которая выбирает каждое действие с равной вероятностью.

    Args:
        num_actions: Размер дискретного пространства действий.

    Returns:
        Функция, которая отображает любое состояние в массив вероятностей
        действий.
    """

    probs = np.full(num_actions, 1.0 / num_actions, dtype=float)

    def _policy(_state: int) -> np.ndarray:
        return probs

    return _policy


# ------------------------- Monte Carlo prediction -------------------------

def generate_episode(
    env: gym.Env,
    policy: Policy,
    max_steps: int,
    rng: np.random.Generator,
) -> Tuple[List[int], List[float]]:
    """Генерирует один эпизод, следуя предоставленной политике.

    Args:
        env: Среда, из которой генерируется траектория.
        policy: Стохастическая политика, возвращающая вероятности действий
            для каждого состояния.
        max_steps: Максимальное количество переходов до усечения эпизода.
        rng: Генератор случайных чисел Numpy, используемый для выбора
            действий.

    Returns:
        Пара списков, содержащих посещенные состояния (включая
        терминальное состояние) и награды, собранные в ходе эпизода.
    """

    state = reset(env)
    states = [state]
    rewards: List[float] = []

    for _ in range(max_steps):
        action_probs = policy(state)
        action = int(rng.choice(len(action_probs), p=action_probs))
        next_state, reward, terminated, truncated, _ = step(env, action)
        rewards.append(reward)
        states.append(next_state)
        state = next_state
        if terminated or truncated:
            break

    return states, rewards


def monte_carlo_prediction(
    env: gym.Env,
    policy: Policy,
    episodes: int,
    gamma: float,
    max_steps: int,
    seed: int,
) -> Tuple[DefaultDict[int, float], List[float]]:
    """Оценивает ценность состояний, используя метод первого посещения
    Монте-Карло.

    Args:
        env: Среда для генерации эпизодов.
        policy: Политика поведения, используемая при сборе данных.
        episodes: Количество эпизодов для оценки.
        gamma: Коэффициент дисконтирования, применяемый к будущим
            вознаграждениям.
        max_steps: Ограничение на длину эпизода для избежания
            бесконечных циклов.
        seed: Seed для управления генерацией эпизодов.

    Returns:
        Словарь, отображающий состояния в их оцененные ценности, и
        история оценки начального состояния после каждого эпизода.
    """

    rng = np.random.default_rng(seed)
    value: DefaultDict[int, float] = defaultdict(float)
    counts: DefaultDict[int, float] = defaultdict(float)
    history: List[float] = []

    for _ in range(episodes):
        states, rewards = generate_episode(env, policy, max_steps, rng)
        G = 0.0
        visited: set[int] = set()

        for t in reversed(range(len(rewards))):
            state_t = states[t]
            G = gamma * G + rewards[t]
            if state_t in visited:
                continue
            counts[state_t] += 1.0
            value[state_t] += (G - value[state_t]) / counts[state_t]
            visited.add(state_t)

        history.append(value[0])  # track start-state estimate

    return value, history


# ------------------------- TD(0) prediction -------------------------

def td0_prediction(
    env: gym.Env,
    policy: Policy,
    episodes: int,
    alpha: float,
    gamma: float,
    max_steps: int,
    seed: int,
) -> Tuple[DefaultDict[int, float], List[float]]:
    """Оценивает ценность состояний с помощью обновлений TD(0).

    Args:
        env: Среда, с которой взаимодействует агент в режиме онлайн.
        policy: Политика поведения, используемая для выбора переходов.
        episodes: Количество разыгрываемых эпизодов.
        alpha: Постоянный размер шага для обновления TD.
        gamma: Коэффициент дисконтирования.
        max_steps: Максимальное количество шагов в эпизоде.
        seed: Seed для воспроизводимого выбора.

    Returns:
        Словарь с изученными ценностями состояний и траектория оценки
        начального состояния по эпизодам.
        
    Notes:
        Bootstrap обнуляется только тогда, когда эпизод завершается
        естественным образом (terminated=True), а не когда он просто
        усекается по лимиту времени.
    """

    rng = np.random.default_rng(seed)
    value: DefaultDict[int, float] = defaultdict(float)
    history: List[float] = []

    for _ in range(episodes):
        state = reset(env)

        for _ in range(max_steps):
            action_probs = policy(state)
            action = int(rng.choice(len(action_probs), p=action_probs))
            next_state, reward, terminated, truncated, _ = step(env, action)
            # Only zero bootstrap on true termination, not time-limit truncation
            td_target = reward if terminated else reward + gamma * value[next_state]
            value[state] += alpha * (td_target - value[state])
            state = next_state
            if terminated or truncated:
                break

        history.append(value[0])

    return value, history


# ------------------------- experiment runner -------------------------

def make_env(seed: int, slippery: bool) -> gym.Env:
    """Создает и инициализирует среду FrozenLake с необязательной
    стохастичностью.

    Args:
        seed: Базовый seed, передаваемый в среду и numpy.
        slippery: Если ``True``, используется стохастический (скользкий)
            вариант.

    Returns:
        Инициализированная среда Gym, готовая к взаимодействию.
    """

    env = gym.make("FrozenLake-v1", is_slippery=slippery)
    set_seeds(env, seed)
    return env


def value_to_grid(value: Dict[int, float], grid_size: Tuple[int, int]) -> np.ndarray:
    """Преобразует словарь скалярных значений в двумерную сетку.

    Args:
        value: Отображение индексов дискретных состояний в значения.
        grid_size: Кортеж ``(rows, cols)``, описывающий размер сетки.

    Returns:
        Двумерный массив numpy со значениями, размещенными в соответствии
        с их положением в сетке. Непосещенные состояния помечаются как
        NaN, чтобы отличать их от состояний с нулевым значением.
    """

    arr = np.full(grid_size, np.nan, dtype=float)
    for idx, val in value.items():
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        arr[row, col] = val
    return arr


def _annotate_frame(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    """Накладывает информационный текст на отображаемый RGB-кадр."""

    if not lines:
        return frame

    text = "\n".join(lines)
    base = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    try:
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        text_width, text_height = draw.multiline_textsize(text, font=font, spacing=4)

    padding = 8
    x, y = 12, 12
    draw.rectangle(
        (x - padding, y - padding, x + text_width + padding, y + text_height + padding),
        fill=(0, 0, 0, 160),
    )
    draw.multiline_text((x, y), text, fill=(255, 255, 255, 255), font=font, spacing=4)
    combined = Image.alpha_composite(base, overlay)
    return np.array(combined.convert("RGB"))


def _save_video(frames: List[np.ndarray], path: Path, fps: int = 15) -> None:
    """Сохраняет список RGB-кадров в виде MP4-файла."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def _simulate_episode(
    policy: Policy,
    max_steps: int,
    slippery: bool,
    seed: int,
    algorithm_name: str,
    episode_idx: int,
) -> Tuple[List[np.ndarray], float]:
    """Разыгрывает один эпизод и захватывает кадры с наложениями."""

    env = gym.make("FrozenLake-v1", is_slippery=slippery, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)

    def _labels(include_return: bool, total_reward: float) -> List[str]:
        lines = [algorithm_name, f"Episode {episode_idx + 1}"]
        if include_return:
            lines.append(f"Return {total_reward:.2f}")
        return lines

    rng = np.random.default_rng(seed)
    state = int(obs)
    total_reward = 0.0
    frames: List[np.ndarray] = []

    frame = env.render()
    frames.append(_annotate_frame(frame, _labels(include_return=False, total_reward=total_reward)))

    for step in range(max_steps):
        action_probs = policy(state)
        action = int(rng.choice(len(action_probs), p=action_probs))
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = int(obs)

        frame = env.render()
        done = bool(terminated or truncated)
        
        if done:
            # Episode ended naturally - append final frame with return overlay
            frames.append(
                _annotate_frame(
                    frame,
                    _labels(include_return=True, total_reward=total_reward),
                )
            )
            break
        
        # Intermediate frame without return overlay
        frames.append(
            _annotate_frame(
                frame,
                _labels(include_return=False, total_reward=total_reward),
            )
        )
    else:
        # Loop exited without break (hit max_steps) - append final frame
        frame = env.render()
        frames.append(
            _annotate_frame(
                frame,
                _labels(include_return=True, total_reward=total_reward),
            )
        )

    env.close()
    return frames, total_reward


def record_policy_rollouts(
    policy: Policy,
    episodes: int,
    max_steps: int,
    seed: int,
    slippery: bool,
    video_dir: str,
    algorithm_name: str,
    top_k: int = 3,
    fps: int = 15,
) -> None:
    """Записывает наиболее успешные эпизоды оценки с наложениями.

    Args:
        policy: Политика поведения, оцениваемая во время записи.
        episodes: Количество эпизодов оценки, генерируемых перед
            выбором лучших.
        max_steps: Ограничение на количество шагов в эпизоде.
        seed: Seed, контролирующий выбор действий во время записи.
        slippery: Использовать ли стохастическую динамику FrozenLake.
        video_dir: Целевой каталог для сгенерированных видеофайлов.
        algorithm_name: Метка, встраиваемая в имена файлов и наложения
            кадров.
        top_k: Количество лучших эпизодов для сохранения.
        fps: Частота кадров воспроизведения экспортируемых видео.
        
    Notes:
        Использует min-кучу для хранения только top-k эпизодов в памяти,
        предотвращая исчерпание памяти при генерации большого
        количества эпизодов.
    """
    import heapq

    target_dir = Path(video_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Min-heap of (reward, episode_idx, frames): maintains top-k automatically
    heap: List[Tuple[float, int, List[np.ndarray]]] = []
    
    for episode_idx in range(episodes):
        episode_seed = seed + episode_idx
        frames, total_reward = _simulate_episode(
            policy=policy,
            max_steps=max_steps,
            slippery=slippery,
            seed=episode_seed,
            algorithm_name=algorithm_name,
            episode_idx=episode_idx,
        )
        
        # If heap not full, add episode
        if len(heap) < top_k:
            heapq.heappush(heap, (total_reward, episode_idx, frames))
        # If current episode better than worst in heap, replace it
        elif total_reward > heap[0][0]:
            heapq.heapreplace(heap, (total_reward, episode_idx, frames))
        # Otherwise discard frames immediately (no buffering)

    if not heap:
        return

    # Sort heap by reward descending for proper ranking
    results = sorted(heap, key=lambda item: item[0], reverse=True)
    
    for rank, (total_reward, episode_idx, frames) in enumerate(results, start=1):
        filename = (
            f"{algorithm_name.lower().replace(' ', '_')}"
            f"_top{rank:02d}_episode_{episode_idx + 1:04d}_return_{total_reward:.2f}.mp4"
        )
        output_path = target_dir / filename
        _save_video(frames, output_path, fps=fps)
        print(
            f"Saved {algorithm_name} episode {episode_idx + 1} "
            f"(return={total_reward:.2f}) to {output_path.resolve()}"
        )


def run_comparison(
    episodes: int = 5000,
    gamma: float = 0.99,
    alpha: float = 0.1,
    max_steps: int = 200,
    slippery: bool = True,
    seed: int = 42,
    video_dir: Optional[str] = None,
    video_episodes: int = 3,
) -> None:
    """Сравнивает предсказания MC и TD(0) на FrozenLake при равномерной
    политике.

    Args:
        episodes: Количество эпизодов, используемых для обучения.
        gamma: Коэффициент дисконтирования, общий для обоих методов.
        alpha: Размер шага для TD(0).
        max_steps: Максимальное количество шагов в эпизоде.
        slippery: Включать ли стохастические переходы в FrozenLake.
        seed: Seed, контролирующий случайность в сравнении.
        video_dir: Необязательный каталог для записи оценочных
            эпизодов. Установите None (по умолчанию), чтобы
            отключить запись видео.
        video_episodes: Количество эпизодов для захвата при записи
            видео.

    Side Effects:
        Отображает графики matplotlib, суммирующие кривые обучения и
        сетки ценностей.
        
    Notes:
        Оба метода, MC и TD(0), теперь используют один и тот же
        экземпляр среды и поток seed, что обеспечивает идентичные
        траектории для справедливого сравнения.
    """

    # Use a SINGLE environment instance for both algorithms
    env = make_env(seed, slippery)
    policy = uniform_policy(env.action_space.n)

    # Both algorithms now use the same seed stream for fair comparison
    mc_values, mc_history = monte_carlo_prediction(
        env,
        policy,
        episodes=episodes,
        gamma=gamma,
        max_steps=max_steps,
        seed=seed,
    )

    # Reset environment to same initial state for TD
    set_seeds(env, seed)
    
    td_values, td_history = td0_prediction(
        env,
        policy,
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        max_steps=max_steps,
        seed=seed,  # Same seed as MC
    )

    plt.figure(figsize=(10, 5))
    mc_smooth, mc_x = moving_average(mc_history, 100)
    td_smooth, td_x = moving_average(td_history, 100)
    plt.plot(mc_x, mc_smooth, label="MC (first-visit)")
    plt.plot(td_x, td_smooth, label="TD(0)")
    plt.xlabel("Episodes")
    plt.ylabel("Estimate of V(start)")
    plt.title("Convergence of MC vs TD on FrozenLake (100-episode moving average)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    grid_shape = (4, 4)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(value_to_grid(mc_values, grid_shape), cmap="Blues")
    axes[0].set_title("MC value estimates")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(value_to_grid(td_values, grid_shape), cmap="Greens")
    axes[1].set_title("TD(0) value estimates")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks(range(grid_shape[1]))
        ax.set_yticks(range(grid_shape[0]))
        ax.grid(False)

    plt.tight_layout()
    plt.show()

    if video_dir and video_episodes > 0:
        record_seed = seed + 10_000
        base_dir = Path(video_dir)
        record_policy_rollouts(
            policy=policy,
            episodes=video_episodes,
            max_steps=max_steps,
            seed=record_seed,
            slippery=slippery,
            video_dir=base_dir / "mc",
            algorithm_name="MC",
        )
        record_policy_rollouts(
            policy=policy,
            episodes=video_episodes,
            max_steps=max_steps,
            seed=record_seed + 1,
            slippery=slippery,
            video_dir=base_dir / "td",
            algorithm_name="TD(0)",
        )

    env.close()


if __name__ == "__main__":
    run_comparison()
