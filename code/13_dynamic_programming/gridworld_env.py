"""
GridWorld Environment для демонстрации Dynamic Programming.

Простая сетка с препятствиями и целевым состоянием.
Полностью наблюдаемая, детерминированная среда для обучения DP.
"""

from typing import Tuple, List, Optional, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """
    Простая среда GridWorld.
    
    Состояния: клетки сетки (i, j)
    Действия: 0=вверх, 1=вниз, 2=влево, 3=вправо
    Награды: -1 за каждый шаг, +10 за достижение цели
    Терминальное состояние: Goal
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    # Отображение действий
    ACTION_TO_STR = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}
    
    def __init__(
        self,
        height: int = 4,
        width: int = 4,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        goal: Tuple[int, int] = (0, 3),
        start: Tuple[int, int] = (3, 0),
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            height: Высота сетки
            width: Ширина сетки
            obstacles: Список координат препятствий [(i1,j1), (i2,j2), ...]
            goal: Координаты целевого состояния (i, j)
            start: Начальная позиция агента (i, j)
            step_reward: Награда за каждый шаг
            goal_reward: Награда за достижение цели
            render_mode: Режим рендеринга
        """
        super().__init__()
        
        self.height = height
        self.width = width
        self.obstacles = set(obstacles or [])
        self.goal = goal
        self.start = start
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.render_mode = render_mode
        
        # Проверка корректности параметров
        assert 0 <= goal[0] < height and 0 <= goal[1] < width, "Goal вне сетки"
        assert 0 <= start[0] < height and 0 <= start[1] < width, "Start вне сетки"
        assert goal not in self.obstacles, "Goal не может быть препятствием"
        assert start not in self.obstacles, "Start не может быть препятствием"
        
        # Gymnasium spaces
        self.observation_space = spaces.Discrete(height * width)
        self.action_space = spaces.Discrete(4)
        
        # Текущее состояние
        self.agent_pos: Optional[Tuple[int, int]] = None
        
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Конвертирует (i, j) в индекс состояния."""
        i, j = pos
        return i * self.width + j
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Конвертирует индекс состояния в (i, j)."""
        i = state // self.width
        j = state % self.width
        return (i, j)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[int, Dict]:
        """Сброс среды в начальное состояние."""
        super().reset(seed=seed)
        self.agent_pos = self.start
        return self._pos_to_state(self.agent_pos), {}
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """
        Выполняет действие в среде.
        
        Args:
            action: 0=up, 1=down, 2=left, 3=right
            
        Returns:
            observation: Новое состояние (индекс)
            reward: Награда
            terminated: True если достигли цели
            truncated: False (эпизод не обрывается)
            info: Дополнительная информация
        """
        assert self.agent_pos is not None, "Вызовите reset() перед step()"
        
        i, j = self.agent_pos
        
        # Вычисляем новую позицию
        if action == 0:  # up
            new_pos = (max(0, i - 1), j)
        elif action == 1:  # down
            new_pos = (min(self.height - 1, i + 1), j)
        elif action == 2:  # left
            new_pos = (i, max(0, j - 1))
        elif action == 3:  # right
            new_pos = (i, min(self.width - 1, j + 1))
        else:
            raise ValueError(f"Неверное действие: {action}")
        
        # Проверяем препятствия
        if new_pos in self.obstacles:
            new_pos = self.agent_pos  # Остаёмся на месте
        
        self.agent_pos = new_pos
        
        # Вычисляем награду и терминацию
        if self.agent_pos == self.goal:
            reward = self.goal_reward
            terminated = True
        else:
            reward = self.step_reward
            terminated = False
        
        observation = self._pos_to_state(self.agent_pos)
        
        return observation, reward, terminated, False, {}
    
    def get_transition_prob(
        self, state: int, action: int
    ) -> List[Tuple[float, int, float, bool]]:
        """
        Возвращает динамику среды: P(s'|s,a).
        
        Для детерминированной среды возвращает одну запись [(1.0, s', r, done)].
        
        Args:
            state: Индекс состояния
            action: Индекс действия
            
        Returns:
            List[(prob, next_state, reward, done)]
        """
        pos = self._state_to_pos(state)
        
        # Если уже в цели — остаёмся там
        if pos == self.goal:
            return [(1.0, state, 0.0, True)]
        
        i, j = pos
        
        # Вычисляем следующую позицию
        if action == 0:  # up
            new_pos = (max(0, i - 1), j)
        elif action == 1:  # down
            new_pos = (min(self.height - 1, i + 1), j)
        elif action == 2:  # left
            new_pos = (i, max(0, j - 1))
        elif action == 3:  # right
            new_pos = (i, min(self.width - 1, j + 1))
        else:
            raise ValueError(f"Неверное действие: {action}")
        
        # Проверяем препятствия
        if new_pos in self.obstacles:
            new_pos = pos  # Остаёмся на месте
        
        next_state = self._pos_to_state(new_pos)
        
        # Награда и терминация
        if new_pos == self.goal:
            reward = self.goal_reward
            done = True
        else:
            reward = self.step_reward
            done = False
        
        return [(1.0, next_state, reward, done)]
    
    def render(self) -> Optional[str]:
        """Визуализация текущего состояния."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None
    
    def _render_ansi(self) -> str:
        """Текстовая визуализация сетки."""
        grid = []
        
        # Верхняя граница
        grid.append("┌" + "───┬" * (self.width - 1) + "───┐")
        
        for i in range(self.height):
            row = "│"
            for j in range(self.width):
                pos = (i, j)
                if pos == self.agent_pos:
                    cell = " A "  # Agent
                elif pos == self.goal:
                    cell = " G "  # Goal
                elif pos in self.obstacles:
                    cell = " X "  # Obstacle
                else:
                    cell = "   "
                row += cell
                if j < self.width - 1:
                    row += "│"
            row += "│"
            grid.append(row)
            
            # Горизонтальная граница
            if i < self.height - 1:
                grid.append("├" + "───┼" * (self.width - 1) + "───┤")
        
        # Нижняя граница
        grid.append("└" + "───┴" * (self.width - 1) + "───┘")
        
        return "\n".join(grid)


# Регистрация среды в Gymnasium
gym.register(
    id="GridWorld-v0",
    entry_point="gridworld_env:GridWorldEnv",
    max_episode_steps=100,
)


if __name__ == "__main__":
    # Демонстрация среды
    env = GridWorldEnv(
        height=4,
        width=4,
        obstacles=[(1, 1), (2, 2)],
        goal=(0, 3),
        start=(3, 0),
        render_mode="human",
    )
    
    print("GridWorld Environment Demo\n")
    env.reset()
    env.render()
    
    print("\nВыполняем последовательность действий: right, right, up, up, right")
    actions = [3, 3, 0, 0, 3]  # right, right, up, up, right
    
    for step, action in enumerate(actions, 1):
        print(f"\nШаг {step}: {GridWorldEnv.ACTION_NAMES[action]}")
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Награда: {reward}, Терминация: {terminated}")
        
        if terminated:
            print("\n🎉 Цель достигнута!")
            break

