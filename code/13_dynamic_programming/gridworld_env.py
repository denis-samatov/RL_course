"""
GridWorld environment for demonstrating Dynamic Programming.

A simple grid with obstacles and a goal state.
A fully observable, deterministic environment for studying DP.
"""

from typing import Tuple, List, Optional, Dict
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """
    A simple GridWorld environment.

    States: grid cells (i, j)
    Actions: 0=up, 1=down, 2=left, 3=right
    Rewards: -1 per step, +10 for reaching the goal
    Terminal state: Goal
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # Action display mapping
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
            height: Grid height
            width: Grid width
            obstacles: List of obstacle coordinates [(i1,j1), (i2,j2), ...]
            goal: Coordinates of the goal state (i, j)
            start: The agent's starting position (i, j)
            step_reward: The reward for every step
            goal_reward: The reward for reaching the goal
            render_mode: The rendering mode
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

        # Validate the parameters
        assert 0 <= goal[0] < height and 0 <= goal[1] < width, "Goal is outside the grid"
        assert 0 <= start[0] < height and 0 <= start[1] < width, "Start is outside the grid"
        assert goal not in self.obstacles, "Goal cannot be an obstacle"
        assert start not in self.obstacles, "Start cannot be an obstacle"

        # Gymnasium spaces
        self.observation_space = spaces.Discrete(height * width)
        self.action_space = spaces.Discrete(4)

        # The current state
        self.agent_pos: Optional[Tuple[int, int]] = None

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Converts (i, j) to a state index."""
        i, j = pos
        return i * self.width + j

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Converts a state index to (i, j)."""
        i = state // self.width
        j = state % self.width
        return (i, j)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[int, Dict]:
        """Resets the environment to its starting state."""
        super().reset(seed=seed)
        self.agent_pos = self.start
        return self._pos_to_state(self.agent_pos), {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """
        Takes an action in the environment.

        Args:
            action: 0=up, 1=down, 2=left, 3=right

        Returns:
            observation: The new state (index)
            reward: The reward
            terminated: True if the goal was reached
            truncated: False (the episode never truncates)
            info: Extra info
        """
        assert self.agent_pos is not None, "Call reset() before step()"

        i, j = self.agent_pos

        # Compute the new position
        if action == 0:  # up
            new_pos = (max(0, i - 1), j)
        elif action == 1:  # down
            new_pos = (min(self.height - 1, i + 1), j)
        elif action == 2:  # left
            new_pos = (i, max(0, j - 1))
        elif action == 3:  # right
            new_pos = (i, min(self.width - 1, j + 1))
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check for obstacles
        if new_pos in self.obstacles:
            new_pos = self.agent_pos  # Stay in place

        self.agent_pos = new_pos

        # Compute the reward and termination
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
        Returns the environment's dynamics: P(s'|s,a).

        For a deterministic environment, returns a single entry [(1.0, s', r, done)].

        Args:
            state: The state index
            action: The action index

        Returns:
            List[(prob, next_state, reward, done)]
        """
        pos = self._state_to_pos(state)

        # If already at the goal, stay there
        if pos == self.goal:
            return [(1.0, state, 0.0, True)]

        i, j = pos

        # Compute the next position
        if action == 0:  # up
            new_pos = (max(0, i - 1), j)
        elif action == 1:  # down
            new_pos = (min(self.height - 1, i + 1), j)
        elif action == 2:  # left
            new_pos = (i, max(0, j - 1))
        elif action == 3:  # right
            new_pos = (i, min(self.width - 1, j + 1))
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check for obstacles
        if new_pos in self.obstacles:
            new_pos = pos  # Stay in place

        next_state = self._pos_to_state(new_pos)

        # Reward and termination
        if new_pos == self.goal:
            reward = self.goal_reward
            done = True
        else:
            reward = self.step_reward
            done = False

        return [(1.0, next_state, reward, done)]

    def render(self) -> Optional[str]:
        """Visualizes the current state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _render_ansi(self) -> str:
        """A text visualization of the grid."""
        grid = []

        # Top border
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

            # Horizontal border
            if i < self.height - 1:
                grid.append("├" + "───┼" * (self.width - 1) + "───┤")

        # Bottom border
        grid.append("└" + "───┴" * (self.width - 1) + "───┘")

        return "\n".join(grid)


# Register the environment with Gymnasium
gym.register(
    id="GridWorld-v0",
    entry_point="gridworld_env:GridWorldEnv",
    max_episode_steps=100,
)


if __name__ == "__main__":
    # A demonstration of the environment
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

    print("\nRunning a sequence of actions: right, right, up, up, right")
    actions = [3, 3, 0, 0, 3]  # right, right, up, up, right

    for step, action in enumerate(actions, 1):
        print(f"\nStep {step}: {GridWorldEnv.ACTION_NAMES[action]}")
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Reward: {reward}, Terminated: {terminated}")

        if terminated:
            print("\n🎉 Goal reached!")
            break
