"""
Visualizations for Dynamic Programming results.

Produces heatmaps of V(s), policy arrows, and a convergence animation.
"""

from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import seaborn as sns

from gridworld_env import GridWorldEnv
from dynamic_programming import (
    policy_iteration,
    value_iteration,
    policy_evaluation,
)


def visualize_value_function(
    env: GridWorldEnv,
    V: np.ndarray,
    title: str = "Value Function",
    save_path: str = None,
):
    """
    Visualizes the value function as a heatmap.

    Args:
        env: The GridWorld environment
        V: The value function (a flat array)
        title: The plot title
        save_path: Where to save (if None, displays instead)
    """
    V_grid = V.reshape(env.height, env.width)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Heatmap
    sns.heatmap(
        V_grid,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Value"},
        linewidths=0.5,
        ax=ax,
    )

    # Mark obstacles
    for obs in env.obstacles:
        i, j = obs
        ax.add_patch(mpatches.Rectangle(
            (j, i), 1, 1,
            fill=True,
            facecolor='gray',
            edgecolor='black',
            linewidth=2,
        ))
        ax.text(j + 0.5, i + 0.5, "X", ha='center', va='center',
                fontsize=20, color='white', weight='bold')

    # Mark the goal
    goal_i, goal_j = env.goal
    ax.add_patch(mpatches.Rectangle(
        (goal_j, goal_i), 1, 1,
        fill=False,
        edgecolor='red',
        linewidth=3,
    ))
    ax.text(goal_j + 0.5, goal_i + 0.5, "G", ha='center', va='center',
            fontsize=20, color='red', weight='bold')

    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel("Column (j)")
    ax.set_ylabel("Row (i)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_policy(
    env: GridWorldEnv,
    policy: Dict[int, np.ndarray],
    V: np.ndarray = None,
    title: str = "Optimal Policy",
    save_path: str = None,
):
    """
    Visualizes the policy as arrows over a V(s) background.

    Args:
        env: The GridWorld environment
        policy: The policy {state: distribution over actions}
        V: The value function (optional, used as the background)
        title: The plot title
        save_path: Where to save
    """
    if V is not None:
        V_grid = V.reshape(env.height, env.width)
    else:
        V_grid = np.zeros((env.height, env.width))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Background heatmap
    im = ax.imshow(V_grid, cmap="YlGnBu", alpha=0.6)
    cbar = plt.colorbar(im, ax=ax, label="Value")

    # Arrow directions
    action_to_arrow = {
        0: (0, -1),    # up
        1: (0, 1),     # down
        2: (-1, 0),    # left
        3: (1, 0),     # right
    }

    # Draw an arrow for every state
    for s in range(env.observation_space.n):
        i, j = env._state_to_pos(s)
        pos = (i, j)

        # Skip obstacles and the goal
        if pos in env.obstacles or pos == env.goal:
            continue

        # Determine the best action
        action_probs = policy[s]
        best_action = np.argmax(action_probs)

        # Arrow coordinates
        dx, dy = action_to_arrow[best_action]

        ax.arrow(
            j, i,
            dx * 0.3, dy * 0.3,
            head_width=0.2,
            head_length=0.15,
            fc='red',
            ec='darkred',
            linewidth=2,
        )

    # Mark obstacles
    for obs in env.obstacles:
        i, j = obs
        ax.add_patch(mpatches.Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            fill=True,
            facecolor='gray',
            edgecolor='black',
            linewidth=2,
        ))
        ax.text(j, i, "X", ha='center', va='center',
                fontsize=20, color='white', weight='bold')

    # Mark the goal
    goal_i, goal_j = env.goal
    ax.add_patch(mpatches.Rectangle(
        (goal_j - 0.5, goal_i - 0.5), 1, 1,
        fill=False,
        edgecolor='red',
        linewidth=3,
    ))
    ax.text(goal_j, goal_i, "G", ha='center', va='center',
            fontsize=20, color='red', weight='bold')

    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel("Column (j)")
    ax.set_ylabel("Row (i)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def compare_algorithms(
    env: GridWorldEnv,
    gamma: float = 0.9,
    save_path: str = None,
):
    """
    Compares Policy Iteration and Value Iteration.

    Plots the convergence and the final V-functions.
    """
    print("Running Policy Iteration...")
    policy_pi, V_pi, iters_pi = policy_iteration(env, gamma=gamma, verbose=True)

    print("\nRunning Value Iteration...")
    policy_vi, V_vi, iters_vi = value_iteration(env, gamma=gamma, verbose=True)

    # Comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Policy Iteration
    V_grid_pi = V_pi.reshape(env.height, env.width)
    sns.heatmap(V_grid_pi, annot=True, fmt=".2f", cmap="YlGnBu",
                cbar_kws={"label": "Value"}, ax=axes[0])
    axes[0].set_title(f"Policy Iteration\n({iters_pi} iterations)", weight='bold')

    # Value Iteration
    V_grid_vi = V_vi.reshape(env.height, env.width)
    sns.heatmap(V_grid_vi, annot=True, fmt=".2f", cmap="YlGnBu",
                cbar_kws={"label": "Value"}, ax=axes[1])
    axes[1].set_title(f"Value Iteration\n({iters_vi} iterations)", weight='bold')

    # Difference
    diff = V_pi - V_vi
    max_diff = np.abs(diff).max()
    sns.heatmap(diff.reshape(env.height, env.width),
                annot=True, fmt=".4f", cmap="RdBu_r", center=0,
                cbar_kws={"label": "Difference"}, ax=axes[2])
    axes[2].set_title(f"Difference (max={max_diff:.6f})", weight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()

    return policy_pi, V_pi, policy_vi, V_vi


def animate_value_iteration(
    env: GridWorldEnv,
    gamma: float = 0.9,
    save_path: str = "value_iteration.gif",
    interval: int = 200,
):
    """
    Creates an animation of the Value Iteration process.

    Args:
        env: The GridWorld environment
        gamma: Discount factor
        save_path: Where to save the GIF
        interval: The interval between frames (ms)
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)

    V_history = [V.copy()]
    theta = 1e-6
    max_iterations = 100

    # Collect the history of V
    for iteration in range(max_iterations):
        V_new = np.zeros(n_states)
        delta = 0

        for s in range(n_states):
            q_values = np.zeros(n_actions)

            for a in range(n_actions):
                transitions = env.get_transition_prob(s, a)
                q_sa = 0.0

                for prob, s_prime, reward, done in transitions:
                    q_sa += prob * (reward + gamma * V[s_prime] * (1 - int(done)))

                q_values[a] = q_sa

            V_new[s] = np.max(q_values)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        V_history.append(V.copy())

        if delta < theta:
            break

    # Build the animation
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        ax.clear()
        V_grid = V_history[frame].reshape(env.height, env.width)

        im = ax.imshow(V_grid, cmap="YlGnBu", vmin=V_history[-1].min(),
                       vmax=V_history[-1].max())

        for i in range(env.height):
            for j in range(env.width):
                text = ax.text(j, i, f"{V_grid[i, j]:.2f}",
                              ha="center", va="center", color="black")

        # Obstacles and the goal
        for obs in env.obstacles:
            i, j = obs
            ax.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=True, facecolor='gray', edgecolor='black', linewidth=2
            ))

        goal_i, goal_j = env.goal
        ax.add_patch(mpatches.Rectangle(
            (goal_j - 0.5, goal_i - 0.5), 1, 1,
            fill=False, edgecolor='red', linewidth=3
        ))

        ax.set_title(f"Value Iteration: Iteration {frame}/{len(V_history)-1}",
                    weight='bold')
        ax.set_xlim(-0.5, env.width - 0.5)
        ax.set_ylim(env.height - 0.5, -0.5)
        ax.set_xticks(range(env.width))
        ax.set_yticks(range(env.height))
        ax.grid(True, alpha=0.3)

        return [im]

    anim = FuncAnimation(fig, update, frames=len(V_history),
                        interval=interval, repeat=True)

    anim.save(save_path, writer='pillow', fps=5)
    print(f"Animation saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    # A visualization demo
    env = GridWorldEnv(
        height=4,
        width=4,
        obstacles=[(1, 1), (2, 2)],
        goal=(0, 3),
        start=(3, 0),
    )

    gamma = 0.9

    # 1. Comparing the algorithms
    print("=== Comparing Policy Iteration vs Value Iteration ===")
    policy_pi, V_pi, policy_vi, V_vi = compare_algorithms(
        env, gamma=gamma, save_path="dp_comparison.png"
    )

    # 2. Visualizing the policy
    print("\n=== Visualizing Optimal Policy ===")
    visualize_policy(env, policy_vi, V_vi,
                    title="Optimal Policy (Value Iteration)",
                    save_path="optimal_policy.png")

    # 3. Animating Value Iteration
    print("\n=== Creating Animation ===")
    animate_value_iteration(env, gamma=gamma, save_path="value_iteration.gif")

    print("\n✅ All visualizations complete!")
