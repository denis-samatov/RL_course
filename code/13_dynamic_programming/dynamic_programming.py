"""
Реализация алгоритмов динамического программирования:
- Policy Evaluation
- Policy Iteration  
- Value Iteration

Все алгоритмы работают на дискретных средах Gymnasium с известной моделью.
"""

from typing import Dict, Tuple, Callable, Optional
import numpy as np
from tqdm import tqdm


PolicyType = Dict[int, np.ndarray]  # {state: distribution over actions}
ValueFunctionType = np.ndarray  # V[state] = value


def policy_evaluation(
    env,
    policy: PolicyType,
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_iterations: int = 1000,
    verbose: bool = False,
) -> Tuple[ValueFunctionType, int]:
    """
    Policy Evaluation: вычисляет V_π(s) для заданной политики.
    
    Args:
        env: Среда Gymnasium с методом get_transition_prob(state, action)
        policy: Словарь {state: probs_over_actions}
        gamma: Discount factor
        theta: Порог сходимости
        max_iterations: Максимум итераций
        verbose: Печатать прогресс
        
    Returns:
        V: Функция ценности V_π(s)
        num_iterations: Количество итераций до сходимости
    """
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    
    for iteration in range(max_iterations):
        delta = 0
        V_new = np.zeros(n_states)
        
        for s in range(n_states):
            v = 0.0
            
            # Суммируем по действиям с учётом политики
            for a, prob_a in enumerate(policy[s]):
                if prob_a == 0:
                    continue
                
                # Получаем динамику среды: [(prob, next_s, reward, done)]
                transitions = env.get_transition_prob(s, a)
                
                for prob_transition, s_prime, reward, done in transitions:
                    # Беллмановское обновление
                    v += prob_a * prob_transition * (
                        reward + gamma * V[s_prime] * (1 - int(done))
                    )
            
            V_new[s] = v
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: delta = {delta:.6f}")
        
        if delta < theta:
            if verbose:
                print(f"Policy Evaluation converged in {iteration + 1} iterations")
            return V, iteration + 1
    
    if verbose:
        print(f"Policy Evaluation reached max iterations ({max_iterations})")
    return V, max_iterations


def policy_improvement(
    env,
    V: ValueFunctionType,
    gamma: float = 0.9,
) -> PolicyType:
    """
    Policy Improvement: улучшает политику жадным выбором на основе V(s).
    
    Args:
        env: Среда Gymnasium
        V: Текущая функция ценности
        gamma: Discount factor
        
    Returns:
        Улучшенная детерминированная политика (greedy)
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    policy = {}
    
    for s in range(n_states):
        # Вычисляем Q(s,a) для всех действий
        q_values = np.zeros(n_actions)
        
        for a in range(n_actions):
            transitions = env.get_transition_prob(s, a)
            q_sa = 0.0
            
            for prob, s_prime, reward, done in transitions:
                q_sa += prob * (reward + gamma * V[s_prime] * (1 - int(done)))
            
            q_values[a] = q_sa
        
        # Жадный выбор: deterministic policy
        best_action = np.argmax(q_values)
        policy[s] = np.eye(n_actions)[best_action]  # one-hot encoding
    
    return policy


def policy_iteration(
    env,
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_iterations: int = 100,
    verbose: bool = False,
) -> Tuple[PolicyType, ValueFunctionType, int]:
    """
    Policy Iteration: чередует Policy Evaluation и Policy Improvement.
    
    Args:
        env: Среда Gymnasium
        gamma: Discount factor
        theta: Порог сходимости для evaluation
        max_iterations: Максимум итераций improvement
        verbose: Печатать прогресс
        
    Returns:
        optimal_policy: Оптимальная политика π*
        V: Оптимальная функция ценности V*
        num_iterations: Количество итераций до сходимости
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Инициализация: uniform random policy
    policy = {
        s: np.ones(n_actions) / n_actions
        for s in range(n_states)
    }
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n=== Policy Iteration: Iteration {iteration + 1} ===")
        
        # 1. Policy Evaluation
        V, eval_iters = policy_evaluation(
            env, policy, gamma, theta, max_iterations=1000, verbose=False
        )
        
        if verbose:
            print(f"Policy Evaluation converged in {eval_iters} iterations")
        
        # 2. Policy Improvement
        new_policy = policy_improvement(env, V, gamma)
        
        # 3. Проверка стабильности политики
        policy_stable = True
        for s in range(n_states):
            if not np.allclose(policy[s], new_policy[s]):
                policy_stable = False
                break
        
        if verbose:
            print(f"Policy stable: {policy_stable}")
        
        policy = new_policy
        
        if policy_stable:
            if verbose:
                print(f"\nPolicy Iteration converged in {iteration + 1} iterations")
            return policy, V, iteration + 1
    
    if verbose:
        print(f"\nPolicy Iteration reached max iterations ({max_iterations})")
    return policy, V, max_iterations


def value_iteration(
    env,
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_iterations: int = 1000,
    verbose: bool = False,
) -> Tuple[PolicyType, ValueFunctionType, int]:
    """
    Value Iteration: итеративно применяет оптимальное уравнение Беллмана.
    
    Args:
        env: Среда Gymnasium
        gamma: Discount factor
        theta: Порог сходимости
        max_iterations: Максимум итераций
        verbose: Печатать прогресс
        
    Returns:
        optimal_policy: Оптимальная политика π*
        V: Оптимальная функция ценности V*
        num_iterations: Количество итераций до сходимости
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    
    for iteration in range(max_iterations):
        delta = 0
        V_new = np.zeros(n_states)
        
        for s in range(n_states):
            # Вычисляем Q(s,a) для всех действий
            q_values = np.zeros(n_actions)
            
            for a in range(n_actions):
                transitions = env.get_transition_prob(s, a)
                q_sa = 0.0
                
                for prob, s_prime, reward, done in transitions:
                    q_sa += prob * (reward + gamma * V[s_prime] * (1 - int(done)))
                
                q_values[a] = q_sa
            
            # Оптимальное обновление Беллмана
            V_new[s] = np.max(q_values)
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        
        if verbose and iteration % 50 == 0:
            print(f"Iteration {iteration}: delta = {delta:.6f}")
        
        if delta < theta:
            if verbose:
                print(f"Value Iteration converged in {iteration + 1} iterations")
            break
    else:
        if verbose:
            print(f"Value Iteration reached max iterations ({max_iterations})")
    
    # Извлекаем оптимальную политику
    policy = policy_improvement(env, V, gamma)
    
    return policy, V, iteration + 1


def extract_greedy_policy(
    env,
    V: ValueFunctionType,
    gamma: float = 0.9,
) -> PolicyType:
    """
    Извлекает жадную политику из функции ценности V.
    
    Эквивалентно policy_improvement, но более явное название.
    """
    return policy_improvement(env, V, gamma)


def evaluate_policy_monte_carlo(
    env,
    policy: PolicyType,
    n_episodes: int = 1000,
    max_steps: int = 100,
    gamma: float = 0.9,
    seed: Optional[int] = None,
) -> ValueFunctionType:
    """
    Оценивает политику методом Монте-Карло (для проверки).
    
    Args:
        env: Среда Gymnasium
        policy: Политика для оценки
        n_episodes: Количество эпизодов
        max_steps: Максимум шагов в эпизоде
        gamma: Discount factor
        seed: Random seed
        
    Returns:
        V: Эмпирическая оценка V_π(s)
    """
    n_states = env.observation_space.n
    returns_sum = np.zeros(n_states)
    returns_count = np.zeros(n_states)
    
    rng = np.random.default_rng(seed)
    
    for ep in tqdm(range(n_episodes), desc="MC Evaluation", leave=False):
        # Генерируем эпизод
        state, _ = env.reset()
        trajectory = []
        
        for t in range(max_steps):
            action = rng.choice(env.action_space.n, p=policy[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, reward))
            state = next_state
            
            if terminated or truncated:
                break
        
        # Вычисляем returns для каждого состояния в эпизоде
        G = 0
        visited_states = set()
        
        for state, reward in reversed(trajectory):
            G = reward + gamma * G
            
            # First-visit MC
            if state not in visited_states:
                returns_sum[state] += G
                returns_count[state] += 1
                visited_states.add(state)
    
    # Усредняем
    V = np.divide(
        returns_sum,
        returns_count,
        out=np.zeros_like(returns_sum),
        where=returns_count > 0,
    )
    
    return V


if __name__ == "__main__":
    # Демонстрация алгоритмов
    from gridworld_env import GridWorldEnv
    
    print("=== Dynamic Programming Demo ===\n")
    
    # Создаём среду
    env = GridWorldEnv(
        height=4,
        width=4,
        obstacles=[(1, 1)],
        goal=(0, 3),
        start=(3, 0),
        step_reward=-1.0,
        goal_reward=10.0,
    )
    
    gamma = 0.9
    
    # 1. Policy Iteration
    print("1. Policy Iteration:")
    policy_pi, V_pi, iters_pi = policy_iteration(
        env, gamma=gamma, verbose=True
    )
    print(f"Converged in {iters_pi} iterations")
    print(f"Max V value: {V_pi.max():.2f}\n")
    
    # 2. Value Iteration
    print("2. Value Iteration:")
    policy_vi, V_vi, iters_vi = value_iteration(
        env, gamma=gamma, verbose=True
    )
    print(f"Converged in {iters_vi} iterations")
    print(f"Max V value: {V_vi.max():.2f}\n")
    
    # 3. Сравнение результатов
    print("3. Comparison:")
    print(f"V functions close: {np.allclose(V_pi, V_vi, atol=1e-3)}")
    print(f"Max difference: {np.abs(V_pi - V_vi).max():.6f}")
    
    # Визуализация V-функций
    print("\nValue Function (Policy Iteration):")
    V_grid = V_pi.reshape(env.height, env.width)
    print(V_grid)
    
    print("\nValue Function (Value Iteration):")
    V_grid = V_vi.reshape(env.height, env.width)
    print(V_grid)

