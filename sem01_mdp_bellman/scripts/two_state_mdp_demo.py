"""Демонстрация для простого MDP с двумя состояниями.

Определение MDP:
  S = {0=s1, 1=s2}, A = {0,1}
  - a=0: s1->s2 с r=+1; s2->s2 с r=0 (поглощающее)
  - a=1: s1->s1 с r=0;  s2->s2 с r=0 (поглощающее)

Оптимальная стратегия: в s1 выбирать a=0, в s2 — любое действие.
Тогда V*(s1)=1, V*(s2)=0 для любого gamma в [0,1).

Скрипт выполняет:
  1) Аналитически вычисляет V*, Q* и проверяет базовые тождества.
  2) Симулирует эпизоды с оптимальной жадной политикой и сравнивает
     среднюю награду с V*(s1).
  3) Демонстрирует сходимость итерации по ценности (Value Iteration)
     и затухание невязки Беллмана.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


S1, S2 = 0, 1
A0, A1 = 0, 1


@dataclass
class TwoStateMDP:
    """Класс, представляющий MDP с двумя состояниями."""
    gamma: float = 0.99

    def step(self, s: int, a: int):
        """Выполняет шаг в среде.

        Args:
            s (int): Текущее состояние (S1 или S2).
            a (int): Действие (A0 или A1).

        Returns:
            tuple: (следующее состояние, награда, done, truncated).
        """
        if s == S1:
            if a == A0:  # s1 --a0--> s2, r=+1
                return S2, 1.0, True, False
            else:        # s1 --a1--> s1, r=0
                return S1, 0.0, False, False
        else:            # s2 absorbing, r=0
            return S2, 0.0, True, False

    def V_star(self):
        """Возвращает аналитически вычисленную оптимальную V-функцию."""
        # V*(s2)=0; V*(s1)=max{1+gamma*V*(s2), 0+gamma*V*(s1)} = 1
        return np.array([1.0, 0.0], dtype=float)

    def Q_star(self):
        """Возвращает аналитически вычисленную оптимальную Q-функцию."""
        # Q*(s1,0)=1+gamma*V(s2)=1; Q*(s1,1)=0+gamma*V(s1)=0
        Q = np.zeros((2, 2), dtype=float)
        Q[S1, A0] = 1.0
        Q[S1, A1] = 0.0
        return Q


def simulate(env: TwoStateMDP, policy, episodes=1000, seed=42):
    """Симулирует эпизоды и оценивает среднюю награду.

    Args:
        env (TwoStateMDP): Среда.
        policy (callable): Функция политики.
        episodes (int): Количество эпизодов.
        seed (int): Зерно для ГСЧ.

    Returns:
        tuple[float, float]: Средняя и стандартное отклонение наград.
    """
    rs = np.random.RandomState(seed)
    totals = []
    for _ in range(episodes):
        s = S1  # для наглядности начинаем всегда с s1
        done = trunc = False
        G = 0.0
        t = 0
        while not (done or trunc):
            a = policy(s)
            s_next, r, done, trunc = env.step(s, a)
            G += (env.gamma ** t) * r
            s = s_next
            t += 1
            if t > 500: # защита от бесконечных циклов
                trunc = True
        totals.append(G)
    return float(np.mean(totals)), float(np.std(totals))


def greedy_star_policy(s):
    """Оптимальная жадная политика."""
    return A0 if s == S1 else A0


def value_iteration(env: TwoStateMDP, iters=10):
    """Выполняет итерацию по ценности (Value Iteration).

    Args:
        env (TwoStateMDP): Среда.
        iters (int): Количество итераций.

    Returns:
        tuple[np.ndarray, np.ndarray]: V-функция и история невязок.
    """
    V = np.zeros(2, dtype=float)
    residuals = []
    for _ in range(iters):
        V_new = np.zeros_like(V)
        # s1: max по действиям
        v_a0 = 1.0 + env.gamma * 0.0   # переход в s2
        v_a1 = 0.0 + env.gamma * V[S1] # остаться в s1
        V_new[S1] = max(v_a0, v_a1)
        # s2: поглощающее состояние
        V_new[S2] = 0.0
        residuals.append(float(np.max(np.abs(V_new - V))))
        V = V_new
    return V, np.array(residuals, dtype=float)


def main():
    """Основная функция для запуска демонстрации."""
    env = TwoStateMDP(gamma=0.99)
    V_star = env.V_star()
    Q_star = env.Q_star()

    # 1) Аналитика
    assert np.allclose(V_star, [1.0, 0.0])
    assert np.isclose(Q_star[S1, A0], 1.0)
    assert np.isclose(Q_star[S1, A1], 0.0)

    # 2) Симуляция с оптимальной политикой
    mean, std = simulate(env, greedy_star_policy, episodes=500)
    print(f"Награда из симуляции (старт s1): mean={mean:.3f} ± {std:.3f}")
    assert abs(mean - V_star[S1]) < 0.1, "Симуляция должна сходиться к V*(s1)≈1"

    # 3) Сходимость Value Iteration
    V_vi, residuals = value_iteration(env, iters=8)
    print("Невязки (residuals) на итерациях:", residuals.tolist())
    print("V после VI:", V_vi.tolist())
    assert residuals[-1] <= residuals[0] + 1e-9, "Невязка не должна расти"
    print("✓ Демонстрация MDP с 2 состояниями завершена успешно")


if __name__ == "__main__":
    main()

