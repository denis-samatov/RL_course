# Теоретический конспект №11

## Тема: Policy Gradient Methods и REINFORCE

> **Связано с:** [note_04_policy_vs_value_methods.md](note_04_policy_vs_value_methods.md) — Policy-Based методы · [note_09_q_learning.md](note_09_q_learning.md) — Q-Learning · [note_10_deep_q_network.md](note_10_deep_q_network.md) — Deep Q-Network

---

## 1. От value-based к policy-based: зачем нужны Policy Gradients?

До этого момента мы изучали **value-based методы** (Q-Learning, DQN), которые оценивают ценность действий и выводят политику косвенно:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

Но такой подход имеет ограничения:

* **Дискретные действия** — в непрерывных пространствах (управление роботом, автопилот) поиск $\arg\max$ становится вычислительно невозможным.
* **Детерминированность** — жадная политика всегда выбирает одно и то же действие, что может быть субоптимально в стохастических средах.
* **Нестабильность** — небольшие изменения Q-значений могут резко изменить политику.

**Решение:** учить политику $\pi_\theta(a|s)$ напрямую, где $\theta$ — параметры (веса нейросети).

---

## 2. Интуиция: что такое Policy Gradient?

**Policy Gradient** — это семейство методов, которые обучают политику, **напрямую максимизируя ожидаемое вознаграждение** через градиентный подъём.

### Идея

Вместо вопроса «Насколько хорошо действие $a$ в состоянии $s$?» (Q-Learning)
задаём вопрос: «Как изменить параметры политики $\theta$, чтобы получать больше вознаграждения?»

---

## 3. Формализация цели: J(θ)

Определим **целевую функцию** (objective) как ожидаемое суммарное вознаграждение при следовании политике $\pi_\theta$:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [G_0] = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_{t+1} \right]
$$

где:

* $\tau = (s_0, a_0, r_1, s_1, a_1, \dots)$ — траектория,
* $G_0$ — возврат (return) от начала эпизода.

**Цель обучения:**

$$
\theta^* = \arg\max_\theta J(\theta)
$$

---

## 4. Градиент целевой функции: Policy Gradient Theorem

Чтобы максимизировать $J(\theta)$, нужно вычислить его градиент $\nabla_\theta J(\theta)$.

### Policy Gradient Theorem (PG Theorem)

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]
$$

**Интуитивная интерпретация:**

> «Увеличивай вероятность действий, которые привели к высокому вознаграждению, и уменьшай вероятность действий с низким вознаграждением.»

---

## 5. Разбор компонентов градиента

| Компонент | Что означает | Интуиция |
|-----------|--------------|----------|
| $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ | Градиент логарифма вероятности действия | Направление изменения параметров для увеличения $P(a_t)$ |
| $G_t$ | Суммарное вознаграждение с момента $t$ | «Вес» действия (насколько оно было выгодным) |
| $\mathbb{E}_{\tau \sim \pi_\theta}$ | Усреднение по траекториям | Нужно собирать опыт, взаимодействуя со средой |

---

## 6. Алгоритм REINFORCE (Monte Carlo Policy Gradient)

**REINFORCE** — простейший алгоритм на основе Policy Gradient, использующий Monte Carlo для оценки возврата.

### Псевдокод

1. Инициализировать параметры политики $\theta$ случайно.
2. Для каждого эпизода $k = 1, 2, \dots$:
   1. Сгенерировать траекторию $\tau = (s_0, a_0, r_1, s_1, a_1, \dots, s_T)$, следуя $\pi_\theta$.
   2. Для каждого временного шага $t = 0, 1, \dots, T-1$:
      * Вычислить возврат:
        $$
        G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}
        $$
      * Обновить параметры:
        $$
        \theta \leftarrow \theta + \alpha \cdot \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t
        $$

---

## 7. Интуитивный пример: CartPole

Рассмотрим среду `CartPole-v1`, где агент балансирует палку:

* **Состояние $s$:** позиция тележки, угол палки, скорости.
* **Действия:** толкнуть влево (0) или вправо (1).
* **Награда:** +1 за каждый шаг, пока палка не упала.

### Как работает REINFORCE

1. Агент проходит эпизод (например, 50 шагов до падения).
2. Возврат $G_0 = 50$ (палка продержалась 50 шагов).
3. Для каждого действия в эпизоде:
   * Если действие было «вправо» и $G_t$ высокий → увеличиваем $P(\text{вправо}|s)$.
   * Если действие было «влево» и $G_t$ низкий → уменьшаем $P(\text{влево}|s)$.

После многих эпизодов политика начинает предпочитать действия, которые приводят к более долгим балансировкам.

---

## 8. Проблема высокой дисперсии

**Основная сложность REINFORCE:** высокая дисперсия градиента.

Возврат $G_t$ может сильно варьироваться между эпизодами даже при одинаковой политике:

* Эпизод 1: $G_0 = 200$ (удача)
* Эпизод 2: $G_0 = 10$ (неудача)

Это приводит к **нестабильному обучению** — градиенты «прыгают», сходимость медленная.

---

## 9. Решение: Baseline (базовая линия)

Чтобы снизить дисперсию, вычитаем из возврата **baseline** $b(s_t)$ — среднее значение, которое не зависит от действия:

$$
\nabla_\theta J(\theta) \approx \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \big(G_t - b(s_t)\big) \right]
$$

### Популярные варианты baseline

| Тип baseline | Формула | Интуиция |
|--------------|---------|----------|
| **Константа** | $b = \bar{G}$ (среднее по эпизодам) | Простейший вариант |
| **State-value function** | $b(s_t) = V_\pi(s_t)$ | Оценка ожидаемого возврата из $s_t$ |
| **Running average** | $b = \alpha \cdot b + (1-\alpha) \cdot G_t$ | Скользящее среднее |

**Важно:** вычитание baseline **не вносит смещения** в градиент, но уменьшает дисперсию.

---

## 10. Advantage Function

Когда в качестве baseline используется $V(s_t)$, разность $G_t - V(s_t)$ называется **функцией преимущества** (advantage):

$$
A_t = G_t - V(s_t)
$$

Интуиция:

> «Насколько действие $a_t$ было **лучше среднего** для состояния $s_t$?»

* $A_t > 0$ → действие лучше ожидаемого → увеличиваем его вероятность.
* $A_t < 0$ → действие хуже ожидаемого → уменьшаем его вероятность.

---

## 11. REINFORCE с baseline (с V-функцией)

### Алгоритм

1. Инициализировать:
   * Политику $\pi_\theta$ (нейросеть).
   * Value-функцию $V_\phi$ (отдельная нейросеть или общий энкодер).

2. Для каждого эпизода:
   1. Собрать траекторию $\tau$, следуя $\pi_\theta$.
   2. Для каждого $t$:
      * Вычислить $G_t$ (возврат).
      * Вычислить advantage:
        $$
        A_t = G_t - V_\phi(s_t)
        $$
      * Обновить политику:
        $$
        \theta \leftarrow \theta + \alpha_\theta \cdot \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A_t
        $$
      * Обновить V-функцию (минимизация MSE):
        $$
        \phi \leftarrow \phi - \alpha_\phi \cdot \nabla_\phi \big(V_\phi(s_t) - G_t\big)^2
        $$

---

## 12. Архитектура нейросети для политики

### Дискретные действия (например, CartPole)

Выход сети — вектор логитов (или вероятностей) для каждого действия:

$$
\pi_\theta(a|s) = \text{Softmax}\big(\text{NN}_\theta(s)\big)
$$

Пример:

```python
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return torch.softmax(logits, dim=-1)
```

---

### Непрерывные действия (например, LunarLander continuous)

Выход сети — параметры распределения (например, среднее $\mu$ и дисперсия $\sigma$ для нормального распределения):

$$
a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))
$$

```python
class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
```

---

## 13. Сравнение REINFORCE и Q-Learning

| Характеристика | Q-Learning (DQN) | REINFORCE |
|----------------|------------------|-----------|
| Тип метода | Value-based | Policy-based |
| Что учится | Q-функция $Q(s,a)$ | Политика $\pi_\theta(a\|s)$ |
| Пространство действий | Дискретное | Дискретное и непрерывное |
| Требует replay buffer | Да | Нет (on-policy) |
| Дисперсия | Низкая | Высокая (нужен baseline) |
| Сходимость | Быстрее, но нестабильна | Медленнее, но устойчивее |
| Стохастические политики | Нет | Да |

---

## 14. Практический пример: обучение на CartPole

```python
import torch
import torch.optim as optim
import gymnasium as gym

# Инициализация
env = gym.make('CartPole-v1')
policy = PolicyNetwork(state_dim=4, action_dim=2)
optimizer = optim.Adam(policy.parameters(), lr=0.01)
gamma = 0.99

for episode in range(1000):
    states, actions, rewards = [], [], []
    state, _ = env.reset()
    done = False
    
    # Сбор траектории
    while not done:
        # Добавляем batch dimension: [state_dim] -> [1, state_dim]
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = policy(state_tensor).squeeze(0)  # [1, action_dim] -> [action_dim]
        
        # Используем torch.distributions для стабильности
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample().item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
        done = terminated or truncated
    
    # Вычисление возвратов
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Нормализация
    
    # Обновление политики
    policy_loss = []
    for state, action, G_t in zip(states, actions, returns):
        # Правильная обработка размерностей
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = policy(state_tensor).squeeze(0)
        
        # Используем Categorical.log_prob для численной стабильности
        dist = torch.distributions.Categorical(probs=probs)
        log_prob = dist.log_prob(torch.tensor(action))
        
        policy_loss.append(-log_prob * G_t)
    
    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()
```

---

## 15. Энтропийная регуляризация

Чтобы политика не становилась **слишком детерминированной** (что мешает exploration), добавляют **энтропийный бонус** к целевой функции:

$$
J(\theta) = \mathbb{E}_\tau \left[ \sum_t \log \pi_\theta(a_t|s_t) \cdot G_t + \beta \cdot H(\pi_\theta(\cdot|s_t)) \right]
$$

где энтропия политики:

$$
H(\pi) = - \sum_a \pi(a|s) \log \pi(a|s)
$$

**Интуиция:** высокая энтропия = равномерное распределение вероятностей = больше исследования.

---

## 16. Ключевые формулы (шпаргалка)

**Policy Gradient Theorem:**

$$
\boxed{ \nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right] }
$$

**REINFORCE с baseline:**

$$
\boxed{ \nabla_\theta J(\theta) \approx \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \big(G_t - b(s_t)\big) }
$$

**Advantage function:**

$$
\boxed{ A_t = G_t - V(s_t) }
$$

**Обновление параметров (gradient ascent):**

$$
\boxed{ \theta \leftarrow \theta + \alpha \cdot \nabla_\theta J(\theta) }
$$

---

## 17. Преимущества и недостатки Policy Gradients

### Преимущества

* **Непрерывные действия** — естественно работает с непрерывными пространствами.
* **Стохастические политики** — может учить вероятностные стратегии.
* **Устойчивость** — плавное изменение политики (нет резких скачков как в value-based методах).
* **Гарантия сходимости** — при правильных условиях сходится к локальному оптимуму.

### Недостатки

* **Высокая дисперсия** — требуется много траекторий для надёжной оценки градиента.
* **Sample inefficiency** — on-policy метод, нужно собирать новые данные после каждого обновления.
* **Медленная сходимость** — по сравнению с DQN может требовать больше шагов обучения.
* **Чувствительность к гиперпараметрам** — $\alpha$, $\gamma$, архитектура сети критичны.

---

## Резюме

| Понятие | Формула/описание |
|---------|------------------|
| **Целевая функция** | $J(\theta) = \mathbb{E}_\tau[G_0]$ |
| **Policy Gradient** | $\nabla_\theta J(\theta) = \mathbb{E}[\nabla \log \pi \cdot G_t]$ |
| **REINFORCE** | Monte Carlo PG с полным возвратом |
| **Baseline** | $G_t - b(s_t)$ для снижения дисперсии |
| **Advantage** | $A_t = G_t - V(s_t)$ |
| **Применение** | Дискретные и непрерывные действия |

---

**Основано на:**

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2020)
* Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015)
* Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* (1992)
* Hugging Face Deep RL Course, Unit 4
* Andrea Lonza, *Алгоритмы обучения с подкреплением на Python* (2020)

