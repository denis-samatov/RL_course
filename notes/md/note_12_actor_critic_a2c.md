# Теоретический конспект №12

## Тема: Actor-Critic методы и A2C

> **Связано с:** [note_11_policy_gradients_reinforce.md](note_11_policy_gradients_reinforce.md) — Policy Gradients и REINFORCE · [note_09_q_learning.md](note_09_q_learning.md) — Q-Learning · [note_10_deep_q_network.md](note_10_deep_q_network.md) — Deep Q-Network

---

## 1. Мотивация: объединение Policy-Based и Value-Based подходов

В предыдущих конспектах мы изучили два основных подхода:

| Подход | Что учит | Преимущества | Недостатки |
|--------|----------|--------------|------------|
| **Value-Based** (DQN) | Q-функцию $Q(s,a)$ | Низкая дисперсия, sample-efficient | Только дискретные действия, детерминированная политика |
| **Policy-Based** (REINFORCE) | Политику $\pi_\theta(a\|s)$ | Непрерывные действия, стохастическая политика | Высокая дисперсия, медленная сходимость |

**Идея Actor-Critic:** объединить оба подхода, чтобы получить преимущества каждого и нивелировать недостатки.

---

## 2. Архитектура Actor-Critic

### Два компонента

1. **Actor (актёр)** — политика $\pi_\theta(a|s)$, которая выбирает действия.
2. **Critic (критик)** — value-функция $V_\phi(s)$ или $Q_\phi(s,a)$, которая оценивает качество действий.

### Как они взаимодействуют

```
        ┌─────────────────────────┐
        │      Среда (Env)        │
        └───────────┬─────────────┘
                    │ state, reward
                    ↓
        ┌─────────────────────────┐
        │   Actor (π_θ)           │ ← Выбирает действие
        │   "Что делать?"         │
        └───────────┬─────────────┘
                    │ action
                    ↓
        ┌─────────────────────────┐
        │   Critic (V_φ или Q_φ)  │ ← Оценивает действие
        │   "Насколько хорошо?"   │
        └─────────────────────────┘
                    │
                    ↓ TD-error / Advantage
        ┌─────────────────────────┐
        │   Обновление параметров │
        │   θ ← θ + α∇J           │
        │   φ ← φ - α∇L           │
        └─────────────────────────┘
```

**Интуиция:**

> Actor учится принимать решения, а Critic учится их оценивать. Critic помогает Actor'у понять, какие действия хороши, а какие нет.

---

## 3. Математическая формализация

### Обновление Actor'а (политики)

Используем Policy Gradient, но вместо полного возврата $G_t$ используем **оценку Critic'а**:

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t \right]
$$

где $\delta_t$ — **TD-ошибка** (temporal difference error):

$$
\delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

### Обновление Critic'а (value-функции)

Минимизируем квадратичную ошибку между предсказанием и целью:

$$
L(\phi) = \mathbb{E} \left[ \big(r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)\big)^2 \right]
$$

Градиент:

$$
\nabla_\phi L(\phi) = - \mathbb{E} \left[ \delta_t \cdot \nabla_\phi V_\phi(s_t) \right]
$$

---

## 4. Почему Actor-Critic лучше REINFORCE?

### Сравнение градиентов

| Метод | Градиент | Оценка |
|-------|----------|--------|
| **REINFORCE** | $\nabla \log \pi \cdot G_t$ | Полный возврат $G_t$ (высокая дисперсия) |
| **Actor-Critic** | $\nabla \log \pi \cdot \delta_t$ | TD-ошибка $\delta_t$ (низкая дисперсия) |

### Преимущества TD-ошибки

1. **Меньше дисперсия** — $\delta_t$ основана на одном шаге, а не на всём эпизоде.
2. **Онлайн обучение** — можно обновлять параметры после каждого шага, не дожидаясь конца эпизода.
3. **Бутстрэппинг** — используем оценку $V_\phi(s_{t+1})$ вместо полного возврата.

---

## 5. Advantage Actor-Critic (A2C): улучшенная версия

### Проблема TD-ошибки

TD-ошибка $\delta_t = r + \gamma V(s') - V(s)$ может быть смещённой, если $V_\phi$ ещё плохо обучена.

### Решение: Advantage Function

Вместо TD-ошибки используем **функцию преимущества** (advantage):

$$
A_t = Q(s_t, a_t) - V(s_t)
$$

Интуиция:

> «Насколько действие $a_t$ **лучше среднего** для состояния $s_t$?»

### Оценка Advantage через TD-ошибку

Поскольку $Q(s,a) = r + \gamma V(s')$, можно записать:

$$
A_t \approx \delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

Это **одношаговая оценка advantage** (1-step advantage).

---

## 6. Generalized Advantage Estimation (GAE)

Для дальнейшего снижения дисперсии и смещения используют **GAE** — взвешенную комбинацию n-step advantage'ей.

### N-step advantage

$$
A_t^{(n)} = \sum_{i=0}^{n-1} \gamma^i r_{t+i+1} + \gamma^n V(s_{t+n}) - V(s_t)
$$

### GAE формула

$$
A_t^{\text{GAE}(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

где $\lambda \in [0,1]$ — параметр компромисса:

* $\lambda = 0$ → 1-step TD (низкая дисперсия, высокое смещение).
* $\lambda = 1$ → Monte Carlo (высокая дисперсия, низкое смещение).

**Типичное значение:** $\lambda = 0.95$.

---

## 7. Алгоритм A2C (Advantage Actor-Critic)

### Псевдокод

1. Инициализировать:
   * Actor $\pi_\theta$ (нейросеть для политики).
   * Critic $V_\phi$ (нейросеть для value-функции).

2. Для каждого шага $t = 1, 2, \dots$:
   1. Наблюдать состояние $s_t$.
   2. Выбрать действие $a_t \sim \pi_\theta(\cdot|s_t)$.
   3. Выполнить $a_t$, получить $r_{t+1}, s_{t+1}$.
   4. Вычислить TD-ошибку:
      $$
      \delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
      $$
   5. Обновить Actor:
      $$
      \theta \leftarrow \theta + \alpha_\theta \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t
      $$
   6. Обновить Critic:
      $$
      \phi \leftarrow \phi - \alpha_\phi \cdot \nabla_\phi \big(V_\phi(s_t) - (r_{t+1} + \gamma V_\phi(s_{t+1}))\big)^2
      $$

---

## 8. Архитектура нейросетей в A2C

### Раздельные сети (separate networks)

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return torch.softmax(logits, dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value
```

### Общий энкодер (shared backbone)

Более эффективный вариант — использовать общие слои для извлечения признаков:

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Общий энкодер
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Linear(64, action_dim)
        
        # Critic head
        self.critic = nn.Linear(64, 1)
    
    def forward(self, state):
        features = self.shared(state)
        policy = torch.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return policy, value
```

---

## 9. Практический пример: A2C на CartPole

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Инициализация
env = gym.make('CartPole-v1')
model = ActorCritic(state_dim=4, action_dim=2)

# Для лучшего контроля можно использовать раздельные оптимизаторы:
# optimizer_actor = optim.Adam(model.actor.parameters(), lr=3e-4)
# optimizer_critic = optim.Adam(model.critic.parameters(), lr=1e-3)
# Здесь для простоты используем один:
optimizer = optim.Adam(model.parameters(), lr=0.001)
gamma = 0.99

for episode in range(1000):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Forward pass
        policy, value = model(state_tensor)
        
        # Выбор действия
        action = torch.multinomial(policy, 1).item()
        
        # Шаг в среде
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        episode_reward += reward
        
        # Вычисление TD-ошибки
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            _, next_value = model(next_state_tensor)
        
        td_target = reward + gamma * next_value * (1 - int(done))
        td_error = td_target - value
        
        # Лоссы
        actor_loss = -torch.log(policy[0, action]) * td_error.detach()
        critic_loss = td_error.pow(2)
        
        loss = actor_loss + 0.5 * critic_loss
        
        # Обновление
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {episode_reward}")
```

---

## 10. Энтропийная регуляризация в A2C

Чтобы политика не становилась слишком детерминированной, добавляют **энтропийный бонус**:

$$
L_{\text{total}} = L_{\text{actor}} + c_1 \cdot L_{\text{critic}} - c_2 \cdot H(\pi)
$$

где энтропия политики:

$$
H(\pi_\theta(\cdot|s)) = - \sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)
$$

**Гиперпараметры:**
* $c_1 = 0.5$ (вес critic loss)
* $c_2 = 0.01$ (вес энтропии)

```python
# Добавление энтропии к лоссу
entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=-1)
loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
```

---

## 11. A3C: Асинхронная версия A2C

**A3C (Asynchronous Advantage Actor-Critic)** — расширение A2C с параллельным обучением.

### Ключевые отличия

| Характеристика | A2C | A3C |
|----------------|-----|-----|
| Параллелизм | Один процесс | Множество параллельных процессов (workers) |
| Обновление | Синхронное | Асинхронное |
| Буфер опыта | Не требуется | Не требуется |
| Скорость | Средняя | Высокая (за счёт параллелизма) |

### Схема работы A3C

```
Global Network (θ_global, φ_global)
         │
    ┌────┼────┬────┬────┐
    │    │    │    │    │
Worker 1 Worker 2 ... Worker N
    │    │    │    │    │
   Env1  Env2  ...  EnvN
    │    │    │    │    │
    └────┴────┴────┴────┘
         Асинхронные градиенты → Global Network
```

Каждый worker:
1. Копирует параметры из глобальной сети.
2. Собирает траекторию в своей среде.
3. Вычисляет градиенты.
4. Обновляет глобальную сеть.

---

## 12. Сравнение методов: DQN vs REINFORCE vs A2C

| Характеристика | DQN | REINFORCE | A2C |
|----------------|-----|-----------|-----|
| Тип метода | Value-based | Policy-based | Actor-Critic |
| Пространство действий | Дискретное | Любое | Любое |
| Дисперсия | Низкая | Высокая | Средняя |
| Sample efficiency | Высокая (off-policy) | Низкая (on-policy) | Средняя (on-policy) |
| Стабильность | Нестабильна (overestimation) | Медленная сходимость | Более стабильна |
| Требует replay buffer | Да | Нет | Нет |
| Сходимость | К оптимальной Q-функции | К локальному оптимуму политики | К локальному оптимуму политики |

---

## 13. Когда использовать A2C

### Подходит для:

* **Непрерывные пространства действий** (управление роботами, автопилоты).
* **Задачи с длинными эпизодами** (не нужно ждать конца эпизода для обновления).
* **Стохастические среды** (где детерминированная политика неоптимальна).
* **Задачи, требующие exploration** (энтропийная регуляризация).

### Не подходит для:

* Задачи с очень высокой дисперсией наград (лучше использовать DQN с replay buffer).
* Задачи, требующие максимальной sample efficiency (off-policy методы лучше).

---

## 14. Практические советы по обучению A2C

### Гиперпараметры

| Параметр | Типичное значение | Назначение |
|----------|-------------------|------------|
| Learning rate (Actor) | $3 \times 10^{-4}$ | Скорость обновления политики |
| Learning rate (Critic) | $1 \times 10^{-3}$ | Скорость обновления value-функции |
| Discount factor $\gamma$ | $0.99$ | Дисконтирование будущих наград |
| GAE $\lambda$ | $0.95$ | Компромисс bias-variance |
| Entropy coefficient $c_2$ | $0.01$ | Сила exploration |
| Value loss coefficient $c_1$ | $0.5$ | Вес critic loss в общем лоссе |

### Нормализация

1. **Нормализация состояний (running statistics):**
   ```python
   # Инициализация running statistics
   running_mean = np.zeros(state_dim)
   running_std = np.ones(state_dim)
   alpha = 0.01  # скорость обновления статистик
   
   # Обновление и применение
   running_mean = alpha * state + (1 - alpha) * running_mean
   running_std = alpha * np.abs(state - running_mean) + (1 - alpha) * running_std
   state = (state - running_mean) / (running_std + 1e-8)
   
   # Альтернатива: использовать VecNormalize из stable-baselines3
   from stable_baselines3.common.vec_env import VecNormalize
   ```

2. **Нормализация advantage:**
   ```python
   advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
   ```

3. **Градиентный клиппинг:**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
   ```

### Отладка

* **Мониторинг TD-ошибки** — должна уменьшаться со временем.
* **Entropy** — не должна падать слишком быстро (политика становится детерминированной).
* **Отношение actor/critic loss** — должно быть сбалансировано.
* **Средняя награда** — должна расти.

---

## 15. Расширения и улучшения A2C

### Proximal Policy Optimization (PPO)

Ограничивает изменение политики за одно обновление через clipping:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\big(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) A_t\big) \right]
$$

где $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ — отношение вероятностей.

**Преимущество:** более стабильное обучение, лучшая sample efficiency.

### Soft Actor-Critic (SAC)

Максимизирует не только награду, но и энтропию (maximum entropy RL):

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_t r(s_t, a_t) + \alpha H(\pi(\cdot|s_t)) \right]
$$

**Преимущество:** лучший exploration, работает с непрерывными действиями.

---

## 16. Ключевые формулы (шпаргалка)

**TD-ошибка (advantage):**

$$
\boxed{ \delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) }
$$

**Обновление Actor (политики):**

$$
\boxed{ \theta \leftarrow \theta + \alpha_\theta \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t }
$$

**Обновление Critic (value-функции):**

$$
\boxed{ \phi \leftarrow \phi - \alpha_\phi \cdot \nabla_\phi \big(V_\phi(s_t) - y_t\big)^2 }
$$

где $y_t = r_{t+1} + \gamma V_\phi(s_{t+1})$ — TD-target.

**GAE (Generalized Advantage Estimation):**

$$
\boxed{ A_t^{\text{GAE}(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} }
$$

**Общий loss с энтропией:**

$$
\boxed{ L_{\text{total}} = L_{\text{actor}} + c_1 \cdot L_{\text{critic}} - c_2 \cdot H(\pi) }
$$

---

## 17. Визуализация обучения A2C

```
Episode reward vs Training steps
    │
400 │                              ╱───
    │                          ╱───
300 │                      ╱───
    │                  ╱───
200 │              ╱───
    │          ╱───
100 │      ╱───
    │  ╱───
  0 │───────────────────────────────────
    0    5k   10k   15k   20k   25k   30k
                Training steps
```

**Типичное поведение:**
* **0-5k шагов:** Случайное поведение, низкие награды.
* **5k-15k шагов:** Быстрое улучшение, политика учится основным паттернам.
* **15k-30k шагов:** Стабилизация, fine-tuning политики.

---

## 18. Сравнение с другими алгоритмами

```
                Sample Efficiency
                        │
    DQN ────────────────┼─────────→ (High)
                        │
    A2C ────────┼───────┤
                        │
    REINFORCE ──┼───────┤
                        │
                        ↓
                  Stability
                        │
    PPO ────────────────┼─────────→ (High)
                        │
    A2C ────────┼───────┤
                        │
    DQN ────────┼───────┤
                        │
    REINFORCE ──┼───────┤
```

---

## Резюме

| Понятие | Описание |
|---------|----------|
| **Actor-Critic** | Комбинация policy-based и value-based методов |
| **Actor** | Политика $\pi_\theta(a\|s)$, выбирает действия |
| **Critic** | Value-функция $V_\phi(s)$, оценивает действия |
| **TD-ошибка** | $\delta_t = r + \gamma V(s') - V(s)$ |
| **Advantage** | $A_t = Q(s,a) - V(s) \approx \delta_t$ |
| **A2C** | Advantage Actor-Critic с GAE и энтропией |
| **A3C** | Асинхронная версия с параллельными workers |
| **Применение** | Непрерывные действия, онлайн обучение, стохастические среды |

---

**Основано на:**

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2020)
* Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning* (2016)
* Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015)
* Schulman et al., *Proximal Policy Optimization Algorithms* (2017)
* Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning* (2018)
* Hugging Face Deep RL Course, Unit 5
* Andrea Lonza, *Алгоритмы обучения с подкреплением на Python* (2020)

