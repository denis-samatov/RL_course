# Теоретический конспект №14

## Тема: Proximal Policy Optimization (PPO) и Trust Region Policy Optimization (TRPO)

> **Связано с:** [note_11_policy_gradients_reinforce.md](note_11_policy_gradients_reinforce.md) — Policy Gradient, REINFORCE · [note_12_actor_critic_a2c.md](note_12_actor_critic_a2c.md) — Actor-Critic, A2C

---

## 1. Мотивация: проблемы Policy Gradient

Вспомним **REINFORCE** и **A2C** ([note_11_policy_gradients_reinforce.md](note_11_policy_gradients_reinforce.md), [note_12_actor_critic_a2c.md](note_12_actor_critic_a2c.md)):

- ✅ Могут обучаться на непрерывных действиях
- ✅ Оптимизируют политику напрямую
- ❌ **Высокая вариативность градиентов** → медленное обучение
- ❌ **Чувствительность к learning rate** → легко разрушить политику
- ❌ **Sample inefficiency** → требуют много траекторий

### Главная проблема: катастрофическое забывание

При больших обновлениях политики может произойти **policy collapse**:

```
Episode 100: Reward = 200 (хорошая политика)
   ↓ [Большой градиентный шаг]
Episode 101: Reward = -50 (политика разрушена)
   ↓ [Невозможно восстановиться]
Episode 200: Reward = -50 (застряли в плохом локальном минимуме)
```

**Интуиция:**

> «Слишком большое обновление политики может вывести агента за пределы "зоны доверия", откуда невозможно восстановиться.»

---

## 2. Trust Region методы: основная идея

**Решение:** Ограничить **размер изменения политики** на каждом шаге.

### KL-дивергенция как метрика расстояния

Используем **KL-дивергенцию** для измерения "расстояния" между старой и новой политикой:

$$
D_{KL}(\pi_{\text{old}} \| \pi_{\text{new}}) = \mathbb{E}_{s \sim \rho_{\pi_{\text{old}}}} \left[ D_{KL}(\pi_{\text{old}}(\cdot|s) \| \pi_{\text{new}}(\cdot|s)) \right]
$$

Для дискретных действий:

$$
D_{KL}(\pi_{\text{old}} \| \pi_{\text{new}}) = \sum_a \pi_{\text{old}}(a|s) \log \frac{\pi_{\text{old}}(a|s)}{\pi_{\text{new}}(a|s)}
$$

**Свойства KL-дивергенции:**

- $D_{KL} \geq 0$, равенство только при $\pi_{\text{old}} = \pi_{\text{new}}$
- Несимметрична: $D_{KL}(p\|q) \neq D_{KL}(q\|p)$
- Мера "информационного расстояния"

---

### Trust Region Constraint

Идея: максимизировать целевую функцию с **ограничением на KL-дивергенцию**:

$$
\max_\theta \mathbb{E}_{\pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s,a) \right]
$$

$$
\text{subject to } \mathbb{E}_{s \sim \rho_{\pi_{\theta_{\text{old}}}}} \left[ D_{KL}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s)) \right] \leq \delta
$$

где:
- $\delta$ — максимально допустимая KL-дивергенция (типично 0.01-0.05)
- $A^{\pi}(s,a)$ — advantage function

**Интуиция:**

> «Улучшай политику, но не уходи слишком далеко от текущей версии.»

---

## 3. TRPO: Trust Region Policy Optimization

**TRPO** (Schulman et al., 2015) — первая успешная реализация trust region идеи.

### Суррогатная целевая функция

Вместо прямой максимизации $J(\theta)$, используем **Conservative Policy Iteration (CPI)** objective:

$$
L^{CPI}(\theta) = \mathbb{E}_{t} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t \right] = \mathbb{E}_t \left[ r_t(\theta) \hat{A}_t \right]
$$

где **probability ratio**:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

**Почему это работает?**

- При $\theta = \theta_{\text{old}}$: $r_t = 1$, градиенты совпадают с policy gradient
- Позволяет **переиспользовать старые траектории** (off-policy)

---

### Constraint Optimization

TRPO решает задачу:

$$
\max_\theta L^{CPI}(\theta) \quad \text{s.t.} \quad \bar{D}_{KL}(\theta_{\text{old}}, \theta) \leq \delta
$$

где $\bar{D}_{KL}$ — средняя KL-дивергенция по состояниям.

**Решение через сопряжённые градиенты:**

1. Аппроксимация KL-constraint локально (второй порядок):
   $$
   \bar{D}_{KL} \approx \frac{1}{2} (\theta - \theta_{\text{old}})^T F (\theta - \theta_{\text{old}})
   $$
   где $F$ — Fisher Information Matrix

2. Решение через **conjugate gradient** для $F^{-1} g$

3. **Line search** для гарантии улучшения и соблюдения constraint

---

### Проблемы TRPO

| Проблема | Описание |
|----------|----------|
| **Сложность реализации** | Conjugate gradient, line search, Hessian-vector products |
| **Вычислительная стоимость** | Многократные вычисления KL и его производных |
| **Чувствительность к гиперпараметрам** | Backtracking line search коэффициенты |
| **Трудности с RNN/Transformer** | Fisher matrix становится огромной |

**Вывод:**

> TRPO теоретически элегантен, но **практически сложен**. Нужна более простая альтернатива!

---

## 4. PPO: Proximal Policy Optimization

**PPO** (Schulman et al., 2017) — упрощённая версия TRPO, ставшая **индустриальным стандартом**.

### Две версии PPO

1. **PPO-Clip** (чаще используется)
2. **PPO-Penalty** (адаптивный KL-штраф)

---

### PPO-Clip

**Идея:** Вместо constraint, **обрезаем (clip)** суррогатную функцию.

**Целевая функция:**

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]
$$

где:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ — probability ratio
- $\epsilon$ — clip range (типично 0.1 или 0.2)
- $\text{clip}(x, a, b) = \max(a, \min(b, x))$

**Интуиция:**

$$
\text{clip}(r_t, 1-\epsilon, 1+\epsilon) \in [0.8, 1.2] \quad (\text{при } \epsilon=0.2)
$$

- Если $r_t$ близко к 1 → обычный policy gradient
- Если $r_t$ далеко от 1 → обрезаем, чтобы не уйти слишком далеко

---

### Разбор PPO-Clip по случаям

**Случай 1:** $\hat{A}_t > 0$ (хорошее действие, хотим увеличить вероятность)

$$
L^{CLIP} = \min(r_t \hat{A}_t, \, (1+\epsilon) \hat{A}_t)
$$

- Если $r_t < 1+\epsilon$: используем $r_t \hat{A}_t$ (продолжаем увеличивать)
- Если $r_t > 1+\epsilon$: используем $(1+\epsilon) \hat{A}_t$ (останавливаем рост)

**Случай 2:** $\hat{A}_t < 0$ (плохое действие, хотим уменьшить вероятность)

$$
L^{CLIP} = \max(r_t \hat{A}_t, \, (1-\epsilon) \hat{A}_t)
$$

- Если $r_t > 1-\epsilon$: используем $r_t \hat{A}_t$ (продолжаем уменьшать)
- Если $r_t < 1-\epsilon$: используем $(1-\epsilon) \hat{A}_t$ (останавливаем падение)

**Визуально:**

```
Для A > 0:
  L(r) = min(r*A, (1+ε)*A)
       |     /-------- (clipped)
       |    /
       |   /
       |  /
  -----+--------
     1-ε  1  1+ε   r

Для A < 0:
       |
  -----+--------
       |\
       | \
       |  \  (clipped)
       |   \----
              r
```

---

### PPO-Penalty

Альтернативный подход: **адаптивный KL-штраф** вместо clipping.

**Целевая функция:**

$$
L^{KLPEN}(\theta) = \mathbb{E}_t \left[ r_t(\theta)\hat{A}_t - \beta \cdot D_{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \right]
$$

где $\beta$ — коэффициент штрафа, **адаптивно изменяется**:

```python
if d_kl < target_kl / 1.5:
    beta = beta / 2  # Уменьшаем штраф
elif d_kl > target_kl * 1.5:
    beta = beta * 2  # Увеличиваем штраф
```

**Сравнение:**

| Метод | Преимущества | Недостатки |
|-------|--------------|------------|
| **PPO-Clip** | Простой, без гиперпараметра $\beta$ | Менее интерпретируем |
| **PPO-Penalty** | Явно контролирует KL | Нужна адаптация $\beta$ |

**На практике:** PPO-Clip используется чаще из-за простоты.

---

## 5. Generalized Advantage Estimation (GAE)

**Проблема:** Advantage $A(s,a)$ имеет компромисс bias-variance:

- **TD-error** ($\hat{A}_t = r_t + \gamma V(s_{t+1}) - V(s_t)$): low variance, high bias
- **Monte-Carlo** ($\hat{A}_t = G_t - V(s_t)$): unbiased, high variance

**GAE** (Schulman et al., 2016) — экспоненциально взвешенное среднее TD-ошибок.

### Формула GAE

$$
\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
$$

где $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ — TD-error.

**Развёрнуто:**

$$
\hat{A}_t^{GAE} = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots
$$

**Параметр $\lambda \in [0,1]$:**

- $\lambda = 0$: $\hat{A}_t = \delta_t$ (TD, high bias, low variance)
- $\lambda = 1$: $\hat{A}_t = G_t - V(s_t)$ (MC, low bias, high variance)
- $\lambda \in (0,1)$: **компромисс** (обычно 0.95-0.99)

**Практическая реализация:**

```python
def compute_gae(rewards, values, dones, gamma=0.99, lambda_gae=0.95):
    """
    Вычисляет GAE advantages.
    
    Args:
        rewards: List[float] — награды длины T
        values: List[float] — V(s) длины T+1 (включая bootstrap V(s_T+1))
        dones: List[bool] — флаги терминации длины T
        gamma: Discount factor
        lambda_gae: GAE lambda параметр
        
    Returns:
        advantages: List[float] длины T
    """
    advantages = []
    gae = 0.0
    
    # Итерируемся в обратном порядке
    for t in reversed(range(len(rewards))):
        # TD-error: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t+1] * (1 - int(dones[t])) - values[t]
        
        # GAE: A_t = δ_t + (γλ) δ_{t+1} + (γλ)^2 δ_{t+2} + ...
        gae = delta + gamma * lambda_gae * (1 - int(dones[t])) * gae
        
        advantages.insert(0, gae)
    
    return advantages

# Пример использования:
# Предполагаем, что values содержит V(s_0), ..., V(s_T), V(s_{T+1})
advantages = compute_gae(rewards, values, dones, gamma=0.99, lambda_gae=0.95)
```

---

## 6. Архитектура PPO Agent

### Сети Actor и Critic

**Два варианта:**

1. **Shared backbone** (экономичнее):
   ```
   Input (state) → [Shared Layers] → Actor Head
                                   → Critic Head
   ```

2. **Separate networks** (гибче):
   ```
   Input (state) → [Actor Layers] → π(a|s)
   Input (state) → [Critic Layers] → V(s)
   ```

**Пример PyTorch:**

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Actor head
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head
        self.critic = nn.Linear(hidden, 1)
    
    def forward(self, state):
        features = self.shared(state)
        # Actor output (для непрерывных действий)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd)
        # Critic output
        value = self.critic(features)
        return action_mean, action_std, value
```

---

### Дискретные vs Непрерывные действия

**Дискретные действия:**

```python
# Actor выдаёт логиты для каждого действия
logits = self.actor(state)
dist = Categorical(logits=logits)
action = dist.sample()
log_prob = dist.log_prob(action)
```

**Непрерывные действия (Gaussian policy):**

```python
# Actor выдаёт mean и std
mean, std = self.actor(state)
dist = Normal(mean, std)
action = dist.sample()
log_prob = dist.log_prob(action).sum(dim=-1)
```

---

## 7. PPO Алгоритм (полный)

**Псевдокод:**

```
for iteration in range(N):
    # 1. Сбор траекторий с текущей политикой π_old
    trajectories = collect_rollouts(env, π_old, n_steps)
    
    # 2. Вычисление advantages через GAE
    advantages = compute_gae(trajectories, V, γ, λ)
    
    # 3. Нормализация advantages (опционально)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # 4. Несколько эпох оптимизации на собранных данных
    for epoch in range(K):  # K = 3-10
        for batch in mini_batches(trajectories):
            # Вычисляем текущие log_probs и values
            log_probs_new, values_new = π_θ(batch)
            
            # Probability ratio
            ratio = exp(log_probs_new - batch.log_probs_old)
            
            # PPO-Clip loss
            surr1 = ratio * batch.advantages
            surr2 = clip(ratio, 1-ε, 1+ε) * batch.advantages
            policy_loss = -min(surr1, surr2).mean()
            
            # Value loss (MSE)
            value_loss = (values_new - batch.returns)^2.mean()
            
            # Entropy bonus (для exploration)
            entropy = π_θ.entropy().mean()
            
            # Полная loss
            loss = policy_loss + c1 * value_loss - c2 * entropy
            
            # Обновление параметров
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters, max_norm=0.5)
            optimizer.step()
```

**Ключевые гиперпараметры:**

| Параметр | Типичное значение | Описание |
|----------|-------------------|----------|
| `ε` (clip_range) | 0.1 - 0.2 | Диапазон clipping для ratio |
| `K` (epochs) | 3 - 10 | Эпох оптимизации на одном батче |
| `γ` (gamma) | 0.99 | Discount factor |
| `λ` (lambda_gae) | 0.95 | GAE lambda |
| `c1` (value_coef) | 0.5 - 1.0 | Вес value loss |
| `c2` (entropy_coef) | 0.01 | Вес entropy bonus |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `lr` | 3e-4 | Learning rate |
| `batch_size` | 64 - 256 | Mini-batch size |
| `n_steps` | 2048 | Шагов для сбора траекторий |

---

## 8. Практические трюки PPO

### 1. Advantage Normalization

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Зачем:**
- Стабилизирует обучение
- Делает масштаб loss независимым от масштаба rewards

---

### 2. Value Function Clipping

**Проблема:** Value function может измениться слишком сильно.

**Решение:** Clipping для value loss:

```python
v_pred_clipped = v_old + torch.clamp(v_pred - v_old, -ε, ε)
value_loss = torch.max(
    (v_pred - returns) ** 2,
    (v_pred_clipped - returns) ** 2
).mean()
```

---

### 3. Learning Rate Annealing

```python
# Линейное уменьшение LR
lr_new = lr_init * (1 - iteration / max_iterations)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr_new
```

**Почему:** Ранние итерации — большие шаги для exploration, поздние — малые для fine-tuning.

---

### 4. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

**Зачем:** Предотвращает gradient explosion.

---

### 5. Multiple Environments (Vectorized Env)

```python
from gymnasium.vector import SyncVectorEnv

envs = SyncVectorEnv([make_env() for _ in range(8)])
```

**Преимущества:**
- Параллельный сбор траекторий
- Уменьшение корреляции между сэмплами
- Ускорение обучения

---

### 6. State Normalization (Running Mean/Std)

```python
# Обновление running statistics
running_mean = alpha * obs + (1 - alpha) * running_mean
running_std = alpha * abs(obs - running_mean) + (1 - alpha) * running_std

# Нормализация
obs_normalized = (obs - running_mean) / (running_std + 1e-8)
```

**Зачем:** Многие среды имеют разный масштаб наблюдений → нормализация стабилизирует обучение.

---

## 9. Сравнение методов

| Метод | Год | Сложность | Стабильность | Sample Efficiency | Популярность |
|-------|-----|-----------|--------------|-------------------|--------------|
| **REINFORCE** | 1992 | Низкая | Низкая | Низкая | 🔵 Учебная |
| **A2C** | 2016 | Средняя | Средняя | Средняя | 🟢 Практическая |
| **TRPO** | 2015 | Высокая | Высокая | Средняя | 🟡 Историческая |
| **PPO** | 2017 | Средняя | Высокая | Высокая | 🟢🟢 State-of-the-art |

**Когда использовать PPO:**

✅ Непрерывные действия (робототехника, управление)  
✅ Дискретные действия с большим пространством  
✅ Нужна стабильность и воспроизводимость  
✅ Ограниченные вычислительные ресурсы  

**Когда НЕ использовать PPO:**

❌ Очень малый датасет (лучше offline RL)  
❌ Нужна максимальная sample efficiency (лучше SAC, TD3)  
❌ Простая дискретная среда (достаточно DQN)  

---

## 10. PPO в индустрии

### OpenAI Five (Dota 2)

- Обучено на PPO
- 256 GPUs, 128,000 CPU cores
- 10 месяцев игрового времени
- Победило профессиональных игроков

### ChatGPT RLHF

- PPO для fine-tuning на человеческих предпочтениях
- Ключевой компонент InstructGPT → ChatGPT
- KL-штраф к SFT-модели для стабильности

### DeepMind's Alphastar (StarCraft II)

- Использует вариант PPO
- Масштабирование до десятков тысяч игр параллельно

---

## 11. Резюме

| Концепция | Описание |
|-----------|----------|
| **Trust Region** | Ограничение изменения политики через KL-дивергенцию |
| **TRPO** | Constraint optimization с conjugate gradient |
| **PPO-Clip** | Clipping probability ratio для ограничения обновлений |
| **PPO-Penalty** | Адаптивный KL-штраф в целевой функции |
| **GAE** | Компромисс bias-variance для advantage estimation |
| **Multiple Epochs** | Переиспользование траекторий для sample efficiency |

**Ключевые выводы:**

1. **PPO решает проблему нестабильности Policy Gradient** через clipping или KL-penalty
2. **GAE обеспечивает оптимальный компромисс** bias-variance для advantage
3. **PPO = индустриальный стандарт** для continuous control и RLHF
4. **Простота реализации** + **стабильность** + **sample efficiency** = успех PPO

---

## 12. Связь с предыдущими семинарами

- **[note_11_policy_gradients_reinforce.md](note_11_policy_gradients_reinforce.md):** REINFORCE — baseline для PG методов
- **[note_12_actor_critic_a2c.md](note_12_actor_critic_a2c.md):** A2C — Actor-Critic архитектура, используемая в PPO
- **[note_13_dynamic_programming.md](note_13_dynamic_programming.md):** GPI — концепция итеративного улучшения политики

---

## 13. Дальнейшее изучение

Рекомендуемые источники:

- **Оригинальные статьи:**
  - TRPO: *Trust Region Policy Optimization* (Schulman et al., 2015)
  - PPO: *Proximal Policy Optimization Algorithms* (Schulman et al., 2017)
  - GAE: *High-Dimensional Continuous Control Using GAE* (Schulman et al., 2016)

- **Имплементации:**
  - [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
  - [Stable-Baselines3: PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

- **Видео:**
  - *Lecture on TRPO/PPO* by John Schulman (UC Berkeley)

---

## 14. Практическое задание

В директории `code/14_ppo_trpo/` реализован полный PPO-агент для `BipedalWalker-v3`.

**Эксперименты:**
- Влияние clip-range $\epsilon$ (0.1, 0.2, 0.3)
- Влияние GAE $\lambda$ (0.9, 0.95, 0.99)
- Shared vs Separate networks для Actor/Critic
- Сравнение с A2C на той же среде

---

**Далее:** [note_15_rlhf_pipeline.md](note_15_rlhf_pipeline.md) — RLHF: Reinforcement Learning from Human Feedback

