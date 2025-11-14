# A2C (Advantage Actor-Critic) на Pendulum-v1

## 📘 Описание

Реализация алгоритма **A2C** (Advantage Actor-Critic) для задачи балансировки перевёрнутого маятника с **непрерывным пространством действий**. Демонстрирует мощь Actor-Critic методов для continuous control.

**Среда:** Pendulum-v1  
**Тип действий:** Непрерывные (torque ∈ [-2, 2])  
**Состояние:** Непрерывное (3-мерное: cos(θ), sin(θ), θ̇)  
**Цель:** Поднять маятник в вертикальное положение и удерживать

---

## 🎯 Особенности реализации

### Алгоритм A2C
- ✅ **Actor-Critic architecture** с общим backbone
- ✅ **Gaussian policy** для непрерывных действий
- ✅ **TD-error (advantage)** для снижения дисперсии
- ✅ **Раздельные оптимизаторы** для Actor и Critic (best practice)
- ✅ **Entropy regularization** для exploration
- ✅ **Online updates** после каждого шага
- ✅ **Gradient clipping** для стабильности

### Архитектура

```
                      State (3)
                          ↓
        ┌─────────────────────────────────┐
        │   Shared Backbone                │
        │   FC(256) → ReLU → FC(256) → ReLU│
        └──────────────┬──────────────────┘
                       ↓
           ┌───────────┴───────────┐
           ↓                       ↓
       Actor Head              Critic Head
       ↓                       ↓
   Mean(1), LogStd(1)      Value(1)
       ↓
   N(μ, σ) → Action
```

### Ключевые формулы

**TD-error (Advantage):**
```
δ_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)
```

**Actor update (Policy Gradient):**
```
∇_θ J(θ) = E [ ∇_θ log π_θ(a_t|s_t) · δ_t + β · H(π_θ) ]
```

**Critic update (TD Learning):**
```
L(φ) = (δ_t)²
```

**Gaussian Policy:**
```
π_θ(a|s) = N(μ_θ(s), σ_θ(s))
log π_θ(a|s) = -½[(a - μ)/σ]² - log σ - ½log(2π)
```

---

## 🚀 Быстрый старт

### Установка зависимостей

```bash
# Из корня репозитория
pip install -r requirements.txt
```

### Базовый запуск

```bash
cd code/actor_critic
python pendulum_a2c.py
```

### Запуск с параметрами

```bash
# Обучение на 1500 эпизодов с записью видео
python pendulum_a2c.py --episodes 1500 --record-video

# С разными learning rates для Actor и Critic
python pendulum_a2c.py --lr-actor 5e-4 --lr-critic 1e-3

# С изменённой энтропией
python pendulum_a2c.py --entropy 0.01

# Другой random seed
python pendulum_a2c.py --seed 123
```

### Все параметры

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--episodes` | 1000 | Число эпизодов обучения |
| `--lr-actor` | 3e-4 | Learning rate для Actor |
| `--lr-critic` | 1e-3 | Learning rate для Critic |
| `--entropy` | 0.001 | Коэффициент энтропийной регуляризации |
| `--seed` | 42 | Random seed для воспроизводимости |
| `--record-video` | False | Записать видео evaluation эпизодов |

---

## 📊 Ожидаемые результаты

### Сходимость

**Типичная кривая обучения:**
```
Episode 0-200:    Reward ~ -1400 to -800  (случайные действия)
Episode 200-400:  Reward ~ -800 to -400   (учится поднимать)
Episode 400-600:  Reward ~ -400 to -200   (удерживает вертикаль)
Episode 600-1000: Reward ~ -200 to -150   (оптимальный контроль)
```

**Критерий решения:** Средняя награда >= -200 за 100 последовательных эпизодов

**Примечание:** В Pendulum-v1 награды отрицательные (штрафы за отклонение), цель — максимизировать (приблизиться к 0).

### Пример вывода

```
============================================================
A2C on Pendulum-v1 (Continuous Control)
============================================================
Episodes: 1000
Actor LR: 0.0003
Critic LR: 0.001
Entropy coefficient: 0.001
Seed: 42
============================================================
Training A2C: 100%|████████| 1000/1000 [05:23<00:00, reward=-156.3, avg_100=-178.4, actor_loss=0.234, critic_loss=12.456]

Evaluating trained policy...
Evaluation over 100 episodes: -165.23 ± 45.67
✓ Environment SOLVED! (Average reward >= -200)
Model saved to pendulum_a2c.pt
```

---

## 📈 Визуализация

После обучения автоматически генерируется график `pendulum_a2c_training.png` с 4 панелями:

1. **Training Rewards** — награды по эпизодам
2. **Episode Lengths** — длительность (всегда 200 для Pendulum)
3. **Actor Loss** — loss функции политики
4. **Critic Loss** — loss функции ценности

---

## 🎥 Запись видео

```bash
python pendulum_a2c.py --record-video
```

Видео сохраняются в `videos/pendulum/`:
- 5 evaluation эпизодов после обучения
- Показывают поведение обученного агента
- Формат: MP4, FPS: 30

---

## 🔬 Эксперименты

### 1. Сравнение learning rates

```bash
# Высокий Actor LR (может быть нестабильно)
python pendulum_a2c.py --lr-actor 1e-3 --lr-critic 1e-3

# Низкий Actor LR (стабильно, но медленно)
python pendulum_a2c.py --lr-actor 1e-4 --lr-critic 1e-3

# Balanced (рекомендуется)
python pendulum_a2c.py --lr-actor 3e-4 --lr-critic 1e-3
```

**Правило:** Critic обычно учится быстрее Actor'а (lr_critic > lr_actor)

### 2. Влияние энтропии

```bash
# Без энтропии (детерминированная политика)
python pendulum_a2c.py --entropy 0.0

# Низкая энтропия (рекомендуется для continuous control)
python pendulum_a2c.py --entropy 0.001

# Высокая энтропия (больше exploration, но медленнее)
python pendulum_a2c.py --entropy 0.01
```

### 3. Длительность обучения

```bash
# Короткое обучение
python pendulum_a2c.py --episodes 500

# Стандартное
python pendulum_a2c.py --episodes 1000

# Длительное (для стабилизации)
python pendulum_a2c.py --episodes 2000
```

---

## 🧪 Связь с теорией

Этот код реализует концепции из **note_12_actor_critic_a2c.md**:

| Концепция | Реализация в коде |
|-----------|-------------------|
| Actor-Critic | `ActorCriticNetwork` с двумя головами |
| Gaussian Policy | `Normal(mean, std)` для continuous actions |
| TD-error (Advantage) | `td_target - value` |
| Shared Backbone | `self.shared` слои для Actor и Critic |
| Раздельные оптимизаторы | `actor_optimizer`, `critic_optimizer` |
| Entropy regularization | `entropy_coef * entropy` |
| Gradient clipping | `clip_grad_norm_()` |
| Online updates | Обновление после каждого шага в `train_step()` |

---

## 🆚 Сравнение с REINFORCE

| Характеристика | REINFORCE | A2C |
|----------------|-----------|-----|
| Обновления | После эпизода | После каждого шага |
| Baseline | Optional value function | Обязательный Critic |
| Дисперсия | Высокая | Средняя |
| Sample efficiency | Низкая | Средняя |
| Скорость сходимости | Медленная | Быстрее |
| Continuous actions | ✓ | ✓ |
| Стабильность | Низкая | Выше |

**Вывод:** A2C более sample-efficient и стабильный благодаря онлайн обновлениям и TD-learning.

---

## 📚 Дополнительные материалы

**Теория:**
- `/notes/md/note_12_actor_critic_a2c.md` — Actor-Critic методы и A2C
- `/notes/md/note_11_policy_gradients_reinforce.md` — Policy Gradients (для сравнения)
- `/notes/md/note_08_monte_carlo_vs_td.md` — TD-learning (основа Critic'а)

**Код:**
- `ActorCriticNetwork` — архитектура с shared backbone
- `get_action()` — сэмплирование из Gaussian policy
- `train_step()` — онлайн A2C update

**Литература:**
- Mnih et al. (2016): "Asynchronous Methods for Deep RL" (A3C)
- Schulman et al. (2015): "High-Dimensional Continuous Control Using GAE"
- Sutton & Barto (2020): Chapter 13 - Policy Gradient Methods

---

## 🐛 Troubleshooting

### Проблема: Actor/Critic losses растут

**Причина:** Слишком высокие learning rates или нестабильные градиенты

**Решение:**
```bash
# Уменьшить learning rates
python pendulum_a2c.py --lr-actor 1e-4 --lr-critic 5e-4

# Или увеличить gradient clipping (в коде изменить config.gradient_clip)
```

### Проблема: Политика становится детерминированной слишком быстро

**Причина:** Низкая энтропия или log_std падает слишком быстро

**Решение:**
```bash
# Увеличить entropy coefficient
python pendulum_a2c.py --entropy 0.01

# Или изменить log_std_min в коде (например, -10 вместо -20)
```

### Проблема: Не сходится к -200

**Причина:** Недостаточно эпизодов или неоптимальные гиперпараметры

**Решение:**
```bash
# Увеличить число эпизодов
python pendulum_a2c.py --episodes 1500

# Или сбалансировать learning rates
python pendulum_a2c.py --lr-actor 5e-4 --lr-critic 1e-3 --episodes 1200
```

---

## 📊 Benchmark

**Система:** MacBook Pro M2, 16GB RAM  
**Время обучения:** ~5-6 минут (1000 эпизодов)  
**Память:** ~150-200 MB  
**Решено за:** ~600-800 эпизодов (с оптимальными гиперпараметрами)

---

## 🎯 Продвинутые улучшения

### 1. GAE (Generalized Advantage Estimation)

Текущая реализация использует 1-step TD-error. Для улучшения можно реализовать GAE:

```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lambda_=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages
```

### 2. PPO (Proximal Policy Optimization)

Следующий шаг эволюции — добавить clipping для более стабильных обновлений:

```python
ratio = torch.exp(new_log_prob - old_log_prob)
clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
```

### 3. Batch Updates

Вместо обновления после каждого шага, накапливать несколько переходов:

```python
# Собрать N шагов
# Затем обновить сразу всё
```

---

## 🎓 Домашнее задание

1. **Запустите базовое обучение** и достигните reward >= -200
2. **Сравните** разные learning rates (постройте графики Actor/Critic loss)
3. **Экспериментируйте** с entropy coefficient (0.0, 0.001, 0.01)
4. **Реализуйте** GAE вместо 1-step TD-error
5. **Попробуйте** другую continuous среду (MountainCarContinuous-v0)
6. **Сравните** A2C с REINFORCE на той же среде

---

## 🔗 Связанные реализации

- **REINFORCE на LunarLander:** `/code/11_policy_gradient/` — для сравнения
- **Q-Learning на CartPole:** `/code/09_q_learning_bellman/` — value-based подход
- **MC vs TD на FrozenLake:** `/code/08_mc_vs_td/` — основы TD-learning

---

**Автор:** Denis Samatov, TPU / 2025  
**Связь с курсом:** Семинар 12 — Actor-Critic и A2C  
**Next steps:** PPO, SAC, DDPG, TD3

