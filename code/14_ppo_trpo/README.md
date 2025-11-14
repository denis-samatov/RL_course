# 🚀 Семинар 14: Proximal Policy Optimization (PPO) и TRPO

> **Теория:** [note_14_ppo_trpo.md](../../notes/md/note_14_ppo_trpo.md)  
> **Алгоритм:** PPO-Clip с GAE, Multiple Epochs, Gradient Clipping

---

## 📖 Обзор

Полная реализация **state-of-the-art** PPO алгоритма для непрерывных действий.

### Ключевые особенности:

✅ **PPO-Clip** — clipping probability ratio для стабильности  
✅ **GAE (λ=0.95)** — оптимальный компромисс bias-variance  
✅ **Vectorized Environments** — параллельный сбор траекторий  
✅ **Multiple Epochs** — переиспользование данных для sample efficiency  
✅ **Gradient Clipping** — предотвращение gradient explosion  
✅ **LR Annealing** — линейное уменьшение learning rate  
✅ **Value Clipping** (опционально) — стабилизация critic  

---

## 🗂️ Структура файлов

```
ppo_trpo/
├── ppo_agent.py           # Полная реализация PPO
├── train_ppo.py           # Скрипт обучения с wandb logging
├── evaluate_ppo.py        # Evaluation и запись видео
├── compare_a2c_ppo.py     # Сравнение с A2C
├── README.md              # Эта документация
└── checkpoints/           # (Создаётся при обучении)
    └── ppo_bipedalwalker.pt
```

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install gymnasium[box2d] numpy torch tqdm matplotlib wandb
```

### 2. Обучение PPO

```bash
python ppo_agent.py
```

**Параметры по умолчанию:**
- Environment: `BipedalWalker-v3`
- Total timesteps: 1,000,000
- Parallel envs: 4
- Clip range: 0.2
- GAE lambda: 0.95

**Ожидаемое время:** ~30-60 минут на CPU, ~10-15 минут на GPU

---

### 3. Evaluation

```bash
python evaluate_ppo.py --model checkpoints/ppo_bipedalwalker.pt --episodes 10
```

---

## 📊 PPO Hyperparameters

| Параметр | Значение | Описание |
|----------|----------|----------|
| `clip_range` | 0.2 | Epsilon для clipping ratio |
| `n_steps` | 2048 | Шагов для сбора траекторий |
| `n_epochs` | 10 | Эпох оптимизации на батче |
| `batch_size` | 64 | Mini-batch size |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `learning_rate` | 3e-4 | Начальный LR (с annealing) |
| `value_coef` | 0.5 | Вес value loss |
| `entropy_coef` | 0.01 | Вес entropy bonus |
| `max_grad_norm` | 0.5 | Gradient clipping |

---

## 🎯 BipedalWalker-v3 Environment

**Описание:** Двуногий робот должен научиться ходить по неровной поверхности.

**Наблюдения (24D):**
- Угловые позиции суставов
- Угловые скорости
- Контакты ног с поверхностью
- LIDAR (10 лучей)

**Действия (4D, непрерывные [-1, 1]):**
- Моменты сил на 4 суставах (бедро и колено для каждой ноги)

**Награды:**
- +300 за прохождение дистанции
- -100 за падение
- Штраф за использование моторов

**Решённая задача:** Средняя награда > 300

---

## 📈 Результаты обучения

### Ожидаемая кривая обучения:

```
Timesteps     Mean Reward    Notes
---------     -----------    -----
0 - 200k      -100 to 0      Учится стоять
200k - 500k   0 to 150       Учится делать шаги
500k - 800k   150 to 250     Учится ходить стабильно
800k - 1M     250 to 300+    Fine-tuning походки
```

### Типичные метрики:

| Метрика | Начало | Конец |
|---------|--------|-------|
| Mean Reward | -100 | 300+ |
| Episode Length | 300 | 1600 |
| Policy Loss | 0.5 | 0.05 |
| Value Loss | 50 | 5 |
| Approx KL | 0.02 | 0.005 |
| Clip Fraction | 0.3 | 0.1 |

---

## 🧪 Эксперименты

### Эксперимент 1: Влияние clip_range

```bash
# Протестировать разные epsilon
python train_ppo.py --clip_range 0.1  # Консервативный
python train_ppo.py --clip_range 0.2  # Baseline
python train_ppo.py --clip_range 0.3  # Агрессивный
```

**Ожидаемое:**
- ε=0.1: Медленнее, но стабильнее
- ε=0.2: Оптимальный баланс
- ε=0.3: Быстрее, но может быть нестабильным

---

### Эксперимент 2: Влияние GAE lambda

```bash
python train_ppo.py --gae_lambda 0.90  # Больше bias
python train_ppo.py --gae_lambda 0.95  # Baseline
python train_ppo.py --gae_lambda 0.99  # Больше variance
```

**Теория:**
- λ→0: TD-like (high bias, low variance)
- λ→1: MC-like (low bias, high variance)

---

### Эксперимент 3: Shared vs Separate networks

```python
# Shared backbone (default)
config = PPOConfig(shared_backbone=True)

# Separate networks
config = PPOConfig(shared_backbone=False)
```

**Trade-off:**
- Shared: Меньше параметров, быстрее обучение
- Separate: Больше гибкости, может быть лучше для сложных задач

---

## 🔬 Сравнение с A2C

```bash
python compare_a2c_ppo.py
```

**Ожидаемые различия:**

| Метрика | A2C | PPO |
|---------|-----|-----|
| Финальная награда | 250 | 300+ |
| Sample efficiency | Ниже | Выше |
| Стабильность | Средняя | Высокая |
| Скорость обучения | Быстрее (per update) | Медленнее (multiple epochs) |
| Wall-clock время | ~40 мин | ~30 мин (эффективнее) |

**Вывод:** PPO обычно превосходит A2C за счёт:
- Multiple epochs на данных (sample efficiency)
- Clipping для стабильности
- Меньше чувствительности к гиперпараметрам

---

## 💡 Ключевые компоненты PPO

### 1. Probability Ratio с Clipping

```python
# Ratio π_new / π_old
ratio = exp(log_prob_new - log_prob_old)

# PPO-Clip objective
surr1 = ratio * advantages
surr2 = clip(ratio, 1-ε, 1+ε) * advantages
policy_loss = -min(surr1, surr2).mean()
```

**Интуиция:** Если ratio далеко от 1, clipping останавливает обновление.

---

### 2. GAE (Generalized Advantage Estimation)

```python
gae = 0
for t in reversed(range(T)):
    delta = reward[t] + gamma * V[t+1] - V[t]
    gae = delta + gamma * lambda * gae
    advantages[t] = gae
```

**Интуиция:** Экспоненциально взвешенное среднее TD-ошибок.

---

### 3. Multiple Epochs

```python
for epoch in range(K):  # K = 10
    for batch in rollout_buffer.get_batches():
        # Обновление на том же батче траекторий
        optimize_policy(batch)
```

**Зачем:** Переиспользование данных для sample efficiency.

---

### 4. Value Function Loss

```python
# Опционально: clipping для value
v_clipped = v_old + clip(v_new - v_old, -ε, ε)
value_loss = max((v_new - returns)^2, (v_clipped - returns)^2).mean()
```

---

## 🐛 Troubleshooting

### Проблема 1: Reward не растёт

**Симптомы:** Застрял на ~-100 после 500k шагов

**Возможные причины:**
- Слишком малый clip_range → увеличьте до 0.3
- Слишком малая entropy → увеличьте entropy_coef до 0.02
- Слишком большой learning rate → уменьшите до 1e-4

---

### Проблема 2: Нестабильное обучение

**Симптомы:** Reward скачет вверх-вниз

**Решение:**
- Уменьшите clip_range до 0.1
- Включите value clipping: `clip_range_vf = 0.2`
- Уменьшите learning rate
- Увеличьте n_steps (больше траекторий на update)

---

### Проблема 3: Высокая KL divergence

**Симптомы:** `approx_kl > 0.05` постоянно

**Причина:** Политика меняется слишком быстро

**Решение:**
- Уменьшите clip_range
- Уменьшите learning rate
- Уменьшите n_epochs (меньше обновлений на батче)

---

## 📚 Дополнительные материалы

### Оригинальные статьи:

1. **PPO:** *Proximal Policy Optimization Algorithms* (Schulman et al., 2017)
2. **TRPO:** *Trust Region Policy Optimization* (Schulman et al., 2015)
3. **GAE:** *High-Dimensional Continuous Control Using GAE* (Schulman et al., 2016)

### Имплементации:

- [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Stable-Baselines3: PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [CleanRL: PPO](https://github.com/vwxyzjn/cleanrl)

---

## 🔗 Связь с другими семинарами

### Откуда пришли:
- **[note_11_policy_gradients_reinforce.md](../../notes/md/note_11_policy_gradients_reinforce.md):** REINFORCE — baseline PG
- **[note_12_actor_critic_a2c.md](../../notes/md/note_12_actor_critic_a2c.md):** A2C — Actor-Critic архитектура

### Куда идём:
- **[note_15_rlhf_pipeline.md](../../notes/md/note_15_rlhf_pipeline.md):** RLHF — PPO для fine-tuning LLM
- **[note_16_dpo_and_variants.md](../../notes/md/note_16_dpo_and_variants.md):** DPO — альтернатива PPO-RLHF

---

## 💻 Примеры использования

### Базовое обучение:

```python
from ppo_agent import PPOAgent, PPOConfig

config = PPOConfig(
    env_id="BipedalWalker-v3",
    total_timesteps=1_000_000,
    n_envs=8,
)

agent = PPOAgent(config)
agent.train()
agent.save("ppo_model.pt")
```

### Кастомизация:

```python
config = PPOConfig(
    clip_range=0.1,         # Более консервативный
    gae_lambda=0.99,        # Меньше bias
    n_epochs=15,            # Больше обновлений
    entropy_coef=0.02,      # Больше exploration
)
```

### Evaluation:

```python
agent = PPOAgent(config)
agent.load("ppo_model.pt")

# Запуск эпизодов
env = gym.make("BipedalWalker-v3", render_mode="human")
obs, _ = env.reset()

for _ in range(1000):
    with torch.no_grad():
        action, _, _, _ = agent.network.get_action_and_value(
            torch.tensor(obs).unsqueeze(0)
        )
    obs, reward, terminated, truncated, _ = env.step(action.squeeze().numpy())
    if terminated or truncated:
        break
```

---

**Автор:** Denis Samatov, TPU / 2025

✅ **Семинар 14 завершён!** Переходим к [Семинару 15: RLHF](../15_rlhf_basics/README.md)

