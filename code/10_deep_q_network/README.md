# 🎯 Семинар 10: Deep Q-Network (DQN)

> **Теория:** [note_10_deep_q_network.md](../../notes/md/note_10_deep_q_network.md)  
> **Алгоритм:** Deep Q-Network с Experience Replay и Target Network

---

## 📖 Обзор

Полная реализация **Deep Q-Network (DQN)** — первого успешного применения глубоких нейронных сетей в Reinforcement Learning. DQN решает проблему масштабирования табличного Q-Learning на большие пространства состояний.

**Среда:** LunarLander-v2  
**Тип действий:** Дискретные (4 действия)  
**Состояние:** Непрерывное (8-мерное)  
**Цель:** Посадить лунный модуль между флагами, минимизируя расход топлива

---

## 🎯 Особенности реализации

### Алгоритм DQN

- ✅ **Deep Q-Network** — MLP для аппроксимации Q-функции
- ✅ **Experience Replay** — буфер для хранения и переиспользования переходов
- ✅ **Target Network** — стабилизация обучения через периодическое копирование весов
- ✅ **ε-greedy exploration** — баланс между exploration и exploitation
- ✅ **Gradient clipping** — предотвращение gradient explosion

### Архитектура

```text
State (8) → FC(128) → ReLU → FC(128) → ReLU → Q-values(4)
```

### Ключевые формулы

**TD-target:**

```text
y = r + γ * max_a' Q_target(s', a')  (если не done)
y = r                                 (если done)
```

**Loss (MSE):**

```text
L(θ) = E[(y - Q_θ(s, a))²]
```

**Target Network Update:**

```text
θ_target ← θ  (каждые C шагов)
```

---

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install gymnasium[box2d] torch numpy matplotlib tqdm
```

### Базовый запуск

```bash
cd code/10_deep_q_network
python dqn_algorithm.py
```

### Запуск с параметрами

```bash
# Обучение на 80k шагов
python dqn_algorithm.py --total-steps 80000

# С другим learning rate
python dqn_algorithm.py --lr 5e-4

# С увеличенным буфером
python dqn_algorithm.py --buffer-size 100000
```

### Все параметры

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--total-steps` | 50000 | Общее число шагов обучения |
| `--lr` | 1e-3 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--epsilon-start` | 1.0 | Начальный epsilon |
| `--epsilon-end` | 0.05 | Конечный epsilon |
| `--epsilon-decay` | 0.997 | Экспоненциальный decay |
| `--buffer-size` | 50000 | Размер replay buffer |
| `--batch-size` | 64 | Размер mini-batch |
| `--target-update` | 1000 | Частота обновления target network |
| `--warmup-steps` | 2000 | Шагов до начала обучения |
| `--seed` | 42 | Random seed |

---

## 📊 Ожидаемые результаты

### Сходимость

**Типичная кривая обучения:**

```text
Steps 0-10000:   Reward ~ -200 to 0    (жёсткие посадки)
Steps 10000-25000: Reward ~ 0 to 150    (учится мягкой посадке)
Steps 25000-40000: Reward ~ 150 to 220  (стабильная посадка)
Steps 40000-50000: Reward ~ 220 to 260  (оптимальная стратегия)
```

**Критерий решения:** Средняя награда >= 200 за 100 последовательных эпизодов

### Пример вывода

```text
==========================================================
DQN Training on LunarLander-v2
==========================================================
Total steps: 50000
Learning rate: 0.001
Buffer size: 50000
Batch size: 64
Target update: 1000
Environment: LunarLander-v2
==========================================================
Training: 100%|████████| 50000/50000 [14:37<00:00, step=3425, reward=236.5, epsilon=0.05]

Evaluating trained policy...
Evaluation over 100 episodes: 212.7 ± 34.8
✓ Environment SOLVED! (Average reward >= 200)
Model saved to dqn_lunarlander.pt
```

---

## 📈 Визуализация

После обучения автоматически генерируется график `dqn_training.png`:

- **Левая панель:** Награды по эпизодам с rolling average
- **Правая панель:** Epsilon decay schedule
- **Красная линия:** Порог решения (reward = 200)

---

## 🔬 Эксперименты

### 1. Влияние Target Network

```bash
# С target network (стабильное обучение)
python dqn_algorithm.py --target-update 1000

# Без target network (нестабильное)
python dqn_algorithm.py --target-update 1
```

**Ожидаемый результат:** С target network сходимость стабильнее на 30-40%

### 2. Влияние Replay Buffer

```bash
# Большой буфер (лучше для стабильности)
python dqn_algorithm.py --buffer-size 100000

# Маленький буфер (быстрее, но нестабильнее)
python dqn_algorithm.py --buffer-size 10000
```

### 3. Разные расписания epsilon

```bash
# Быстрый decay (быстрее exploitation)
python dqn_algorithm.py --epsilon-decay 0.995

# Медленный decay (больше exploration)
python dqn_algorithm.py --epsilon-decay 0.999
```

---

## 🧪 Связь с теорией

Этот код реализует концепции из **note_10_deep_q_network.md**:

| Концепция | Реализация в коде |
|-----------|-------------------|
| Deep Q-Network | `DQN` класс — MLP для Q(s,a) |
| Experience Replay | `ReplayBuffer` — хранение и сэмплирование переходов |
| Target Network | `target_net.load_state_dict(...)` каждые N шагов |
| TD-target | `compute_td_target()` — r + γ * max Q_target |
| ε-greedy | `epsilon_schedule()` + `select_action()` |
| Gradient clipping | `torch.nn.utils.clip_grad_norm_()` |

**Код:**

- `dqn_algorithm.py` — полная реализация DQN для LunarLander-v2
- `homework.ipynb` — практические задания
- `homework_solution.ipynb` — решения с комментариями

**Литература:**

- Mnih et al. (2015): "Human-level control through deep reinforcement learning" (Nature DQN)
- Van Hasselt et al. (2016): "Deep Reinforcement Learning with Double Q-learning" (Double DQN)
- Wang et al. (2016): "Dueling Network Architectures" (Dueling DQN)

---

## 🐛 Troubleshooting

### Проблема: Не сходится после 50000 шагов

**Решение:**

```bash
# Увеличить число шагов
python dqn_algorithm.py --total-steps 80000

# Или уменьшить learning rate
python dqn_algorithm.py --lr 5e-4 --total-steps 20000
```

### Проблема: Нестабильное обучение

**Решение:**

```bash
# Увеличить частоту обновления target network
python dqn_algorithm.py --target-update 500

# Увеличить размер буфера
python dqn_algorithm.py --buffer-size 100000
```

### Проблема: Слишком медленное обучение

**Решение:**

```bash
# Увеличить learning rate
python dqn_algorithm.py --lr 2e-3

# Уменьшить batch size (больше обновлений)
python dqn_algorithm.py --batch-size 32
```

---

## 📊 Benchmark

**Система:** MacBook Pro M2, 16GB RAM  
**Время обучения:** ~12-15 минут (50000 шагов)  
**Память:** ~300-400 MB  
**Решено за:** ~40000-50000 шагов

---

## 🎓 Домашнее задание

1. **Запустите базовое обучение** и достигните reward >= 200
2. **Сравните** DQN с/без target network (постройте графики)
3. **Экспериментируйте** с размером replay buffer (10K, 50K, 100K)
4. **Реализуйте** Double DQN (см. homework.ipynb)
5. **Добавьте** Prioritized Experience Replay или Dueling Network

---

**Автор:** Denis Samatov, TPU / 2025  
**Связь с курсом:** Семинар 10 — Deep Q-Network
