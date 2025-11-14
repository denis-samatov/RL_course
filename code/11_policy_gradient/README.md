# Policy Gradient (REINFORCE) на LunarLander-v2

## 📘 Описание

Реализация алгоритма **REINFORCE** (Monte Carlo Policy Gradient) для задачи посадки лунного модуля. Демонстрирует **policy-based методы**, которые напрямую учат стохастическую политику через градиентный подъём по ожидаемому вознаграждению.

**Среда:** LunarLander-v2  
**Тип действий:** Дискретные (4 действия)  
**Состояние:** Непрерывное (8-мерное)  
**Цель:** Посадить лунный модуль между флагами, минимизируя расход топлива

---

## 🎯 Особенности реализации

### Алгоритм REINFORCE
- ✅ **Чистый policy gradient** с Monte Carlo оценкой возврата
- ✅ **Baseline (value function)** для снижения дисперсии градиента
- ✅ **Entropy regularization** для поддержания exploration
- ✅ **Gradient clipping** для стабильности обучения
- ✅ **Advantage normalization** для улучшения сходимости

### Архитектура
```
State (8) → FC(128) → ReLU → FC(128) → ReLU → Logits(4) → Softmax → Policy
State (8) → FC(128) → ReLU → FC(128) → ReLU → Value(1)
```

### Ключевые формулы

**Policy Gradient Theorem:**
```
∇_θ J(θ) = E_τ [ Σ_t ∇_θ log π_θ(a_t|s_t) · (G_t - V(s_t)) ]
```

**Advantage (с baseline):**
```
A_t = G_t - V_φ(s_t)
где G_t = Σ_{k=t}^T γ^(k-t) r_{k+1}
```

---

## 🚀 Быстрый старт

### Установка зависимостей

```bash
# Из корня репозитория
pip install -r requirements.txt

# Дополнительно для LunarLander
pip install gymnasium[box2d]
```

### Базовый запуск

```bash
cd code/policy_gradient
python lunarlander_reinforce.py
```

### Запуск с параметрами

```bash
# Обучение на 3000 эпизодов с записью видео
python lunarlander_reinforce.py --episodes 3000 --record-video

# Без baseline (чистый REINFORCE)
python lunarlander_reinforce.py --baseline False --episodes 2000

# С увеличенной энтропией для exploration
python lunarlander_reinforce.py --entropy 0.05

# Изменить learning rate
python lunarlander_reinforce.py --lr 1e-3
```

### Все параметры

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--episodes` | 2000 | Число эпизодов обучения |
| `--lr` | 3e-4 | Learning rate для политики |
| `--baseline` | True | Использовать value baseline |
| `--entropy` | 0.01 | Коэффициент энтропийной регуляризации |
| `--seed` | 42 | Random seed для воспроизводимости |
| `--record-video` | False | Записать видео лучших эпизодов |

---

## 📊 Ожидаемые результаты

### Сходимость

**Типичная кривая обучения:**
```
Episode 0-500:    Reward ~ -300 to -100 (случайные действия)
Episode 500-1000: Reward ~ -100 to 0    (учится мягкой посадке)
Episode 1000-1500: Reward ~ 0 to 150   (стабильная посадка)
Episode 1500-2000: Reward ~ 150 to 250 (оптимальная стратегия)
```

**Критерий решения:** Средняя награда >= 200 за 100 последовательных эпизодов

### Пример вывода

```
==========================================================
REINFORCE on LunarLander-v2
==========================================================
Episodes: 2000
Learning rate: 0.0003
Baseline: True
Entropy coefficient: 0.01
Seed: 42
==========================================================
Training REINFORCE: 100%|████████| 2000/2000 [12:34<00:00, reward=-45.2, avg_100=178.3, length=234]

Evaluating trained policy...
Evaluation over 100 episodes: 203.45 ± 38.22
✓ Environment SOLVED! (Average reward >= 200)
Model saved to lunarlander_reinforce.pt
```

---

## 📈 Визуализация

После обучения автоматически генерируется график `lunarlander_reinforce_training.png`:

- **Левая панель:** Награды по эпизодам с rolling average
- **Правая панель:** Длительность эпизодов
- **Красная линия:** Порог решения (reward = 200)

---

## 🎥 Запись видео

```bash
python lunarlander_reinforce.py --record-video
```

Видео сохраняются в `videos/lunarlander/`:
- 5 лучших эпизодов после обучения
- Формат: MP4
- FPS: 30

---

## 🔬 Эксперименты

### 1. Влияние baseline

```bash
# С baseline (низкая дисперсия)
python lunarlander_reinforce.py --baseline --episodes 1500

# Без baseline (высокая дисперсия)
python lunarlander_reinforce.py --episodes 1500
```

**Ожидаемый результат:** С baseline сходимость быстрее на ~30-40%

### 2. Влияние энтропии

```bash
# Низкая энтропия (быстрая сходимость, риск локального минимума)
python lunarlander_reinforce.py --entropy 0.001

# Высокая энтропия (медленнее, но лучше exploration)
python lunarlander_reinforce.py --entropy 0.05
```

### 3. Разные learning rates

```bash
python lunarlander_reinforce.py --lr 1e-3  # Быстрее, но нестабильнее
python lunarlander_reinforce.py --lr 1e-4  # Медленнее, но стабильнее
python lunarlander_reinforce.py --lr 5e-4  # Компромисс
```

---

## 🧪 Связь с теорией

Этот код реализует концепции из **note_11_policy_gradients_reinforce.md**:

| Концепция | Реализация в коде |
|-----------|-------------------|
| Policy Gradient Theorem | `train_episode()` → вычисление градиента |
| REINFORCE | Полный Monte Carlo возврат `compute_returns()` |
| Baseline | `ValueNetwork` и вычитание `values.detach()` |
| Advantage | `advantages = returns - values.detach()` |
| Entropy regularization | `entropy_loss = -entropies.mean()` |
| Gradient clipping | `torch.nn.utils.clip_grad_norm_()` |

---

## 📚 Дополнительные материалы

**Теория:**
- `/notes/md/note_11_policy_gradients_reinforce.md` — Policy Gradients и REINFORCE
- `/notes/md/note_04_policy_vs_value_methods.md` — Policy-Based vs Value-Based методы

**Код:**
- `PolicyNetwork` — дискретная стохастическая политика
- `ValueNetwork` — baseline для снижения дисперсии
- `compute_returns()` — Monte Carlo оценка возврата

**Литература:**
- Williams (1992): "Simple Statistical Gradient-Following Algorithms"
- Sutton & Barto (2020): Chapter 13 - Policy Gradient Methods
- Schulman et al. (2015): "High-Dimensional Continuous Control Using GAE"

---

## 🐛 Troubleshooting

### Проблема: Не сходится после 2000 эпизодов

**Решение:**
```bash
# Увеличить число эпизодов
python lunarlander_reinforce.py --episodes 3000

# Или уменьшить learning rate
python lunarlander_reinforce.py --lr 1e-4 --episodes 2000
```

### Проблема: Слишком высокая дисперсия наград

**Решение:**
```bash
# Убедиться, что baseline включен
python lunarlander_reinforce.py --baseline

# Увеличить размер сети
# (в коде изменить hidden_sizes)
```

### Проблема: ImportError для box2d

**Решение:**
```bash
pip install gymnasium[box2d]
# или
pip install box2d-py swig
```

---

## 📊 Benchmark

**Система:** MacBook Pro M2, 16GB RAM  
**Время обучения:** ~12-15 минут (2000 эпизодов)  
**Память:** ~200-300 MB  
**Решено за:** ~1500-1800 эпизодов (с baseline)

---

## 🎓 Домашнее задание

1. **Запустите базовое обучение** и достигните reward >= 200
2. **Сравните** REINFORCE с/без baseline (постройте графики)
3. **Экспериментируйте** с entropy coefficient (0.001, 0.01, 0.05)
4. **Реализуйте** n-step returns вместо полного MC
5. **Попробуйте** другую среду (CartPole-v1)

---

**Автор:** Denis Samatov, TPU / 2025  
**Связь с курсом:** Семинар 11 — Policy Gradient Methods

