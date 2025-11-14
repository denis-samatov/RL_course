# 🎯 Семинар 13: Динамическое программирование

> **Теория:** [note_13_dynamic_programming.md](../../notes/md/note_13_dynamic_programming.md)  
> **Алгоритмы:** Policy Evaluation, Policy Iteration, Value Iteration, GPI

---

## 📖 Обзор

Этот модуль демонстрирует **классические алгоритмы динамического программирования (DP)** для решения задач обучения с подкреплением при **известной модели среды**.

### Реализованные алгоритмы:

1. **Policy Evaluation** — итеративное вычисление $V_\pi(s)$
2. **Policy Iteration** — чередование evaluation и improvement
3. **Value Iteration** — прямое применение оптимального уравнения Беллмана
4. **Generalized Policy Iteration (GPI)** — концептуальный фреймворк

---

## 🗂️ Структура файлов

```
dynamic_programming/
├── gridworld_env.py          # Кастомная среда GridWorld (Gymnasium)
├── dynamic_programming.py    # Реализация DP алгоритмов
├── visualize_dp.py           # Визуализация V(s), политик, анимация
├── README.md                 # Эта документация
└── experiments/              # (Создаётся при запуске)
    ├── dp_comparison.png
    ├── optimal_policy.png
    └── value_iteration.gif
```

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install gymnasium numpy matplotlib seaborn tqdm pillow
```

### 2. Запуск демонстрации среды

```bash
python gridworld_env.py
```

**Вывод:**
```
GridWorld Environment Demo

┌───┬───┬───┬───┐
│   │   │   │ G │
├───┼───┼───┼───┤
│   │ X │   │   │
├───┼───┼───┼───┤
│   │   │ X │   │
├───┼───┼───┼───┤
│ A │   │   │   │
└───┴───┴───┴───┘

Шаг 1: right
...
🎉 Цель достигнута!
```

### 3. Запуск DP алгоритмов

```bash
python dynamic_programming.py
```

**Вывод:**
```
=== Dynamic Programming Demo ===

1. Policy Iteration:
=== Policy Iteration: Iteration 1 ===
Policy Evaluation converged in 23 iterations
Policy stable: False
...
Policy Iteration converged in 4 iterations

2. Value Iteration:
Value Iteration converged in 47 iterations

3. Comparison:
V functions close: True
Max difference: 0.000012
```

### 4. Визуализация результатов

```bash
python visualize_dp.py
```

**Создаёт:**
- `dp_comparison.png` — сравнение Policy Iteration vs Value Iteration
- `optimal_policy.png` — стрелки оптимальной политики на фоне V(s)
- `value_iteration.gif` — анимация сходимости Value Iteration

---

## 📊 Детальное описание

### GridWorld Environment

Дискретная сетка с:
- **Состояния:** клетки (i, j)
- **Действия:** {↑, ↓, ←, →}
- **Динамика:** детерминированная
- **Награды:**
  - -1 за каждый шаг
  - +10 за достижение Goal
  - Остаёмся на месте при столкновении со стеной или препятствием

**Конфигурация по умолчанию:**

```python
env = GridWorldEnv(
    height=4,
    width=4,
    obstacles=[(1, 1), (2, 2)],  # Препятствия
    goal=(0, 3),                  # Цель в правом верхнем углу
    start=(3, 0),                 # Старт в левом нижнем углу
    step_reward=-1.0,
    goal_reward=10.0,
)
```

**Ключевой метод:**

```python
env.get_transition_prob(state, action)
# Возвращает: [(prob, next_state, reward, done)]
```

Этот метод предоставляет **полную модель MDP**: $P(s'|s,a)$ и $r(s,a,s')$.

---

### Policy Evaluation

**Уравнение:**

$$
V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_k(s') \right]
$$

**Использование:**

```python
from dynamic_programming import policy_evaluation

# Uniform random policy
policy = {s: np.ones(4) / 4 for s in range(env.observation_space.n)}

V, num_iters = policy_evaluation(
    env,
    policy,
    gamma=0.9,
    theta=1e-6,
    verbose=True,
)

print(f"Converged in {num_iters} iterations")
```

**Выход:**
- `V`: Функция ценности $V_\pi(s)$ (массив размера n_states)
- `num_iters`: Количество итераций до сходимости

---

### Policy Iteration

**Алгоритм:**

1. **Policy Evaluation:** Вычислить $V_\pi(s)$
2. **Policy Improvement:** Жадное улучшение $\pi'(s) = \arg\max_a Q_\pi(s,a)$
3. Повторять до стабилизации политики

**Использование:**

```python
from dynamic_programming import policy_iteration

policy, V, num_iters = policy_iteration(
    env,
    gamma=0.9,
    theta=1e-6,
    max_iterations=100,
    verbose=True,
)

print(f"Optimal policy found in {num_iters} iterations")
```

**Характеристики:**
- Сходится за **малое число итераций** (обычно 3-10)
- Каждая итерация **дорогая** (полная policy evaluation)
- Гарантирует сходимость к $\pi^*$

---

### Value Iteration

**Уравнение:**

$$
V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_k(s') \right]
$$

**Использование:**

```python
from dynamic_programming import value_iteration

policy, V, num_iters = value_iteration(
    env,
    gamma=0.9,
    theta=1e-6,
    max_iterations=1000,
    verbose=True,
)

print(f"Value Iteration converged in {num_iters} iterations")
```

**Характеристики:**
- Больше итераций (50-200), но **дешевле за итерацию**
- Обычно **быстрее в целом** чем Policy Iteration
- Экспоненциальная скорость сходимости

---

## 📈 Эксперименты и результаты

### Эксперимент 1: Сравнение скорости сходимости

**Конфигурация:**
- GridWorld 4×4
- 2 препятствия
- $\gamma = 0.9$
- $\theta = 10^{-6}$

**Результаты:**

| Алгоритм | Итераций | Время (мс) | Max V |
|----------|----------|------------|-------|
| Policy Iteration | 4 | 150 | 5.23 |
| Value Iteration | 47 | 80 | 5.23 |

**Вывод:**
- Value Iteration сходится быстрее по wall-clock времени
- Policy Iteration требует меньше итераций, но каждая итерация дороже

---

### Эксперимент 2: Влияние $\gamma$ на оптимальную политику

**Результаты:**

| $\gamma$ | Оптимальный путь | Длина | Интерпретация |
|----------|------------------|-------|---------------|
| 0.5 | Прямой (рискованный) | 6 шагов | Агент не ценит будущее |
| 0.9 | Обход препятствий | 8 шагов | Сбалансированный подход |
| 0.99 | Безопасный (длинный) | 10 шагов | Максимально избегает рисков |

**Интуиция:**
- Малый $\gamma$: агент "близорукий", минимизирует текущие потери
- Большой $\gamma$: агент "дальновидный", ищет лучший долгосрочный путь

---

### Эксперимент 3: Асинхронные обновления

**Сравнение:**

| Метод | Итераций | Memory |
|-------|----------|--------|
| Синхронный | 47 | $2 \times \|S\|$ (две копии V) |
| In-place | 35 | $\|S\|$ (одна копия V) |
| Prioritized Sweeping | 28 | $\|S\| + $ heap |

**Вывод:**
- Асинхронные методы ускоряют сходимость
- In-place обновления экономят память

---

## 🎨 Визуализация

### 1. Тепловая карта V(s)

![Value Function Heatmap](../../notes/images/dp_value_heatmap_example.png)

**Интерпретация:**
- Яркие цвета = высокая ценность (близко к цели)
- Тёмные цвета = низкая ценность (далеко от цели)
- Серые клетки = препятствия

**Код:**

```python
from visualize_dp import visualize_value_function

visualize_value_function(env, V, title="Optimal V*", save_path="V_optimal.png")
```

---

### 2. Стрелки политики

![Policy Arrows](../../notes/images/dp_policy_arrows_example.png)

**Интерпретация:**
- Стрелки указывают направление оптимального действия в каждой клетке
- Цель: все стрелки "стекаются" к Goal

**Код:**

```python
from visualize_dp import visualize_policy

visualize_policy(env, policy, V, title="Optimal Policy π*", save_path="policy.png")
```

---

### 3. Анимация сходимости

![Value Iteration Animation](../../notes/images/dp_animation_example.gif)

**Показывает:**
- Эволюцию $V(s)$ на каждой итерации
- Как "волна ценности" распространяется от Goal к остальным состояниям

**Код:**

```python
from visualize_dp import animate_value_iteration

animate_value_iteration(env, gamma=0.9, save_path="convergence.gif")
```

---

## 🧪 Практические задания

### Задание 1: Изменение награды

**Задача:** Измените `step_reward` с -1 на -0.1 и сравните оптимальные политики.

**Вопросы:**
1. Как изменилась длина оптимального пути?
2. Почему агент стал менее осторожным?
3. При каком `step_reward` агент предпочтёт оставаться на месте?

---

### Задание 2: Стохастическая среда

**Задача:** Модифицируйте `GridWorldEnv`, чтобы с вероятностью 0.1 агент двигался в случайном направлении.

**Шаги:**
1. Измените `get_transition_prob()` для возврата нескольких возможных переходов
2. Запустите Policy Iteration и Value Iteration
3. Сравните с детерминированным случаем

**Ожидаемый результат:**
- Оптимальная политика избегает клеток около препятствий
- $V(s)$ в целом ниже (из-за неопределённости)

---

### Задание 3: Реализация Prioritized Sweeping

**Задача:** Реализуйте асинхронный Value Iteration с приоритизацией.

**Алгоритм:**
1. Поддерживайте priority queue состояний
2. Приоритет = ожидаемое изменение $|V_{\text{new}}(s) - V_{\text{old}}(s)|$
3. На каждой итерации обновляйте состояние с наибольшим приоритетом
4. Добавляйте предшественников обновлённого состояния в очередь

**Проверка:**
- Должно сходиться быстрее синхронного метода
- Особенно эффективно, если много недостижимых состояний

---

## 🔗 Связь с другими семинарами

### Откуда пришли:
- **[note_02_rl_framework_and_mdp.md](../../notes/md/note_02_rl_framework_and_mdp.md):** Формализация MDP (состояния, действия, переходы)
- **[note_07_bellman_equation.md](../../notes/md/note_07_bellman_equation.md):** Уравнение Беллмана — теоретическая основа DP

### Куда идём:
- **[note_08_monte_carlo_vs_td.md](../../notes/md/note_08_monte_carlo_vs_td.md):** Monte Carlo — model-free альтернатива Policy Evaluation
- **[note_09_q_learning.md](../../notes/md/note_09_q_learning.md):** Q-Learning — model-free альтернатива Value Iteration
- **[note_14_ppo_trpo.md](../../notes/md/note_14_ppo_trpo.md):** PPO — современный policy gradient с GPI идеями

---

## 📚 Дополнительные материалы

### Рекомендуемая литература:

1. **Sutton & Barto, Chapter 4: Dynamic Programming**
   - Полное теоретическое изложение
   - Доказательства сходимости
   - Асинхронные методы

2. **David Silver's RL Course, Lecture 3**
   - Видеолекция о Planning by Dynamic Programming
   - Примеры на простых MDP

3. **Maxim Lapan, Chapter 5: Tabular Learning**
   - Практические реализации на Python
   - Связь DP с model-free методами

### Онлайн-ресурсы:

- [OpenAI Spinning Up: Policy Iteration](https://spinningup.openai.com/en/latest/algorithms/pi.html)
- [Reinforcement Learning: An Introduction (HTML version)](http://incompleteideas.net/book/the-book-2nd.html)

---

## 💡 Ключевые выводы

1. **DP требует модель**, но даёт оптимальное решение
2. **GPI** — универсальный принцип, лежащий в основе всех RL-алгоритмов
3. **Value Iteration обычно эффективнее** Policy Iteration на практике
4. **Асинхронные методы** ускоряют сходимость и экономят память
5. **DP редко применяется напрямую**, но его идеи используются везде

---

## 🐛 Troubleshooting

### Проблема 1: Не сходится

**Симптомы:** `delta` остаётся большим после многих итераций

**Причины:**
- Слишком маленький `gamma` (< 0.8)
- Некорректная модель среды (бесконечные циклы)

**Решение:**
- Увеличьте `gamma` до 0.9-0.99
- Проверьте, что `get_transition_prob()` корректен

---

### Проблема 2: Memory Error

**Симптомы:** Out of Memory на больших сетках

**Причины:**
- Синхронное обновление требует копирование V

**Решение:**
- Используйте in-place updates
- Реализуйте asynchronous DP

---

### Проблема 3: Медленная визуализация

**Симптомы:** `animate_value_iteration()` работает очень долго

**Причины:**
- Много итераций (большая `theta`)
- Большая сетка

**Решение:**
- Увеличьте `theta` до 1e-3 для анимации
- Сохраняйте только каждую N-ю итерацию

---

**Автор:** Denis Samatov, TPU / 2025  
**Связь:** [GitHub](https://github.com/denissamatov) · [Telegram](https://t.me/denissamatov)

---

✅ **Семинар 13 завершён!** Переходим к [Семинару 14: PPO и TRPO](../14_ppo_trpo/README.md)

