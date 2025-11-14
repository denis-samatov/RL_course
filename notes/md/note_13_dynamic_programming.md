# Теоретический конспект №13

## Тема: Динамическое программирование в Reinforcement Learning

> **Связано с:** [note_02_rl_framework_and_mdp.md](note_02_rl_framework_and_mdp.md) — MDP формализация · [note_07_bellman_equation.md](note_07_bellman_equation.md) — Уравнение Беллмана

---

## 1. Что такое динамическое программирование (DP) в RL?

**Динамическое программирование (Dynamic Programming, DP)** — это класс методов для нахождения оптимальной политики, когда **полностью известна модель среды** (MDP):

- Известны вероятности переходов $P(s'|s,a)$
- Известны награды $r(s,a)$ или $r(s,a,s')$

> «DP — это вычисление оптимальной политики через итеративное применение уравнений Беллмана.»

---

### Model-based vs Model-free

| Подход | Требует модель? | Примеры |
|--------|----------------|---------|
| **Model-based** (DP) | ✅ Да | Policy Iteration, Value Iteration |
| **Model-free** | ❌ Нет | Q-Learning, SARSA, MC, TD |

**Важно:** В реальных задачах модель среды часто неизвестна, поэтому DP применяется редко. Однако изучение DP критически важно, так как:

1. Формирует теоретическую базу для всех RL-алгоритмов
2. Вводит концепцию **Generalized Policy Iteration (GPI)**
3. Показывает, как итеративно улучшать политику

---

## 2. Уравнение Беллмана для оптимальной политики

Напомним уравнения Беллмана (детали в [note_07_bellman_equation.md](note_07_bellman_equation.md)):

**Для функции ценности состояния:**

$$
V_\pi(s) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma V_\pi(S_{t+1}) \mid S_t = s \right]
$$

$$
V_\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_\pi(s') \right]
$$

**Для оптимальной функции ценности:**

$$
V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V^*(s') \right]
$$

**Для Q-функции:**

$$
Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma \max_{a'} Q^*(s',a') \right]
$$

---

## 3. Policy Evaluation (оценка политики)

**Задача:** Дана фиксированная политика $\pi$. Вычислить $V_\pi(s)$ для всех состояний.

### Итеративный алгоритм

Начинаем с произвольных значений $V_0(s)$ и итеративно применяем уравнение Беллмана:

$$
V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_k(s') \right]
$$

**Алгоритм (псевдокод):**

```python
# Инициализация
V = {s: 0 for s in states}
theta = 1e-6  # порог сходимости

while True:
    delta = 0
    for s in states:
        v = V[s]
        # Беллмановское обновление
        V[s] = sum(
            pi[a|s] * sum(
                P[s'|s,a] * (r[s,a,s'] + gamma * V[s'])
                for s' in next_states(s, a)
            )
            for a in actions(s)
        )
        delta = max(delta, abs(v - V[s]))
    
    if delta < theta:
        break
```

**Сходимость:** Гарантируется при $\gamma < 1$ или если все состояния достижимы и конечны.

---

## 4. Policy Improvement (улучшение политики)

**Задача:** Дана $V_\pi(s)$. Построить **лучшую политику** $\pi'$.

### Жадное улучшение

Идея: выбираем действие, которое максимизирует ожидаемую ценность:

$$
\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_\pi(s') \right]
$$

Эквивалентно через Q-функцию:

$$
\pi'(s) = \arg\max_a Q_\pi(s,a)
$$

где

$$
Q_\pi(s,a) = \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_\pi(s') \right]
$$

**Теорема улучшения политики (Policy Improvement Theorem):**

Если $\pi'$ получена жадным улучшением из $\pi$, то:

$$
V_{\pi'}(s) \geq V_\pi(s) \quad \text{для всех } s
$$

Равенство достигается, только если $\pi$ уже оптимальна: $\pi = \pi^*$.

---

## 5. Policy Iteration

**Идея:** Чередуем **оценку** и **улучшение** политики до сходимости.

### Алгоритм Policy Iteration

1. **Инициализация:**
   - Произвольная политика $\pi_0$ (например, равномерная)
   - Произвольные значения $V_0(s) = 0$

2. **Policy Evaluation:**
   - Вычислить $V_{\pi_k}(s)$ для всех $s$ (итеративно, пока не сойдётся)

3. **Policy Improvement:**
   - Для каждого состояния:
     $$
     \pi_{k+1}(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_{\pi_k}(s') \right]
     $$

4. **Проверка сходимости:**
   - Если $\pi_{k+1} = \pi_k$, стоп (найдена $\pi^*$)
   - Иначе вернуться к шагу 2

**Псевдокод (исполняемый Python):**

```python
import numpy as np

# 1. Инициализация
n_states = env.observation_space.n
n_actions = env.action_space.n

# Uniform policy: pi[s, a] = probability of action a in state s
pi = np.ones((n_states, n_actions)) / n_actions
V = np.zeros(n_states)

theta = 1e-6  # Convergence threshold
gamma = 0.9

while True:
    # 2. Policy Evaluation
    while True:
        delta = 0
        V_new = np.zeros(n_states)
        
        for s in range(n_states):
            v = 0.0
            # Сумма по всем действиям
            for a in range(n_actions):
                # Получаем динамику среды: [(prob, next_s, reward, done)]
                transitions = env.get_transition_prob(s, a)
                
                for prob_transition, s_prime, reward, done in transitions:
                    # Беллмановское обновление
                    v += pi[s, a] * prob_transition * (
                        reward + gamma * V[s_prime] * (1 - int(done))
                    )
            
            V_new[s] = v
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        if delta < theta:
            break
    
    # 3. Policy Improvement
    policy_stable = True
    
    for s in range(n_states):
        # Сохраняем старое действие
        old_action = np.argmax(pi[s])
        
        # Вычисляем Q(s, a) для всех действий
        q_values = np.zeros(n_actions)
        
        for a in range(n_actions):
            transitions = env.get_transition_prob(s, a)
            q_sa = 0.0
            
            for prob, s_prime, reward, done in transitions:
                q_sa += prob * (reward + gamma * V[s_prime] * (1 - int(done)))
            
            q_values[a] = q_sa
        
        # Жадное улучшение: выбираем лучшее действие
        best_action = np.argmax(q_values)
        
        # Обновляем политику (детерминированная)
        pi[s] = np.zeros(n_actions)
        pi[s, best_action] = 1.0
        
        # Проверяем стабильность
        if best_action != old_action:
            policy_stable = False
    
    # 4. Проверка сходимости
    if policy_stable:
        print("Policy Iteration converged!")
        break
```

**Сложность одной итерации:** $O(|\mathcal{S}|^2 |\mathcal{A}|)$

**Сходимость:** Гарантируется за **полиномиальное число** итераций.

---

## 6. Value Iteration

**Идея:** Объединить evaluation и improvement в **один шаг**.

Вместо полной оценки $V_\pi$, делаем **одно** обновление Беллмана с максимизацией:

$$
V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_k(s') \right]
$$

Это **оптимальное уравнение Беллмана** в форме обновления.

### Алгоритм Value Iteration

1. **Инициализация:** $V_0(s) = 0$ для всех $s$

2. **Итеративное обновление:**
   - Для каждого состояния $s$:
     $$
     V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_k(s') \right]
     $$
   - Продолжать, пока $\max_s |V_{k+1}(s) - V_k(s)| < \theta$

3. **Извлечение политики:**
   - После сходимости $V^* \approx V_k$:
     $$
     \pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V^*(s') \right]
     $$

**Псевдокод:**

```python
# Инициализация
V = {s: 0 for s in states}
theta = 1e-6

while True:
    delta = 0
    for s in states:
        v = V[s]
        # Оптимальное обновление Беллмана
        V[s] = max(
            sum(P[s'|s,a] * (r[s,a,s'] + gamma * V[s'])
                for s' in next_states(s,a))
            for a in actions(s)
        )
        delta = max(delta, abs(v - V[s]))
    
    if delta < theta:
        break

# Извлечение оптимальной политики
pi = {}
for s in states:
    pi[s] = argmax_a(
        sum(P[s'|s,a] * (r + gamma * V[s'])
            for s' in next_states(s,a))
    )
```

**Сложность одной итерации:** $O(|\mathcal{S}|^2 |\mathcal{A}|)$

**Сходимость:** **Экспоненциальная скорость** — обычно быстрее, чем Policy Iteration.

---

## 7. Generalized Policy Iteration (GPI)

**GPI** — это общая идея, лежащая в основе **всех RL-алгоритмов**:

> «Пусть процесс оценки политики и процесс улучшения политики идут параллельно, взаимодействуя друг с другом.»

### Схема GPI

```
       ┌─────────────┐
       │  Политика π │
       └──────┬──────┘
              │
         (улучшение)
              │
              ▼
       ┌─────────────┐
       │ V-функция V │◄──────┐
       └──────┬──────┘       │
              │              │
         (оценка)       (улучшение)
              │              │
              ▼              │
       ┌─────────────┐       │
       │  Политика π'│───────┘
       └─────────────┘
```

**Ключевые идеи GPI:**

1. **Оценка и улучшение конкурируют:**
   - Оценка делает $V$ согласованной с текущей $\pi$
   - Улучшение делает $\pi$ жадной относительно текущей $V$

2. **Не обязательно ждать полной сходимости:**
   - Policy Iteration: полная оценка перед улучшением
   - Value Iteration: одно обновление оценки перед улучшением
   - Можно делать **асинхронные обновления** (см. ниже)

3. **Гарантия сходимости к $\pi^*$ и $V^*$:**
   - Оба процесса стабилизируются только в оптимуме

---

## 8. Сравнение Policy Iteration и Value Iteration

| Аспект | Policy Iteration | Value Iteration |
|--------|------------------|-----------------|
| **Обновление** | Полная оценка $V_\pi$, затем жадное улучшение | Одно оптимальное обновление Беллмана |
| **Итераций до сходимости** | Мало (3-10 обычно) | Больше (зависит от $\gamma$ и $\theta$) |
| **Стоимость итерации** | Высокая (много подитераций evaluation) | Низкая (один проход по состояниям) |
| **Общее время** | Зависит от задачи | Обычно быстрее на практике |
| **Политика** | Всегда определена явно | Извлекается в конце |
| **Применимость** | Малые MDP (сотни состояний) | Средние MDP (тысячи состояний) |

**Практический совет:**

- Value Iteration обычно предпочтительнее для сред с дискретными состояниями.
- Policy Iteration лучше, если policy evaluation быстро сходится.

---

## 9. Асинхронные методы DP

**Проблема синхронных методов:** Требуется полный проход по **всем состояниям** на каждой итерации → дорого для больших MDP.

**Решение:** Обновлять состояния **асинхронно**, в произвольном порядке.

### Типы асинхронных методов:

1. **In-place updates:**
   - Обновляем $V(s)$ сразу, используя уже обновлённые значения соседей
   - Быстрее сходится (использует самую свежую информацию)

2. **Prioritized Sweeping:**
   - Поддерживаем очередь состояний по приоритету
   - Приоритет = величина ожидаемого изменения $|V_{\text{new}}(s) - V_{\text{old}}(s)|$
   - Обновляем сначала состояния с наибольшим изменением

3. **Real-time DP:**
   - Обновляем только состояния, которые посещает агент
   - Полезно, если большинство состояний недостижимы

**Пример In-place Value Iteration:**

```python
# Синхронный (классический)
V_new = {}
for s in states:
    V_new[s] = max_a bellman_update(s, a, V_old)
V_old = V_new  # копируем целиком

# Асинхронный (in-place)
for s in states:
    V[s] = max_a bellman_update(s, a, V)  # используем V напрямую
```

**Преимущества:**
- Меньше памяти (не нужна копия $V$)
- Быстрее сходится (использует обновлённые значения)

---

## 10. Ограничения динамического программирования

| Ограничение | Описание | Как преодолеть |
|-------------|----------|----------------|
| **Требует модель** | Нужны $P(s'\|s,a)$ и $r(s,a)$ | Model-free методы (Q-Learning, SARSA) |
| **Проклятие размерности** | $O(\|\mathcal{S}\|^2 \|\mathcal{A}\|)$ неприемлемо для больших MDP | Аппроксимация функций (Deep RL) |
| **Дискретные состояния** | Трудно для непрерывных пространств | Discretization или function approximation |
| **Полный проход по состояниям** | Обновляет даже недостижимые состояния | Асинхронные методы, Real-time DP |

**Вывод:**

> DP редко применяется напрямую в современном RL, но его принципы (GPI, итеративное улучшение) лежат в основе всех алгоритмов.

---

## 11. От DP к Model-Free RL

**Связь между методами:**

| DP метод | Model-Free аналог | Ключевое отличие |
|----------|-------------------|------------------|
| Policy Evaluation | Monte Carlo Prediction | Не требует модель, использует sample returns |
| Policy Iteration | SARSA (on-policy TD) | Обновляет Q(s,a) по sample transitions |
| Value Iteration | Q-Learning (off-policy TD) | Обновляет Q(s,a) по sample transitions с max |

**Общая идея:**

$$
\text{DP: } V(s) \leftarrow \mathbb{E}[\cdots] \quad \Rightarrow \quad \text{Model-free: } V(s) \leftarrow \text{sample}
$$

Вместо **полного математического ожидания** (требует модель) используем **сэмплированные траектории** (не требует модель).

---

## 12. Практический пример: GridWorld

Рассмотрим сетку 4×4 с одним препятствием и целевым состоянием:

```
┌───┬───┬───┬───┐
│ S │   │   │ G │  S = Start, G = Goal
├───┼───┼───┼───┤
│   │ X │   │   │  X = Obstacle
├───┼───┼───┼───┤
│   │   │   │   │
├───┼───┼───┼───┤
│   │   │   │   │
└───┬───┬───┴───┘
```

**MDP:**
- Состояния: 16 клеток (15 обычных + 1 терминальное Goal)
- Действия: {↑, ↓, ←, →}
- Переходы: детерминированные (если не стена)
- Награды: -1 за каждый шаг, +10 за достижение Goal

**Применение Value Iteration:**

```python
# Псевдокод для GridWorld
V = np.zeros((4, 4))
gamma = 0.9
theta = 1e-4

while True:
    delta = 0
    for i in range(4):
        for j in range(4):
            if (i,j) == (0,3):  # Goal
                continue
            if (i,j) == (1,1):  # Obstacle
                continue
            
            v = V[i,j]
            # Макс по 4 направлениям
            values = []
            for action in ['up', 'down', 'left', 'right']:
                ni, nj = next_pos(i, j, action)
                reward = 10 if (ni,nj)==(0,3) else -1
                values.append(reward + gamma * V[ni, nj])
            
            V[i,j] = max(values)
            delta = max(delta, abs(v - V[i,j]))
    
    if delta < theta:
        break
```

**Результат:**

После сходимости $V(s)$ показывает "расстояние до цели" (с учётом $\gamma$).

Оптимальная политика: стрелки, указывающие к Goal.

---

## 13. Резюме

| Концепция | Описание |
|-----------|----------|
| **DP** | Итеративное применение уравнений Беллмана при известной модели MDP |
| **Policy Evaluation** | Вычисление $V_\pi(s)$ для фиксированной политики |
| **Policy Improvement** | Жадное улучшение политики на основе $V$ |
| **Policy Iteration** | Чередование evaluation и improvement до сходимости |
| **Value Iteration** | Комбинация evaluation и improvement в одном оптимальном обновлении |
| **GPI** | Общая схема взаимодействия оценки и улучшения (основа всех RL) |
| **Асинхронные методы** | Обновление подмножества состояний для ускорения |

**Ключевые выводы:**

1. DP требует полного знания MDP → применим только в симуляциях
2. Принципы DP (GPI) лежат в основе всех RL-алгоритмов
3. Model-free методы заменяют математическое ожидание на сэмплирование
4. Value Iteration обычно эффективнее Policy Iteration на практике

---

## 14. Связь с предыдущими семинарами

- **[note_02_rl_framework_and_mdp.md](note_02_rl_framework_and_mdp.md):** Формализация MDP (состояния, действия, переходы, награды)
- **[note_07_bellman_equation.md](note_07_bellman_equation.md):** Уравнение Беллмана — основа DP
- **[note_08_monte_carlo_vs_td.md](note_08_monte_carlo_vs_td.md):** Monte Carlo — model-free альтернатива Policy Evaluation
- **[note_09_q_learning.md](note_09_q_learning.md):** Q-Learning — model-free альтернатива Value Iteration

---

## 15. Дальнейшее изучение

Рекомендуемые источники:

- **Sutton & Barto, Chapter 4:** Dynamic Programming
- **Maxim Lapan, Chapter 5:** Tabular Learning and the Bellman Equation
- **David Silver's RL Course, Lecture 3:** Planning by Dynamic Programming

---

## 16. Практическое задание

В директории `code/13_dynamic_programming/` реализованы:

1. **GridWorld environment** — кастомная среда с препятствиями
2. **Policy Evaluation** — итеративная оценка произвольной политики
3. **Policy Iteration** — полный алгоритм с чередованием evaluation/improvement
4. **Value Iteration** — оптимальное обновление Беллмана
5. **Визуализация** — тепловые карты $V(s)$, стрелки политики, анимация сходимости

**Эксперименты:**
- Сравнить скорость сходимости Policy Iteration vs Value Iteration
- Изучить влияние $\gamma$ на оптимальную политику
- Реализовать Prioritized Sweeping и сравнить с синхронным методом

---

**Далее:** [note_14_ppo_trpo.md](note_14_ppo_trpo.md) — PPO и TRPO

