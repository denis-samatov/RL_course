# Теоретический конспект №8

## Тема: Monte Carlo vs Temporal Difference Learning (MC vs TD)

> **Связано с:** [note_07_bellman_equation.md](note_07_bellman_equation.md) — Уравнение Беллмана · [note_06_value_based_methods.md](note_06_value_based_methods.md) — V и Q функции

---

## 1. Почему этот блок важен?

После того как мы разобрали уравнение Беллмана, возникает практический вопрос:

> «Как агент может *приближённо вычислить* эту ценность, если у него нет доступа ко всей среде и всем будущим вознаграждениям?»

Два базовых подхода:

* **Monte Carlo (MC)** — обучение по *завершённым эпизодам*;
* **Temporal Difference (TD)** — обучение *во время взаимодействия*, шаг за шагом.

---

## 2. Monte Carlo: обучение по целому эпизоду

**Интуиция:** агент **сначала доигрывает весь эпизод**, затем агрегирует вознаграждения и обновляет оценки ценностей.

### Математически

$$
V(S_t) \leftarrow V(S_t) + \alpha \big[G_t - V(S_t)\big]
$$
где совокупное вознаграждение (возврат)
$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
$$

### Что происходит на практике

1. Агент начинает эпизод и действует по текущей политике (например, $\varepsilon$‑жадной).
2. После окончания эпизода вычисляет $G_t$ для всех временных шагов.
3. Обновляет все посещённые состояния $S_t$ с использованием соответствующих $G_t$.

### Пример

Пусть награды за эпизод: $R = [1, 0, 0, 0, 1, 1]$.
Тогда $G_0 = 1 + 0 + 0 + 0 + 1 + 1 = 3$. При скорости обучения $\alpha = 0.1$:
$$
V(S_0) \leftarrow 0 + 0.1\,(3 - 0) = 0.3
$$
После нескольких эпизодов оценки $V(S_t)$ начинают отражать ожидаемое качество состояния.

### Преимущества MC

* Прост в понимании и реализации.
* Не требует знания динамики среды $P(s'\mid s,a)$.
* Даёт **несмещённые** оценки $V_\pi(s)$ (при достаточной выборке).

### Недостатки MC

* Требуется дождаться конца эпизода.
* Неэффективен/неприменим в бесконечных задачах (continuous tasks без естественных терминальных состояний).
* Обновления редкие → высокая дисперсия оценок.

---

## 3. Temporal Difference (TD): обучение по ходу взаимодействия

**Интуиция:** агент **обновляет оценки сразу после каждого шага**, не дожидаясь конца эпизода, используя текущую оценку будущей ценности $V(S_{t+1})$.

### Формула TD(0)

$$
V(S_t) \leftarrow V(S_t) + \alpha \big[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \big]
$$
Здесь
$$
R_{t+1} + \gamma V(S_{t+1})
$$
— **TD‑цель** (*TD target*), а выражение в квадратных скобках — **TD‑ошибка** (*TD error*, $\delta_t$):
$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t).
$$

### Интуиция (пример)

Если $R_{t+1}=1$ и $V(S_{t+1})=0$, то при $\alpha=0.1,\ \gamma=1$:
$$
V(S_0) \leftarrow 0 + 0.1\,[1 + 1\cdot 0 - 0] = 0.1.
$$
Мы обновили значение **сразу**, без ожидания завершения эпизода.

### Ключевая особенность TD — бутстрэппинг

TD *частично* опирается на собственные оценки будущего $V(S_{t+1})$ (бутстрэппинг), а не на полный возврат $G_t$ как в MC.

---

## 4. Сравнение Monte Carlo и Temporal Difference

| Критерий                       | Monte Carlo                          | Temporal Difference                          |
| ------------------------------ | ------------------------------------ | -------------------------------------------- |
| Основа                         | Полный эпизод                        | Один шаг                                     |
| Что используется               | Реальное $G_t$                     | Приближённое $R_{t+1} + \gamma V(S_{t+1})$ |
| Обновления                     | После эпизода                        | После каждого шага                           |
| Требует завершения эпизода     | Да                                   | Нет                                          |
| Тип оценки                     | Несмещённая, но с большой дисперсией | Смещённая, но с меньшей дисперсией           |
| Подходит для бесконечных задач | Нет                                  | Да                                           |
| Бутстрэппинг                   | Нет                                  | Да                                           |
| Примеры                        | Blackjack, короткие эпизоды          | CartPole, FrozenLake                         |

---

## 5. Объединяющая идея: TD($\lambda$)

На практике часто комбинируют оба подхода: **MC** даёт точность, **TD** — скорость. Промежуточная форма — **TD($\lambda$)**, где $\lambda \in [0,1]$:

* $\lambda = 0$ → TD(0)
* $\lambda = 1$ → Monte Carlo
* Промежуточные значения $\lambda$ дают баланс между смещением и дисперсией.

(Реализуется через **следы посещений** — eligibility traces.)

---

## 6. Визуальная аналогия

**Monte Carlo:** «Я подожду, пока всё закончится, и потом подведу итоги».

**TD:** «Я уже понял тенденцию и корректируюсь на каждом шаге».

---

## 7. Обновления в коде (интуитивно)

**Примечание:** Код использует Gymnasium API (версия ≥ 0.26.0). Если вы используете старый Gym (<0.26), замените:
- `state, info = env.reset()` → `state = env.reset()`
- `next_state, reward, terminated, truncated, info = env.step(action)` → `next_state, reward, done, _ = env.step(action)`

```python
# Monte Carlo (по окончании эпизода)
# Сначала собираем траекторию
trajectory = []  # List[(state, reward)]
state, info = env.reset()
done = False

while not done:
    action = choose_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    trajectory.append((state, reward))
    state = next_state
    done = terminated or truncated

# Затем обновляем V для каждого посещённого состояния
G = 0
visited_states = set()
for state, reward in reversed(trajectory):
    G = reward + gamma * G
    # First-visit MC: обновляем только при первом посещении
    state_key = tuple(state) if isinstance(state, np.ndarray) else state
    if state_key not in visited_states:
        V[state_key] = V.get(state_key, 0.0) + alpha * (G - V.get(state_key, 0.0))
        visited_states.add(state_key)

# Temporal Difference (во время эпизода)
state, info = env.reset()
for t in range(max_steps):
    action = choose_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Конвертируем состояния в hashable keys
    state_key = tuple(state) if isinstance(state, np.ndarray) else state
    next_state_key = tuple(next_state) if isinstance(next_state, np.ndarray) else next_state
    
    # TD(0) update
    V[state_key] = V.get(state_key, 0.0) + alpha * (
        reward + gamma * V.get(next_state_key, 0.0) - V.get(state_key, 0.0)
    )
    
    state = next_state
    if done:
        break
```

---

## 8. Вывод

Оба метода учат **ценность состояний $$V(s)$$**, но по-разному используют опыт. TD стал основой многих алгоритмов:

* **SARSA** (on‑policy TD)
* **Q‑Learning** (off‑policy TD)
* **Expected SARSA**
* **TD($$\lambda$$)**
* **DQN** (Deep Q‑Network)
* **A3C/A2C**

---

## 9. Сравнение формул (шпаргалка)

**Monte Carlo:**
$$
\boxed{ V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)] }
$$

**Temporal Difference:**
$$
\boxed{ V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)] }
$$

Где $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ — TD‑ошибка.

---

**Основано на:**

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2020)
* Hugging Face Deep RL Course, Unit 2
* Andrea Lonza, *Алгоритмы обучения с подкреплением на Python* (2020)
