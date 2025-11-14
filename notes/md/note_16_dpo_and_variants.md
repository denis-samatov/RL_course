# Теоретический конспект №16

## Тема: DPO — Direct Preference Optimization

> **Связано с:** [note_15_rlhf_pipeline.md](note_15_rlhf_pipeline.md) — RLHF Pipeline

---

## 1. Мотивация: упрощение RLHF

**Проблемы PPO-RLHF:**

1. **Сложность:** 4 модели (Actor, Critic, Reference, RM)
2. **Нестабильность:** Требует тщательной настройки гиперпараметров
3. **Вычислительная стоимость:** Огромные ресурсы
4. **Reward hacking:** RM может быть обманута

**Идея DPO:** Можно ли обучаться **напрямую на предпочтениях**, минуя Reward Model и RL?

---

## 2. Теоретическое обоснование DPO

### Оптимальная политика в RLHF

В RLHF мы решаем:

$$
\max_\pi \mathbb{E}_{x, y \sim \pi} \left[ r(x, y) - \beta D_{KL}(\pi \| \pi_{ref}) \right]
$$

**Теорема (Rafailov et al., 2023):** Оптимальная политика имеет **closed-form решение**:

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta}r(x,y)\right)
$$

где $Z(x)$ — partition function.

### Решая для reward:

$$
r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

### Подставляя в Bradley-Terry:

$$
P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
$$

$$
= \sigma\left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} \right)
$$

**Ключевое наблюдение:** $Z(x)$ **сократилось**! Можем оптимизировать политику напрямую.

---

## 3. DPO Loss

**Итоговая функция потерь:**

$$
L_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

**Упрощённо:**

$$
L_{DPO} = -\mathbb{E} \left[ \log \sigma \left( \beta \left[ \log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x) \right] - \beta \left[ \log \pi_{ref}(y_w|x) - \log \pi_{ref}(y_l|x) \right] \right) \right]
$$

**Интуиция:**
- Увеличиваем $\log \pi_\theta(y_w|x)$ (хороший ответ)
- Уменьшаем $\log \pi_\theta(y_l|x)$ (плохой ответ)
- Контролируемся через $\pi_{ref}$ (неявный KL-штраф!)

---

## 4. DPO vs PPO-RLHF

| Аспект | PPO-RLHF | DPO |
|--------|----------|-----|
| **Этапы** | 3 (SFT → RM → PPO) | 2 (SFT → DPO) |
| **Модели в памяти** | 4 (Actor, Critic, RM, Ref) | 2 (Policy, Reference) |
| **Reward Model** | Нужна (отдельное обучение) | Не нужна |
| **RL алгоритм** | PPO (сложный) | Нет RL (просто supervised) |
| **Сложность реализации** | Высокая | Низкая |
| **Стабильность** | Средняя (нужна настройка) | Высокая |
| **Sample Efficiency** | Средняя | Высокая |
| **Гибкость** | Высокая (можно менять reward) | Низкая (фиксированный датасет) |
| **Online сбор данных** | ✅ Да | ❌ Нет |

---

## 5. Преимущества DPO

### 1. Простота
- Обучается как supervised classification
- Нет сложного RL-цикла
- Меньше гиперпараметров

### 2. Стабильность
- Нет reward hacking (нет отдельной RM)
- Неявный KL-штраф встроен в формулу
- Не нужна тонкая настройка PPO

### 3. Эффективность
- Меньше памяти (2 модели вместо 4)
- Быстрее обучается (одна эпоха на preference данных)
- Sample efficient

### 4. Интерпретируемость
- Прямая связь с предпочтениями
- Понятная функция потерь

---

## 6. Недостатки DPO

### 1. Offline Only
- Требует фиксированный датасет предпочтений
- Нельзя собирать новые данные онлайн
- Ограничен качеством SFT-модели

### 2. Менее гибкий
- Нельзя комбинировать несколько reward functions
- Сложно добавить constraints (safety, length)

### 3. Зависимость от Reference Policy
- Качество зависит от $\pi_{ref}$ (обычно SFT)
- Если SFT плохая → DPO тоже плохая

---

## 7. Алгоритм DPO

```python
# 1. Инициализация
π_θ = copy(π_SFT)  # Обучаемая политика
π_ref = π_SFT      # Reference (frozen)
β = 0.1            # Temperature

# 2. Обучение
for epoch in range(N):
    for (x, y_w, y_l) in preference_dataset:
        # Вычисляем log-probs
        log_prob_w = π_θ.log_prob(y_w | x)
        log_prob_l = π_θ.log_prob(y_l | x)
        
        # Reference log-probs (frozen)
        with torch.no_grad():
            log_prob_w_ref = π_ref.log_prob(y_w | x)
            log_prob_l_ref = π_ref.log_prob(y_l | x)
        
        # DPO loss
        logits_w = β * (log_prob_w - log_prob_w_ref)
        logits_l = β * (log_prob_l - log_prob_l_ref)
        
        loss = -log_sigmoid(logits_w - logits_l)
        
        # Update (ВАЖНО: сбрасываем градиенты перед backward!)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 8. Варианты DPO

### IPO (Identity Preference Optimization)

Использует MSE вместо log-sigmoid:

$$
L_{IPO} = \mathbb{E} \left[ \left( \frac{1}{2\beta} \left( r_w - r_l \right) - 1 \right)^2 \right]
$$

**Преимущество:** Меньше чувствителен к outliers.

### KTO (Kahneman-Tversky Optimization)

Учитывает **асимметрию** человеческих предпочтений (loss aversion):

$$
L_{KTO} = \lambda_{pos} \mathbb{E}[loss_{pos}] + \lambda_{neg} \mathbb{E}[loss_{neg}]
$$

где $\lambda_{pos} < \lambda_{neg}$ (люди сильнее реагируют на плохое).

### ORPO (Odds Ratio Preference Optimization)

**Объединяет SFT и DPO в один этап:**

$$
L_{ORPO} = L_{SFT} + \lambda L_{DPO}
$$

**Преимущество:** Экономит один этап обучения.

---

## 9. Когда использовать DPO vs PPO

### Используйте DPO если:

✅ Есть фиксированный датасет предпочтений  
✅ Нужна простота и стабильность  
✅ Ограниченные вычислительные ресурсы  
✅ Не требуется online сбор данных  

### Используйте PPO если:

✅ Нужна гибкость в reward (multiple objectives)  
✅ Можно собирать данные онлайн  
✅ Сложная composable reward function  
✅ Есть ресурсы для настройки и отладки  

---

## 10. Практические рекомендации DPO

### 1. Quality Preference Data

- Предпочтения должны быть **clear и consistent**
- Agreement rate между аннотаторами > 70%
- Сбалансированный coverage разных типов промптов

### 2. Выбор β

| β | Эффект |
|---|--------|
| 0.01 | Слабый контроль, сильное обновление |
| 0.1 | Baseline (рекомендуется) |
| 0.5 | Сильный контроль, консервативное обновление |

### 3. Learning Rate

- Меньше, чем для SFT: `1e-6` to `5e-6`
- Linear warmup первые 10% шагов

### 4. Batch Size

- Больше = стабильнее: 32-128 пар
- Gradient accumulation если не хватает памяти

---

## 11. Результаты в индустрии

### Zephyr-7B (HuggingFace)

- DPO на 60K preference pairs
- Превосходит Llama-2-70B-Chat на MT-Bench
- Обучен за **несколько часов** на 8× A100

### Mistral-7B-Instruct

- Использует DPO для alignment
- State-of-the-art для 7B моделей

### Intel Neural Chat

- DPO + distillation
- Efficient alignment для edge deployment

---

## 12. Резюме

| Концепция | Описание |
|-----------|----------|
| **DPO Loss** | Прямая оптимизация на предпочтениях без RM |
| **Closed-form Solution** | Оптимальная политика выражается через reward |
| **Implicit KL** | Контроль через reference policy встроен |
| **Supervised Learning** | Обучается как classification, не RL |
| **2 Этапа** | SFT → DPO (проще чем SFT → RM → PPO) |

**Ключевые выводы:**

1. **DPO = более простая альтернатива RLHF**
2. **Теоретически обоснован** через closed-form решение
3. **Практически эффективен** для offline preference optimization
4. **Trade-off:** Простота vs Гибкость

---

## 13. Практическое задание

В `code/16_dpo/` реализован DPO и сравнение с PPO на одном датасете.

**Эксперименты:**
- Влияние β
- Sample efficiency: DPO vs PPO
- Quality: Win rate против baseline

---

**Далее:** [note_17_final_project.md](note_17_final_project.md) — Финальный проект и безопасность

