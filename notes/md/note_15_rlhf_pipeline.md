# Теоретический конспект №15

## Тема: RLHF — Reinforcement Learning from Human Feedback

> **Связано с:** [note_14_ppo_trpo.md](note_14_ppo_trpo.md) — PPO · [note_11_policy_gradients_reinforce.md](note_11_policy_gradients_reinforce.md) — Policy Gradient

---

## 1. Мотивация: выравнивание LLM с человеческими ценностями

**Проблема:** Large Language Models (LLM), обученные на огромных текстовых корпусах, могут:

- Генерировать токсичный контент
- Давать вредные советы
- Не следовать инструкциям пользователя
- "Галлюцинировать" (выдумывать факты)

**Почему Supervised Fine-Tuning (SFT) недостаточно?**

- Дорого собирать тысячи примеров ответов вручную
- Сложно покрыть все возможные сценарии
- "Правильный" ответ субъективен (tone, style, verbosity)

**Решение: RLHF** — обучить модель на **предпочтениях** людей, а не на абсолютных ответах.

---

## 2. История RLHF

| Год | Модель | Вклад |
|-----|--------|-------|
| 2017 | Deep RL from Human Preferences | Первая работа по RLHF для Atari |
| 2020 | GPT-3 | Показал потенциал LLM, но без alignment |
| 2022 | **InstructGPT** | Первая масштабная RLHF для языка → ChatGPT |
| 2023 | GPT-4, Claude, Llama 2 | RLHF становится стандартом |

**Ключевая insight:** Легче для человека **сравнивать** два ответа, чем писать идеальный ответ.

---

## 3. Трёхэтапный пайплайн RLHF

```
Pretrained LLM
      ↓
[1] Supervised Fine-Tuning (SFT)
      ↓
[2] Reward Model Training (RM)
      ↓
[3] RL Fine-Tuning (PPO)
      ↓
Aligned LLM
```

---

### Этап 1: Supervised Fine-Tuning (SFT)

**Цель:** Адаптировать pretrained LLM к формату "инструкция → ответ".

**Данные:**
- Пары $(x, y)$: prompt + high-quality response
- Обычно 10K-50K примеров (собраны людьми)

**Обучение:** Стандартный language modeling loss

$$
L_{SFT}(\theta) = -\sum_{t=1}^T \log p_\theta(y_t \mid y_{<t}, x)
$$

**Пример:**

```
x (prompt): "Explain quantum computing to a 5-year-old"
y (response): "Imagine you have a magic coin..."
```

**Результат:** $\pi_{SFT}$ — модель, умеющая следовать инструкциям, но несовершенная.

---

### Этап 2: Reward Model Training (RM)

**Цель:** Обучить модель, предсказывающую "качество" ответа с точки зрения человека.

**Данные: предпочтения (preferences)**

Для одного промпта $x$, собираем 4-9 ответов от $\pi_{SFT}$, человек ранжирует их:

$$
(x, y_w, y_l) \quad \text{где } y_w \succ y_l
$$

- $y_w$ — "winning" response (предпочтительнее)
- $y_l$ — "losing" response

Обычно 50K-100K таких пар.

**Reward Model:** Скалярная функция $r_\phi(x, y) \in \mathbb{R}$

Инициализируется от $\pi_{SFT}$, последний слой заменяется на линейную голову для скаляра.

**Loss: Bradley-Terry Model**

Вероятность того, что $y_w$ предпочтительнее $y_l$:

$$
P(y_w \succ y_l \mid x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
$$

где $\sigma$ — сигмоида.

**Loss:**

$$
L_{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

**Интуиция:** Maximize разницу в rewards между хорошими и плохими ответами.

---

### Этап 3: RL Fine-Tuning (PPO)

**Цель:** Оптимизировать политику $\pi_\theta$ для максимизации reward от RM, **НО** не уходить далеко от $\pi_{SFT}$.

**Целевая функция:**

$$
\max_\theta \mathbb{E}_{x \sim D_{prompts}, y \sim \pi_\theta(y|x)} \left[ r_\phi(x, y) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{SFT}) \right]
$$

где:
- $r_\phi(x, y)$ — награда от RM
- $\beta$ — коэффициент KL-штрафа (типично 0.01-0.1)
- $D_{KL}$ — KL-дивергенция от reference policy $\pi_{SFT}$

**KL-штраф:**

$$
D_{KL}(\pi_\theta \| \pi_{SFT}) = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y|x)}{\pi_{SFT}(y|x)} \right]
$$

**Зачем KL-штраф?**

1. **Предотвращает reward hacking** (модель находит лазейки в RM)
2. **Сохраняет языковые способности** (не забывает грамматику, факты)
3. **Стабилизирует обучение**

---

### PPO для LLM

Используем **PPO** (см. [note_14_ppo_trpo.md](note_14_ppo_trpo.md)) с модификациями:

**Алгоритм:**

```
for iteration in range(N):
    # 1. Sample prompts
    prompts = sample_batch(D_prompts)
    
    # 2. Generate responses с π_θ
    responses = π_θ.generate(prompts)
    
    # 3. Compute rewards
    rewards = r_φ(prompts, responses)
    
    # 4. Compute KL penalty
    kl_penalty = KL(π_θ || π_SFT)
    
    # 5. Compute advantages через GAE
    advantages = compute_gae(rewards - β * kl_penalty, V_critic)
    
    # 6. PPO update (multiple epochs)
    for epoch in range(K):
        for batch in minibatches:
            # PPO-Clip loss
            policy_loss = ppo_clip_loss(batch, advantages)
            
            # Value loss
            value_loss = (V(s) - returns)^2
            
            # Update
            loss = policy_loss + c1 * value_loss
            optimizer.step()
```

**Отличия от обычного PPO:**

- **4 модели в памяти одновременно:**
  1. Actor $\pi_\theta$ (обучаем)
  2. Critic $V_\psi$ (обучаем)
  3. Reference $\pi_{SFT}$ (frozen)
  4. Reward Model $r_\phi$ (frozen)
  
- **Огромные вычислительные требования:** GPT-4 обучался на тысячах GPU

---

## 4. KL-штраф: критически важный компонент

### Без KL-штрафа: катастрофа

```
Iteration 1: reward = 5.0, text = "Great answer!"
Iteration 10: reward = 8.0, text = "!!!!!!!!!!!!!!!!!!!"
Iteration 50: reward = 15.0, text = "kdfj;alskdjf;laksdjf"
```

**Что происходит:** Модель находит **adversarial примеры**, которые получают высокую reward от RM, но бессмысленны.

### С KL-штрафом: стабильность

$$
\text{Total Reward} = r_\phi(x, y) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{SFT})
$$

Если модель уходит далеко от $\pi_{SFT}$ → большой KL-штраф → низкая total reward.

**Типичные значения $\beta$:**

| $\beta$ | Эффект |
|---------|--------|
| 0.001 | Слабый контроль, риск reward hacking |
| 0.01 | Хороший баланс (стандарт) |
| 0.1 | Сильный контроль, медленное улучшение |

---

## 5. Проблемы и челленджи RLHF

### 1. Reward Hacking

**Проблема:** RM обучена на ограниченном датасете → модель находит OOD примеры с высокой reward.

**Пример:**
- RM научилась, что длинные ответы лучше
- Модель генерирует многословные, но бесполезные ответы

**Решение:**
- KL-штраф (основной механизм)
- Adversarial testing
- Итеративное обновление RM

---

### 2. Distribution Shift

**Проблема:** RM обучена на ответах $\pi_{SFT}$, но оценивает ответы $\pi_\theta$.

При $\pi_\theta \neq \pi_{SFT}$ → RM экстраполирует → ненадёжные rewards.

**Решение:**
- KL-штраф ограничивает shift
- Iterative RLHF (периодически обновлять RM)

---

### 3. Reward Over-Optimization

**Наблюдение:** При слишком долгом обучении качество **ухудшается**, несмотря на рост reward!

```
RM Reward: ↑↑↑↑↑↑
Human Eval: ↑↑↑↓↓↓ (Goodhart's Law)
```

**Goodhart's Law:** "When a measure becomes a target, it ceases to be a good measure."

**Решение:**
- Early stopping по human evaluation
- Регулярный мониторинг метрик

---

### 4. Вычислительная стоимость

**Требования:**

- 4 модели в памяти (Actor, Critic, Reference, RM)
- Для 7B модели: ~40-60GB VRAM
- Для GPT-3.5/4: тысячи GPU, недели обучения

**Решение:**
- LoRA, QLoRA для параметр-эффективного fine-tuning
- Distillation
- DPO (см. [note_16_dpo_and_variants.md](note_16_dpo_and_variants.md)) — более эффективная альтернатива

---

## 6. Метрики оценки RLHF

| Метрика | Описание | Как измерить |
|---------|----------|--------------|
| **RM Reward** | Reward от trained RM | Автоматически |
| **KL Divergence** | Расстояние от $\pi_{SFT}$ | $D_{KL}(\pi_\theta \| \pi_{SFT})$ |
| **Human Eval** | Ручная оценка качества | Win rate против baseline |
| **Perplexity** | Сохранение языковых способностей | На held-out data |
| **Safety Metrics** | Toxicity, bias, harmful content | Perspective API, Red teaming |
| **Task Performance** | Точность на downstream tasks | Benchmarks (MMLU, HumanEval) |

**Ключевой trade-off:**

$$
\text{RM Reward} \uparrow \quad \leftrightarrow \quad \text{KL Divergence} \uparrow
$$

Нужно найти баланс через подбор $\beta$.

---

## 7. Варианты и расширения RLHF

### Constitutional AI (Anthropic)

- Модель самостоятельно критикует и улучшает свои ответы
- Меньше зависимости от человеческой аннотации

### RLAIF (RL from AI Feedback)

- Используем сильную LLM (GPT-4) вместо людей для генерации предпочтений
- Дешевле и быстрее масштабируется

### Iterative RLHF

```
Round 1: SFT → RM1 → PPO1 → Model v1
Round 2: Collect new preferences on Model v1 → RM2 → PPO2 → Model v2
...
```

### Multi-objective RLHF

$$
\max_\theta \mathbb{E} \left[ \alpha_1 r_{\text{helpful}} + \alpha_2 r_{\text{harmless}} + \alpha_3 r_{\text{honest}} - \beta D_{KL} \right]
$$

Обучаем несколько reward models для разных целей.

---

## 8. Practical Tips для RLHF

### 1. Quality > Quantity для предпочтений

- 10K высококачественных пар лучше 100K шумных
- Важна согласованность аннотаторов (agreement rate > 70%)

### 2. Balanced Dataset

- Покрыть разные типы промптов (вопросы, инструкции, creative writing)
- Избегать сильных biases в предпочтениях

### 3. KL Warm-up

```python
# Постепенно увеличиваем β
β = β_min + (β_max - β_min) * min(1.0, iteration / warmup_steps)
```

### 4. Мониторинг KL

Если $D_{KL} > 10$: модель слишком далеко ушла → увеличить $\beta$ или early stop.

### 5. Length Normalization

$$
r_{\text{normalized}} = \frac{r_\phi(x, y)}{\text{length}(y)}
$$

Предотвращает bias к длинным ответам.

---

## 9. RLHF в индустрии

### OpenAI: InstructGPT → ChatGPT → GPT-4

- **InstructGPT (2022):** Первая масштабная RLHF для GPT-3
- **ChatGPT (Nov 2022):** Применение RLHF для диалоговой модели
- **GPT-4 (Mar 2023):** Масштабирование RLHF на multimodal модель

**Ключевые insights:**
- RLHF критичен для "человекоподобного" поведения
- KL-штраф к SFT абсолютно необходим

### Anthropic: Claude

- Constitutional AI + RLHF
- Акцент на harmlessness и honesty

### Meta: Llama 2-Chat

- Open-source модель с RLHF
- Детальное описание процесса в paper

---

## 10. Резюме

| Концепция | Описание |
|-----------|----------|
| **RLHF Pipeline** | SFT → RM → PPO в три этапа |
| **Reward Model** | Обучается на парных предпочтениях (Bradley-Terry) |
| **KL Constraint** | $\beta D_{KL}(\pi_\theta \| \pi_{SFT})$ предотвращает reward hacking |
| **PPO для LLM** | 4 модели в памяти, огромные вычисления |
| **Reward Hacking** | Основная проблема, решается KL-штрафом |
| **Goodhart's Law** | Over-optimization RM приводит к ухудшению качества |

**Ключевые выводы:**

1. **RLHF = индустриальный стандарт** для alignment LLM
2. **KL-штраф критически важен** для стабильности
3. **Вычислительно дорого**, но необходимо для качества
4. **Альтернативы (DPO)** активно развиваются

---

## 11. Связь с предыдущими семинарами

- **[note_14_ppo_trpo.md](note_14_ppo_trpo.md):** PPO — RL алгоритм для RLHF
- **[note_11_policy_gradients_reinforce.md](note_11_policy_gradients_reinforce.md):** Policy Gradient — основа PPO
- **[note_12_actor_critic_a2c.md](note_12_actor_critic_a2c.md):** Actor-Critic — архитектура для LLM RL

---

## 12. Дальнейшее изучение

**Ключевые статьи:**

- *Training language models to follow instructions with human feedback* (InstructGPT, 2022)
- *Constitutional AI: Harmlessness from AI Feedback* (Anthropic, 2022)
- *Llama 2: Open Foundation and Fine-Tuned Chat Models* (Meta, 2023)

**Библиотеки:**

- [Hugging Face TRL](https://github.com/huggingface/trl) — RLHF для transformers
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeed) — Efficient RLHF

---

## 13. Практическое задание

В `code/15_rlhf_basics/` реализован упрощённый RLHF pipeline на игрушечной задаче.

**Эксперименты:**
- Влияние $\beta$ на качество и KL
- Reward hacking без KL-штрафа
- Сравнение SFT baseline vs RLHF

---

**Далее:** [note_16_dpo_and_variants.md](note_16_dpo_and_variants.md) — DPO: Direct Preference Optimization

