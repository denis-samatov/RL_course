# Теоретический конспект №17

## Тема: Финальный проект — Итоги курса и направления развития

> **Итоговый семинар:** Интеграция всех изученных компонентов

---

## 1. Что мы изучили: полный путь от MDP до RLHF

### Фундаментальные концепции (Семинары 1-4)

| Семинар | Тема | Ключевые концепции |
|---------|------|-------------------|
| 1 | Введение в RL | Agent-Environment, Reward, MDP |
| 2 | MDP Framework | Состояния, действия, переходы, Bellman |
| 3 | Exploration vs Exploitation | ε-greedy, Softmax, UCB |
| 4 | Policy vs Value-Based | Два подхода к решению RL |

### Классические методы (Семинары 5-10)

| Семинар | Тема | Алгоритмы |
|---------|------|-----------|
| 5 | Deep RL | Нейросетевая аппроксимация |
| 6 | Value Functions | V(s) и Q(s,a) |
| 7 | Bellman Equation | Рекуррентные обновления |
| 8 | MC vs TD | Два способа оценки V |
| 9 | Q-Learning | Off-policy TD control |
| 10 | DQN | Deep Q-Network + расширения |

### Policy Gradient (Семинары 11-12)

| Семинар | Тема | Методы |
|---------|------|--------|
| 11 | Policy Gradients | REINFORCE, Baseline, Entropy |
| 12 | Actor-Critic | A2C, GAE, Continuous Control |

### Advanced RL (Семинары 13-14)

| Семинар | Тема | Методы |
|---------|------|--------|
| 13 | Dynamic Programming | Policy/Value Iteration, GPI |
| 14 | PPO и TRPO | Trust regions, Clipping |

### RLHF (Семинары 15-16)

| Семинар | Тема | Методы |
|---------|------|--------|
| 15 | RLHF | SFT → RM → PPO pipeline |
| 16 | DPO | Direct Preference Optimization |

---

## 2. Финальный проект: Mini-RLHF Pipeline

### Цель проекта

Построить **end-to-end RLHF систему** на упрощённой задаче:

1. Pretrained base model
2. Supervised Fine-Tuning на демонстрациях
3. Reward Model на предпочтениях
4. RL fine-tuning (PPO или DPO)
5. Комплексная оценка

### Компоненты проекта

```
final_project/
├── data/
│   ├── base_prompts.json        # Промпты для обучения
│   ├── sft_demonstrations.json  # Качественные ответы
│   ├── preferences.json         # Парные предпочтения
│   └── test_set.json            # Evaluation prompts
├── models/
│   ├── base_model/              # Исходная модель
│   ├── sft_model/               # После SFT
│   ├── reward_model/            # Обученная RM
│   ├── ppo_model/               # После PPO-RLHF
│   └── dpo_model/               # После DPO
├── src/
│   ├── train_sft.py
│   ├── train_rm.py
│   ├── train_ppo.py
│   ├── train_dpo.py
│   └── evaluation/
│       ├── metrics.py           # Reward, KL, Perplexity
│       ├── safety_checks.py     # Toxicity, Bias
│       └── compare_models.py    # A/B testing
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sft_training.ipynb
│   ├── 03_rm_training.ipynb
│   ├── 04_rlhf_comparison.ipynb
│   └── 05_final_analysis.ipynb
└── README.md
```

---

## 3. Метрики оценки

### Автоматические метрики

| Метрика | Описание | Код |
|---------|----------|-----|
| **RM Reward** | Средняя награда от RM | `r_φ(x, y).mean()` |
| **KL Divergence** | Расстояние от SFT | `KL(π_θ \|\| π_SFT)` |
| **Perplexity** | Языковое качество | `exp(-log_prob.mean())` |
| **Length** | Средняя длина ответов | `len(tokens).mean()` |
| **Diversity** | Уникальные n-граммы | `unique_ngrams / total_ngrams` |

### Ручная оценка

**Win Rate:** Для каждого промпта, человек выбирает лучший ответ:

$$
\text{Win Rate} = \frac{\# \text{times model wins}}{\# \text{total comparisons}}
$$

**Типичные сравнения:**
- SFT vs PPO-RLHF
- SFT vs DPO
- PPO vs DPO

---

### Safety метрики

| Метрика | Инструмент | Threshold |
|---------|-----------|-----------|
| **Toxicity** | Perspective API | < 0.1 |
| **Bias** | Gender/Race templates | Balanced |
| **Refusal Rate** | На harmful prompts | > 90% |
| **Hallucination** | Fact-checking | < 10% |

---

## 4. Безопасность в RL

### Типы проблем

1. **Reward Hacking**
   - Модель эксплуатирует недостатки RM
   - Пример: Повторяет слова для увеличения длины

2. **Goal Misgeneralization**
   - Модель обобщает не те паттерны
   - Пример: Учится копировать стиль, а не смысл

3. **Distributional Shift**
   - Деградация на OOD промптах
   - Пример: Хорошо на вежливых запросах, плохо на грубых

4. **Value Alignment**
   - Модель не разделяет человеческие ценности
   - Пример: Помогает в опасных действиях

---

### Методы предотвращения

| Метод | Описание | Эффективность |
|-------|----------|---------------|
| **KL Penalty** | $\beta D_{KL}(\pi_\theta \|\| \pi_{SFT})$ | ✅✅✅ Критически важен |
| **Adversarial Testing** | Red teaming на harmful prompts | ✅✅ Очень полезен |
| **Constitutional AI** | Модель самокритикует ответы | ✅✅ Помогает, но дорого |
| **Human-in-the-Loop** | Регулярная проверка людьми | ✅✅✅ Необходим |
| **Multiple RMs** | Разные reward models для разных целей | ✅ Улучшает robustness |
| **Filtered Training Data** | Удаление токсичных примеров | ✅ Baseline |

---

## 5. Этика и социальные аспекты

### Bias в данных

**Проблема:** Датасеты отражают biases общества (gender, race, etc.)

**Пример:**
```
Prompt: "The doctor said..."
Biased: "...he will see you now"
Balanced: "...they will see you now"
```

**Решения:**
- Balanced аннотаторы (diversity)
- Explicit de-biasing в SFT данных
- Monitoring bias метрик

---

### Представительность аннотаторов

**Проблема:** Если все аннотаторы из одной демографии → preferences не универсальны.

**Решение:**
- Нанимать аннотаторов из разных стран/культур
- Учитывать разногласия (disagreement)
- Multiple reward models для разных аудиторий

---

### Long-term эффекты

**Вопросы:**
- Как RLHF влияет на восприятие AI обществом?
- Делает ли это модели "too aligned" (sycophantic)?
- Централизация контроля над alignment?

---

## 6. Направления дальнейшего изучения

### Multi-Agent RL

- Agents взаимодействуют друг с другом
- Nash equilibrium, coordination
- Applications: игры, переговоры, multi-robot systems

### Model-Based RL

- Обучение модели среды $P(s'|s,a)$
- Planning с learned model (Dreamer, MuZero)
- Sample efficiency

### Offline RL

- Обучение на фиксированном датасете (без взаимодействия)
- Conservative Q-Learning, IQL
- Applications: медицина, финансы

### Meta-RL и Few-Shot RL

- Обучение "учиться учиться"
- Быстрая адаптация к новым задачам
- MAML, RL²

### Hierarchical RL

- Decomposition на sub-tasks
- Options, HAM
- Long-horizon tasks

### Safe RL и Constrained RL

- Explicit constraints (безопасность, fairness)
- CPO (Constrained Policy Optimization)
- Critical для real-world deployment

### RL для робототехники

- Sim-to-real transfer
- Imitation learning + RL
- Continuous control

### Advanced RLHF

- Constitutional AI (Anthropic)
- Debate (OpenAI)
- Recursive Reward Modeling
- Scalable Oversight

---

## 7. Итоговая таблица алгоритмов

| Алгоритм | Тип | On/Off Policy | Continuous | Sample Efficiency | Стабильность | Когда использовать |
|----------|-----|---------------|------------|-------------------|--------------|-------------------|
| **Q-Learning** | Value-based | Off | ❌ | Низкая | Средняя | Дискретные, малые |
| **DQN** | Value-based | Off | ❌ | Средняя | Средняя | Дискретные, большие |
| **REINFORCE** | Policy | On | ✅ | Низкая | Низкая | Простые задачи |
| **A2C** | Actor-Critic | On | ✅ | Средняя | Средняя | Общего назначения |
| **PPO** | Actor-Critic | On | ✅ | Высокая | Высокая | State-of-the-art |
| **SAC** | Actor-Critic | Off | ✅ | Очень высокая | Высокая | Робототехника |
| **TD3** | Actor-Critic | Off | ✅ | Очень высокая | Высокая | Continuous control |

---

## 8. Практические рекомендации для real-world RL

### 1. Начните с простого

- Табличный Q-Learning для понимания
- DQN для дискретных действий
- PPO для continuous control

### 2. Tune гиперпараметры

**Критичные:**
- Learning rate (обычно 1e-4 до 3e-4)
- Discount γ (0.99 для episodic, 0.95 для continuing)
- Exploration (ε decay, entropy bonus)

### 3. Мониторьте метрики

- Episode return (должен расти)
- Episode length (может меняться)
- Loss (должен стабилизироваться)
- Gradient norm (не должен взрываться)

### 4. Используйте проверенные библиотеки

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — PyTorch, PPO/DQN/SAC
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — Минималистичные имплементации
- [RLlib](https://docs.ray.io/en/latest/rllib/index.html) — Масштабируемый RL
- [Hugging Face TRL](https://github.com/huggingface/trl) — RLHF для LLM

### 5. Воспроизводимость

```python
# Seed everything
np.random.seed(42)
torch.manual_seed(42)
env.reset(seed=42)
env.action_space.seed(42)
```

---

## 9. Чему мы научились (Learning Outcomes)

После прохождения курса вы можете:

✅ Формализовать задачу как MDP  
✅ Выбрать подходящий алгоритм для задачи  
✅ Реализовать DQN, REINFORCE, A2C, PPO с нуля  
✅ Понимать trade-offs: sample efficiency vs стабильность  
✅ Применять RLHF для alignment LLM  
✅ Оценивать качество и безопасность RL-систем  
✅ Дебажить и тюнить RL алгоритмы  
✅ Читать research papers и имплементировать новые методы  

---

## 10. Финальный чеклист проекта

### Технические компоненты

- [ ] Реализован SFT pipeline
- [ ] Обучена Reward Model на предпочтениях
- [ ] Реализован PPO с KL-штрафом
- [ ] (Опционально) Реализован DPO
- [ ] Все метрики логируются (RM reward, KL, perplexity)

### Эксперименты

- [ ] Ablation: влияние β (KL-penalty)
- [ ] Сравнение: SFT vs PPO vs DPO
- [ ] Safety testing: adversarial prompts
- [ ] Human evaluation: win rate

### Документация

- [ ] README с инструкциями
- [ ] Описание датасетов
- [ ] Графики обучения
- [ ] Примеры лучших/худших ответов
- [ ] Анализ ограничений

---

## 11. Резюме курса

**Пройденный путь:**

```
MDP → Bellman → Q-Learning → DQN → Policy Gradient → 
Actor-Critic → PPO → RLHF → DPO
```

**Ключевые уроки:**

1. **RL = итеративное улучшение** через trial and error
2. **Trade-offs везде:** bias-variance, exploration-exploitation, sample efficiency-stability
3. **Инженерия важна:** правильные tricks (GAE, clipping, normalization) критичны
4. **Безопасность = приоритет:** KL-penalty, testing, monitoring

---

## 12. Заключение

Reinforcement Learning — это **мощный инструмент** для решения задач sequential decision making.

От **игр** (AlphaGo, Dota 2) до **робототехники** и **языковых моделей** (ChatGPT) — RL продолжает трансформировать AI.

**Следующие шаги:**

1. Реализуйте финальный проект
2. Изучите advanced topics (multi-agent, model-based)
3. Читайте новые papers (arxiv.org/list/cs.LG/recent)
4. Участвуйте в competitions (Kaggle RL, NeurIPS competitions)
5. Применяйте в реальных задачах

---

**🎓 Поздравляем с завершением курса!**

**Автор:** Denis Samatov, TPU / 2025  
**Контакт:** [GitHub](https://github.com/denissamatov) · [Email](mailto:denissamatov@example.com)

---

## 📚 Полный список рекомендуемой литературы

### Книги

1. **Sutton & Barto** — *Reinforcement Learning: An Introduction (2nd ed.)*
2. **Maxim Lapan** — *Deep Reinforcement Learning Hands-On*
3. **Andrea Lonza** — *Reinforcement Learning Algorithms with Python*
4. **Csaba Szepesvári** — *Algorithms for Reinforcement Learning*

### Онлайн-курсы

1. **David Silver's RL Course** (DeepMind/UCL)
2. **CS285: Deep RL** (UC Berkeley, Sergey Levine)
3. **OpenAI Spinning Up** (Practical RL guide)

### Research Papers (Must-Read)

1. DQN: *Playing Atari with Deep RL* (Mnih et al., 2013)
2. PPO: *Proximal Policy Optimization* (Schulman et al., 2017)
3. RLHF: *InstructGPT* (Ouyang et al., 2022)
4. DPO: *Direct Preference Optimization* (Rafailov et al., 2023)

---

✅ **Курс завершён!** Желаем успехов в применении Reinforcement Learning! 🚀

