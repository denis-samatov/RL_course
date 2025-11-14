# 🏆 Семинар 17: Финальный проект — Mini-RLHF Pipeline

> **Теория:** [note_17_final_project.md](../../notes/md/note_17_final_project.md)  
> **Цель:** Интеграция всех компонентов в полный RLHF пайплайн

---

## 📖 Обзор проекта

Построение **end-to-end системы** для alignment LLM:

1. **Supervised Fine-Tuning (SFT)** — адаптация к инструкциям
2. **Reward Model (RM)** — обучение на предпочтениях
3. **RL Fine-Tuning** — PPO или DPO для оптимизации
4. **Evaluation** — комплексная оценка качества и безопасности

---

## 🗂️ Структура проекта

```
final_project/
├── data/
│   ├── sft_demonstrations.json      # SFT данные
│   ├── preferences.json             # Парные предпочтения
│   └── test_prompts.json            # Evaluation
├── src/
│   ├── train_sft.py                 # Supervised Fine-Tuning
│   ├── train_rm.py                  # Reward Model
│   ├── train_ppo.py                 # PPO-RLHF
│   ├── train_dpo.py                 # DPO
│   └── evaluation/
│       ├── metrics.py               # Reward, KL, Perplexity
│       ├── safety_checks.py         # Toxicity, Bias
│       └── compare_models.py        # A/B testing
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sft_training.ipynb
│   ├── 03_rm_training.ipynb
│   ├── 04_rlhf_comparison.ipynb     # PPO vs DPO
│   └── 05_final_analysis.ipynb      # Итоговый отчёт
├── models/                           # (Создаётся при обучении)
│   ├── sft_model/
│   ├── reward_model/
│   ├── ppo_model/
│   └── dpo_model/
├── scripts/
│   ├── train_full_pipeline.sh       # Полный пайплайн
│   └── run_evaluation.sh            # Evaluation
└── README.md                         # Эта документация
```

---

## 🚀 Быстрый старт

### 1. Полный pipeline (автоматический)

```bash
bash scripts/train_full_pipeline.sh
```

**Этапы:**
1. SFT обучение (~10 мин)
2. RM обучение (~5 мин)
3. PPO fine-tuning (~30 мин)
4. DPO fine-tuning (~10 мин)
5. Evaluation (~ 5 мин)

**Total:** ~60 минут на GPU

---

### 2. Пошаговое выполнение

```bash
# Этап 1: SFT
python src/train_sft.py --data data/sft_demonstrations.json

# Этап 2: RM
python src/train_rm.py --data data/preferences.json --base models/sft_model

# Этап 3: PPO-RLHF
python src/train_ppo.py --rm models/reward_model --ref models/sft_model --beta 0.01

# Этап 4: DPO (альтернатива)
python src/train_dpo.py --data data/preferences.json --ref models/sft_model --beta 0.1

# Этап 5: Evaluation
python src/evaluation/compare_models.py --models models/sft_model models/ppo_model models/dpo_model
```

---

## 📊 Метрики оценки

### Автоматические метрики

```python
from evaluation.metrics import evaluate_model

metrics = evaluate_model(model, test_prompts)
print(f"RM Reward: {metrics['reward']:.2f}")
print(f"KL Divergence: {metrics['kl']:.3f}")
print(f"Perplexity: {metrics['perplexity']:.2f}")
print(f"Avg Length: {metrics['length']:.1f}")
```

### Ручная оценка (Win Rate)

```bash
python src/evaluation/human_eval.py --model_a models/sft_model --model_b models/ppo_model
```

**Результаты:**
- Model A wins: 35%
- Model B wins: 55%
- Tie: 10%

**Win Rate (B vs A):** 55%

---

## 🎯 Целевые метрики (Success Criteria)

| Метрика | Baseline (SFT) | Target (RLHF) | Достигнуто? |
|---------|----------------|---------------|-------------|
| RM Reward | 5.0 | > 7.0 | ✅ 7.8 |
| KL Divergence | 0.0 | < 5.0 | ✅ 2.3 |
| Perplexity | 12.5 | < 15.0 | ✅ 13.2 |
| Win Rate vs SFT | — | > 60% | ✅ 68% |
| Toxicity | 0.05 | < 0.10 | ✅ 0.07 |

---

## 🧪 Эксперименты

### Эксперимент 1: PPO vs DPO

```bash
python src/evaluation/compare_methods.py
```

**Результаты:**

| Method | Reward | KL | Training Time | Memory |
|--------|--------|----| -------------|--------|
| SFT | 5.0 | 0.0 | — | — |
| PPO | 7.8 | 2.3 | 30 min | 40GB |
| DPO | 7.5 | 1.8 | 10 min | 20GB |

**Вывод:** DPO достигает 96% качества PPO за 33% времени и 50% памяти.

---

### Эксперимент 2: Влияние β (KL penalty)

| β | RM Reward | KL | Quality (Human) |
|---|-----------|----| ---------------|
| 0.001 | 9.2 | 8.5 | Poor (reward hacking) |
| 0.01 | 7.8 | 2.3 | Good ✅ |
| 0.1 | 6.2 | 0.5 | Okay (too conservative) |

---

### Эксперимент 3: Safety Testing

```bash
python src/evaluation/safety_checks.py --model models/ppo_model
```

**Результаты:**

| Metric | Score | Pass? |
|--------|-------|-------|
| Toxicity (avg) | 0.07 | ✅ < 0.10 |
| Bias (gender) | 0.52 | ✅ ≈ 0.50 |
| Refusal Rate (harmful) | 94% | ✅ > 90% |
| Hallucination Rate | 8% | ✅ < 10% |

---

## 📚 Notebooks

### 01_data_exploration.ipynb

- Анализ SFT данных
- Распределение длин промптов/ответов
- Частые токены

### 02_sft_training.ipynb

- SFT loss кривые
- Примеры генераций до/после SFT

### 03_rm_training.ipynb

- RM accuracy на validation
- Примеры high/low reward ответов

### 04_rlhf_comparison.ipynb

- Сравнение PPO vs DPO
- Эволюция метрик (reward, KL, perplexity)
- A/B testing результаты

### 05_final_analysis.ipynb

- Итоговые метрики всех моделей
- Safety analysis
- Лучшие/худшие примеры
- Recommendations

---

## 💡 Ключевые выводы проекта

1. **RLHF значительно улучшает качество** (Win Rate 68% vs SFT)
2. **KL-penalty критически важен** для предотвращения reward hacking
3. **DPO — эффективная альтернатива PPO** (96% качества, 33% времени)
4. **Safety testing обязателен** перед deployment
5. **Human evaluation незаменим** для финальной оценки

---

## 🐛 Troubleshooting

### Проблема: Out of Memory

**Решение:**
- Уменьшите batch size
- Используйте gradient accumulation
- Попробуйте LoRA/QLoRA

### Проблема: Reward hacking

**Симптомы:** RM reward растёт, но human eval падает

**Решение:**
- Увеличьте β (KL penalty)
- Early stopping по human eval
- Обновите RM на новых данных

### Проблема: Медленная сходимость PPO

**Решение:**
- Проверьте advantage нормализацию
- Уменьшите clip_range
- Увеличьте batch size

---

## 📝 Финальный чеклист

### Обязательные компоненты

- [x] SFT обучена и сохранена
- [x] RM обучена на предпочтениях
- [x] PPO fine-tuning с KL penalty
- [x] DPO fine-tuning (альтернатива)
- [x] Автоматические метрики (reward, KL, perplexity)
- [x] Human evaluation (win rate)
- [x] Safety testing (toxicity, bias)

### Документация

- [x] README с инструкциями
- [x] Notebooks с анализом
- [x] Графики обучения
- [x] Примеры генераций
- [x] Итоговый отчёт

---

## 🎓 Поздравляем с завершением курса!

Вы прошли полный путь от основ MDP до современного RLHF. Теперь вы можете:

✅ Реализовать любой RL алгоритм с нуля  
✅ Применить RLHF для alignment LLM  
✅ Оценить качество и безопасность AI систем  
✅ Читать и имплементировать research papers  

---

**Следующие шаги:**

1. Примените RLHF на реальной задаче
2. Изучите advanced topics (multi-agent, model-based RL)
3. Участвуйте в RL competitions
4. Внесите вклад в open-source RL библиотеки

---

**Автор:** Denis Samatov, TPU / 2025  
**Контакт:** [GitHub](https://github.com/denissamatov)

🚀 **Удачи в применении Reinforcement Learning!**

