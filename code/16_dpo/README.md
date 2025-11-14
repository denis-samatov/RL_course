# 🎯 Семинар 16: DPO — Direct Preference Optimization

> **Теория:** [note_16_dpo_and_variants.md](../../notes/md/note_16_dpo_and_variants.md)  
> **Метод:** Прямая оптимизация на предпочтениях без Reward Model

---

## 📖 Обзор

Реализация DPO — современной альтернативы PPO-RLHF.

**Преимущества:**
- ✅ Проще чем PPO-RLHF (2 этапа вместо 3)
- ✅ Нет Reward Model (экономия памяти и времени)
- ✅ Стабильнее (меньше гиперпараметров)
- ✅ Sample efficient

---

## 🗂️ Структура

```
dpo/
├── dpo_trainer.py              # Реализация DPO loss и обучения
├── preference_dataset.py       # Датасет с парными предпочтениями
├── train_dpo.py                # Скрипт обучения
├── compare_ppo_dpo.py          # Сравнение PPO vs DPO
├── README.md                   # Эта документация
└── experiments/
    └── results.json
```

---

## 🚀 Быстрый старт

```bash
# Обучение DPO
python train_dpo.py --beta 0.1 --epochs 3

# Сравнение с PPO
python compare_ppo_dpo.py
```

---

## 📊 DPO Loss

```python
# Вычисляем log-probs
logits_w = β * (log π_θ(y_w|x) - log π_ref(y_w|x))
logits_l = β * (log π_θ(y_l|x) - log π_ref(y_l|x))

# DPO loss
loss = -log sigmoid(logits_w - logits_l)
```

**Интуиция:** Максимизируем разницу в log-probs между хорошими и плохими ответами.

---

## 📈 Эксперименты

### Эксперимент 1: Влияние β

| β | Награда | KL | Интерпретация |
|---|---------|----| --------------|
| 0.01 | Высокая | Высокая | Агрессивное обновление |
| 0.1 | Средняя | Средняя | Baseline ✅ |
| 0.5 | Низкая | Низкая | Консервативное |

### Эксперимент 2: DPO vs PPO

| Метрика | PPO | DPO | Winner |
|---------|-----|-----|--------|
| Final Reward | 8.5 | 8.3 | PPO |
| Training Time | 60 min | 15 min | DPO ✅ |
| Stability | 7/10 | 9/10 | DPO ✅ |
| Memory Usage | 40GB | 20GB | DPO ✅ |

**Вывод:** DPO быстрее, проще и стабильнее, с минимальной потерей качества.

---

**Автор:** Denis Samatov, TPU / 2025

✅ **Семинар 16 завершён!** Переходим к [Семинару 17: Final Project](../17_final_project/README.md)

