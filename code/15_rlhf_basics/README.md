# 🤖 Семинар 15: RLHF — Reinforcement Learning from Human Feedback

> **Теория:** [note_15_rlhf_pipeline.md](../../notes/md/note_15_rlhf_pipeline.md)  
> **Pipeline:** SFT → Reward Model → PPO с KL-штрафом

---

## 📖 Обзор

Упрощённая демонстрация RLHF пайплайна на игрушечной задаче.

**Особенности:**
- 3-этапный pipeline (SFT → RM → PPO)
- Reward Model на предпочтениях
- KL-penalty к reference policy
- Мониторинг reward hacking

---

## 🗂️ Структура

```
rlhf_basics/
├── simple_text_env.py      # Игрушечная среда для текстовой генерации
├── sft_model.py            # Supervised Fine-Tuning
├── reward_model.py         # Reward Model обучение
├── ppo_rlhf.py             # PPO с KL-штрафом
├── generate_data.py        # Генерация preference данных
├── train_pipeline.py       # Полный pipeline
├── README.md               # Эта документация
└── data/                   # (создаётся при запуске)
    ├── sft_data.json
    └── preferences.json
```

---

## 🚀 Быстрый старт

```bash
# 1. Генерация данных
python generate_data.py

# 2. Полный RLHF pipeline
python train_pipeline.py
```

**Результат:**
- SFT model checkpoint
- Reward Model checkpoint
- RLHF-aligned model
- Графики сравнения

---

## 📊 Эксперименты

### Эксперимент 1: Влияние β (KL-штраф)

```bash
python ppo_rlhf.py --beta 0.001  # Слабый контроль
python ppo_rlhf.py --beta 0.01   # Baseline
python ppo_rlhf.py --beta 0.1    # Сильный контроль
```

**Ожидаемое:**
- β=0.001: Высокая reward, но reward hacking
- β=0.01: Баланс
- β=0.1: Стабильно, но медленное улучшение

---

### Эксперимент 2: Reward Hacking без KL

```bash
python ppo_rlhf.py --beta 0.0  # Без KL-штрафа
```

**Ожидаемое:** Модель находит adversarial примеры с высокой reward, но бессмысленные.

---

## 💡 Ключевые компоненты

### 1. Reward Model Loss

```python
# Bradley-Terry Model
P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

loss = -log σ(r(x, y_w) - r(x, y_l))
```

### 2. PPO с KL-penalty

```python
total_reward = rm_reward - β * KL(π_θ || π_SFT)
```

### 3. Мониторинг метрик

- RM Reward (должна расти)
- KL Divergence (должна быть < 10)
- Response Length (не должна взорваться)
- Perplexity (должна остаться разумной)

---

**Автор:** Denis Samatov, TPU / 2025

✅ **Семинар 15 завершён!** Переходим к [Семинару 16: DPO](../16_dpo/README.md)

