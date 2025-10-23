# 🎓 Reinforcement Learning — Практический курс (2025)

> **Цель:** Понять и реализовать ключевые алгоритмы обучения с подкреплением (RL) — от MDP и уравнений Беллмана до RLHF (PPO, DPO).  
> **Формат:** 10 семинаров × 1 час, с кодом на Python, Gymnasium и PyTorch.

---

## 📘 Общая информация

| Параметр | Описание |
|-----------|-----------|
| **Период** | 20 октября → 28 декабря 2025 |
| **Ритм** | 1 семинар в неделю (10 занятий × 60 минут) |
| **Формат занятия** | 10 мин теория → 40 мин код/эксперимент → 10 мин обсуждение |
| **Инструменты** | `python 3.10`, `gymnasium`, `torch`, `numpy`, `matplotlib`, `stable-baselines3`, `trl`, `wandb` |
| **Оценивание** | мини-дз после каждого семинара + финальный мини-проект |
| **Ресурсы** | CPU / GPU (желательно), `venv` окружение |

---

## 🗓️ План семинаров (Октябрь → Декабрь 2025)

| № | Тема | Ключевые понятия | Практика |
|---|------|------------------|-----------|
| **1** | Введение в RL и MDP | агент–среда–награда, функции `V`, `Q`, уравнения Беллмана | `CartPole-v1` (random policy, MC-оценка) |
| **2** | Динамическое программирование | Policy / Value Iteration, GPI | `GridWorld` — визуализация `V(s)` |
| **3** | Монте-Карло и TD | Sarsa, Q-Learning, ε-greedy | `FrozenLake-v1` (on/off-policy) |
| **4** | DQN и его расширения | Target-network, Replay, Double/Dueling DQN | `CartPole-v1` — PyTorch реализация |
| **5** | Policy Gradients, REINFORCE | PG-теорема, baseline, энтропия | `LunarLander-v2` — PG с baseline |
| **6** | Actor–Critic, A2C и SAC (идея) | GAE, A2C, max-entropy RL | `CartPole`, `Pendulum-v1` |
| **7** | PPO и TRPO | суррогатная цель, clipping, KL-контроль | `Pendulum-v1` — PPO (SB3) |
| **8** | RLHF: SFT → RM → PPO | KL-штраф, стабильность и drift | `trl` — PPO-fine-tune на игрушечных данных |
| **9** | Reward Modeling и DPO | бинарная RM, DPO-обновление | `trl.DPOTrainer` — сравнение PPO vs DPO |
| **10** | Финальный проект и безопасность | метрики reward/KL/safety, отчёт | Мини-пайплайн RLHF (SFT + RM + PPO / DPO) |

---

## 🗂️ Структура семинаров

Все занятия оформляем по одному шаблону — это облегчает навигацию и повторное использование кода:

- `semXX_topic/`
  - `README.md` — цели, как запускать ноутбук и скрипты, ссылки на теорию.
  - `docs/` — теоретические конспекты (Markdown, формулы, визуализации).
  - `notebooks/` — практические ноутбуки, где код импортируется из `src/`.
  - `scripts/` — CLI-демки, запись видео, вспомогательные утилиты.
  - `assets/` — результаты (например, `assets/videos/`, `assets/figures/`), игнорируются Git.
  - `tools/` — сервисные скрипты (патчеры ноутбуков, генераторы данных).
  - `tests/` *(опционально)* — автопроверки, мини-ДЗ, sample-solution проверки.
- `src/rlcourse/semXX/` — модуль с алгоритмами, политиками, утилитами и фабриками сред, которые импортируются ноутбуками и скриптами.

> Для новых семинаров копируйте структуру `sem01_mdp_bellman/` и заменяйте `sem01…` на нужный индекс/тему.

---

## 🎯 Результаты обучения

| Семинар | Результат |
|----------|------------|
| 1 | Формализует MDP, объясняет `V`, `Q`, `π` |
| 2 | Пишет Value / Policy Iteration и анализирует сходимость |
| 3 | Реализует Sarsa и Q-Learning, тюнит ε-greedy |
| 4 | Создаёт DQN (PyTorch), добавляет Double / Dueling |
| 5 | Выводит PG-формулы, снижает дисперсию (baseline/entropy) |
| 6 | Запускает A2C, понимает GAE и идею max-entropy |
| 7 | Тюнит PPO (clip / KL / entropy), сравнивает с A2C |
| 8 | Понимает пайплайн RLHF и роль KL-штрафа |
| 9 | Обучает Reward Model и делает DPO-обновление |
| 10 | Собирает mini-RLHF pipeline и оценивает метрики |

---

## 💡 Итоговые проекты и исследовательские направления

1. **Mini-RLHF** — обучение LLM на коротких инструкциях: SFT → RM (200–500 пар) → PPO / DPO.  
2. **PPO vs DPO** — анализ стабильности и ресурсоёмкости при разных `β`, `LR`, `batch_size`.  
3. **RL + Human Scoring** — ручной скоринг траекторий и оценка сдвига политики.

---

## 📚 Рекомендуемая литература

| Источник | Содержание |
|-----------|-------------|
| **Sutton & Barto** — *Reinforcement Learning: An Introduction* | классика (MDP, DP, TD, PG) :contentReference[oaicite:0]{index=0} |
| **Maxim Lapan** — *Deep RL Hands-On* | практические реализации на PyTorch :contentReference[oaicite:1]{index=1} |
| **Andrea Lonza** — *RL Algorithms with Python* | SARSA, Q-Learning, PPO, TRPO, DDPG / TD3 :contentReference[oaicite:2]{index=2} |
| **RL Theory Book** — Forts & Mills | строгие выводы PG, PPO, GAE :contentReference[oaicite:3]{index=3} |
| **Hugging Face TRL Docs** | RLHF, PPO, DPO, ORPO — современные пайплайны |

---

> 💬 Разработано на основе курсов **Sutton & Barto**, **Lapan**, **Lonza**, **RL Theory Book**, **Hugging Face TRL**.  
> Автор: *Denis Samatov, TPU / 2025*  
>  
> 📂 Репозиторий включает:  
> — методички (формулы, визуализации)  
> — практические тетради с кодом  
> — демо, запись видео, mini-MDP  
> — общие модули (алгоритмы, политики, утилиты)  
> — мини-проекты RL и RLHF  
>  
> 🧭 Цель: сформировать мост между **академическим RL** и **прикладным Deep RL / RLHF**.

---
