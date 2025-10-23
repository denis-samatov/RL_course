Семинар 1 — MDP и уравнения Беллмана

Структура
- notebooks/sem01_mdp_bellman.ipynb — основной практический ноутбук
- docs/theory.md — теоретические заметки и ссылки
- src/rlcourse/sem01 — переиспользуемый код (среды, алгоритмы, дискретизация, политики, видео)
- scripts/ — запускаемые демо-скрипты (run_demo.py, record_video.py, two_state_mdp_demo.py)
- tools/ — вспомогательные скрипты для патча ноутбука (необязательно)

Установка
- Python 3.10+
- pip install -r ../requirements.txt (запустить из корня репозитория)

Использование
- Демо: `python scripts/run_demo.py --env cartpole --mode random|mc|td0` (для Q-learning: `--env frozenlake --mode ql`)
- Запись видео: `python scripts/record_video.py` (файл будет в `assets/videos/`)
- Демо двухсостоянийного MDP: `python scripts/two_state_mdp_demo.py`
