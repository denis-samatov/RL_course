## CartPole-v1 — Q-learning with the Bellman update (state discretization)

This module implements tabular Q-learning for `CartPole-v1` with discretization of the continuous observations and Bellman updates. The script supports training, saving the Q-table, evaluating the greedy policy, and recording episode videos.

### Files
- `q_learning_cartpole.py` — the main script with the agent, training, evaluation, and video recording.
- `cartpole/` — folder where recorded episode videos are saved (created automatically when recording).

### Installation
Dependencies are listed in the root `requirements.txt`. Install any missing packages if needed:
```bash
pip install gym numpy tqdm
```
Recording video (and its codecs) requires a video pipeline (FFmpeg) and Gym's recording wrapper:
```bash
pip install moviepy
# install FFmpeg on your system if needed, e.g. via brew: brew install ffmpeg
```

### Running training and evaluation
```bash
python code/09_q_learning_bellman/q_learning_cartpole.py \
  --episodes 4000 \
  --max-steps 500 \
  --lr 0.1 \
  --gamma 0.99 \
  --eps-start 1.0 \
  --eps-end 0.05 \
  --eps-decay-episodes 2000 \
  --bins-x 8 --bins-xdot 8 --bins-theta 16 --bins-thetadot 16 \
  --eval-episodes 10 \
  --output cartpole_q_table.npy \
  --video-dir videos/cartpole \
  --video-episodes 3
```
- After training, the script prints the average reward over the last 100 episodes, evaluates the greedy policy, and saves the Q-table.
- If `--video-dir` is given, `--video-episodes` greedy episodes will be recorded.

### Key ideas and math
- The continuous state $s=(x,\dot{x},\theta,\dot{\theta})$ is discretized along each coordinate into the given number of bins. The state index is the flattened index of the 4D grid.
- Actions are discrete: $a \in \{0,1\}$.
- The Q-learning update (off-policy, Bellman-based):

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\,\bigl(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\bigr).
$$

- Exploration: an $\varepsilon$-greedy policy with $\varepsilon$ linearly decayed from `eps_start` to `eps_end` over `eps_decay_episodes`.

### Default hyperparameters
- Episodes: `4000`, steps per episode: `500`
- Training: `lr=0.1`, `gamma=0.99`
- Epsilon: `1.0 → 0.05` over `2000` episodes
- Discretization bins: `x=8, x_dot=8, theta=16, theta_dot=16`

### CLI arguments
- `--episodes`, `--max-steps`, `--lr`, `--gamma`
- `--eps-start`, `--eps-end`, `--eps-decay-episodes`
- `--bins-x`, `--bins-xdot`, `--bins-theta`, `--bins-thetadot`
- `--eval-episodes`
- `--output` — path to save the Q-table to (`.npy`)
- `--video-dir` — folder for videos (empty string disables recording)
- `--video-episodes` — how many evaluation episodes to record

### Evaluation and video
- Evaluation: after training, `evaluate(...)` is called, which runs the greedy policy $\arg\max_a Q(s,a)$ and prints the average reward and episode length.
- Video: when `--video-dir` is given, episodes are recorded via `RecordVideo`; note the overwrite warning if the folder already exists.

### Tips
- Increasing the number of bins increases the size of the state space: balance granularity against learnability.
- If training is unstable, try lowering `lr`, increasing the number of episodes, or adjusting the clipping ranges used for discretization.
