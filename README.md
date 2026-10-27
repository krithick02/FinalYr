# IoT Task Offloading with Tabular Q-Learning

Simulates ESP32 device(s) learning to route tasks — **LOCAL**, **EDGE**, or **CLOUD** — using tabular Q-learning. The agent maximises a composite reward balancing energy efficiency, latency, and inference accuracy, while avoiding battery depletion.

---

## Project Structure

```
FinalYr/
├── sim/
│   ├── config.yaml          # All hyperparameters & mode switch
│   ├── offload_env.py       # Environment (state, transitions, reward)
│   ├── train_qlearning.py   # Q-table, epsilon-greedy, training loop
│   ├── evaluate.py          # Greedy evaluation, baselines, plots
│   ├── dashboard.py         # Interactive Streamlit dashboard (4 tabs)
│   └── utils.py             # Shared helpers (config loader, ensure_dir)
├── tests/
│   ├── test_env.py          # Environment unit tests
│   └── test_qtable.py       # Q-learning algorithm unit tests
├── requirements.txt         # All Python dependencies
└── README.md
```

---

## Interactive Dashboard

A four-tab Streamlit dashboard that loads the real Q-table and replays actual environment steps — **no fabricated numbers**.

### How to Run

```bash
# 1. Install all dependencies (one-time)
pip install -r requirements.txt

# 2. Train the Q-table if you haven't yet
cd sim
python train_qlearning.py      # writes sim/data/q_table.npy

# 3. Launch the dashboard
cd ..                          # back to project root
streamlit run sim/dashboard.py
```

The dashboard opens at **http://localhost:8501** in your browser.

### Tabs

| Tab | Description |
|-----|-------------|
| **▶ Live Simulation** | Play / Pause / Step / Reset controls; policy dropdown; ESP32→destination routing diagram; live battery, network, queue gauges; rolling charts |
| **📊 Compare Policies** | All 5 policies run on the same seed; summary table + 4-panel bar chart |
| **🔍 Ask the Agent** | Pick any (battery, network, task, queue) state; see Q-values per action; warns on unvisited states |
| **🗺️ Policy Heatmaps** | Battery × Network heatmaps for different task/queue combos; unvisited cells shown in grey |

---

## Quick Start

### 1 — Install dependencies
```bash
pip install numpy matplotlib pandas pytest pyyaml
```

### 2 — Train
```bash
cd sim
python train_qlearning.py
```
Outputs `results/qtable_dev0.npy` and `results/training_rewards.csv`.

### 3 — Evaluate
```bash
python evaluate.py
```
Outputs four plots and `results/evaluation_summary.csv`.

### 4 — Run tests
```bash
cd ..
pytest tests/ -v
```

---

## State Space

| Dimension       | Values                          | Bins/States |
|-----------------|---------------------------------|-------------|
| Battery level   | 0–100 %                         | 5           |
| Network quality | poor / medium / good            | 3           |
| Task type       | light / medium / heavy          | 3           |
| Edge queue depth| 0 – `max_edge_queue`            | 5           |
| Device ID       | 0 or 1                          | 2           |

**Total states:** 5 × 3 × 3 × 5 × 2 = **450**  
**Actions:** 0 = LOCAL, 1 = EDGE, 2 = CLOUD  
**Q-table shape:** (450, 3)

---

## Reward Function

```
R = -w1·energy + w2·(1/latency) + w3·accuracy
  − low_battery_penalty   if action=LOCAL and battery < 20%
  − death_penalty         if battery drops to 0
  − fairness_penalty      (two-device mode: penalise action imbalance)
```

All weights and penalties are configurable in `config.yaml`.

---

## Modes

Switch between modes by editing `config.yaml`:

```yaml
mode: "single"      # one ESP32
# mode: "two_device"  # two ESP32s sharing one edge server
```

In two-device mode each device maintains its own Q-table; they share the edge queue and incur a fairness penalty when their actions diverge.

---

## Config Reference (`sim/config.yaml`)

| Key | Description |
|-----|-------------|
| `mode` | `"single"` or `"two_device"` |
| `episodes` | Number of training episodes |
| `steps_per_episode` | Max steps per episode |
| `seed` | RNG seed (`null` for random) |
| `alpha` | Q-learning rate |
| `gamma` | Discount factor |
| `epsilon.start/end` | Exploration bounds |
| `epsilon.decay_type` | `"linear"` or `"exponential"` |
| `epsilon.decay_rate` | Rate for exponential decay |
| `weights.w1/w2/w3` | Reward component weights |
| `penalties.*` | Low-battery, death, fairness penalties |
| `energy/latency/accuracy` | Per-(task, action) lookup tables |
| `network_transition` | Markov transition matrix for network quality |
| `queue_arrival_prob` | P(new job arrives at edge each step) |
| `queue_departure_prob` | P(job departs from edge each step) |
| `eval_episodes` | Episodes for greedy evaluation |
| `results_dir` | Output folder (relative to `sim/`) |

---

## Output Files (`sim/results/`)

| File | Description |
|------|-------------|
| `qtable_dev0.npy` | Trained Q-table (device 0) |
| `qtable_dev1.npy` | Trained Q-table (device 1) — two-device mode only |
| `training_rewards.csv` | Per-episode reward, steps, epsilon |
| `evaluation_summary.csv` | Mean reward/energy/latency/accuracy per policy |
| `plot_reward_curve.png` | Smoothed training reward over episodes |
| `plot_action_dist.png` | Action distribution: Q-Learning vs baselines |
| `plot_battery_traj.png` | Battery level trajectory (mean ± std) |
| `plot_energy_latency.png` | Energy vs latency scatter with reward bubble size |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Q-table, numerical ops |
| `matplotlib` | Plots |
| `pandas` | CSV export / summary tables |
| `pytest` | Unit testing |
| `pyyaml` | Config parsing |

No `gym`, no deep learning frameworks.
