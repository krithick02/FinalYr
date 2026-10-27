"""
train_qlearning.py -- Tabular Q-Learning Trainer  (Phase 1)
============================================================
Public API
----------
    train_qtable(cfg)         -> np.ndarray   (no disk I/O -- for evaluate.py)
    main()                    -> saves data/q_table.npy + data/reward_history.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from offload_env import OffloadEnv, STATE_SIZE, N_ACTIONS
from utils import load_config, ensure_dir

# ── Q-table ────────────────────────────────────────────────────────────────────

def init_qtable(state_size: int = STATE_SIZE,
                n_actions: int  = N_ACTIONS) -> np.ndarray:
    return np.zeros((state_size, n_actions), dtype=float)


# ── Bellman TD(0) update ───────────────────────────────────────────────────────

def q_update(qtable: np.ndarray, state: int, action: int,
             reward: float, next_state: int,
             alpha: float, gamma: float) -> None:
    """
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    """
    td_target = reward + gamma * float(np.max(qtable[next_state]))
    qtable[state, action] += alpha * (td_target - qtable[state, action])


# ── Epsilon schedule ───────────────────────────────────────────────────────────

def make_epsilon_schedule(cfg: dict):
    eps_cfg    = cfg["epsilon"]
    start      = float(eps_cfg["start"])
    end        = float(eps_cfg["end"])
    decay_type = str(eps_cfg.get("decay_type", "exponential"))
    n_eps      = int(cfg["episodes"])

    if decay_type == "linear":
        decrement = (start - end) / max(n_eps - 1, 1)
        def schedule(ep):
            return max(end, start - decrement * ep)
    else:
        rate = float(eps_cfg.get("decay_rate", 0.9985))
        def schedule(ep):
            return max(end, start * (rate ** ep))
    return schedule

def make_alpha_schedule(cfg: dict):
    if isinstance(cfg.get("alpha"), dict):
        alpha_cfg = cfg["alpha"]
        start      = float(alpha_cfg["start"])
        end        = float(alpha_cfg["end"])
        decay_type = str(alpha_cfg.get("decay_type", "exponential"))
        n_eps      = int(cfg["episodes"])
        if decay_type == "linear":
            decrement = (start - end) / max(n_eps - 1, 1)
            def schedule(ep):
                return max(end, start - decrement * ep)
        else:
            rate = float(alpha_cfg.get("decay_rate", 0.9985))
            def schedule(ep):
                return max(end, start * (rate ** ep))
        return schedule
    else:
        val = float(cfg.get("alpha", 0.15))
        def schedule(ep):
            return val
        return schedule



# ── Epsilon-greedy action ──────────────────────────────────────────────────────

def choose_action(qtable: np.ndarray, state: int,
                  epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    q_vals = qtable[state]
    max_q  = float(np.max(q_vals))
    ties   = np.where(q_vals == max_q)[0]
    return int(rng.choice(ties))


# ── Core training loop (no disk I/O) ──────────────────────────────────────────

def train_qtable(cfg: dict, verbose: bool = False) -> tuple:
    """
    Train a Q-table from scratch using the supplied config.

    Returns
    -------
    qtable   : np.ndarray  shape (STATE_SIZE, N_ACTIONS)
    history  : list of (episode, total_reward, steps, epsilon)
    """
    env        = OffloadEnv(cfg)
    qtable     = init_qtable()
    rng        = np.random.default_rng(cfg.get("seed", None))
    alpha_fn   = make_alpha_schedule(cfg)
    gamma      = float(cfg["gamma"])
    n_episodes = int(cfg["episodes"])
    eps_fn     = make_epsilon_schedule(cfg)
    two_device = cfg.get("mode", "single") == "two_device"

    if two_device:
        qtables = [init_qtable(), init_qtable()]
    else:
        qtables = [qtable]

    history = []
    t0      = time.time()

    for ep in range(n_episodes):
        states  = env.reset()
        epsilon = eps_fn(ep)
        alpha   = alpha_fn(ep)

        if not two_device:
            states = [states]

        ep_reward = [0.0] * len(qtables)
        step      = 0

        while True:
            actions = [
                choose_action(qtables[i], states[i], epsilon, rng)
                for i in range(len(qtables))
            ]

            if two_device:
                next_states, rewards, done, info = env.step(actions)
            else:
                next_s, reward, done, info = env.step(actions[0])
                next_states = [next_s]
                rewards     = [reward]

            for i in range(len(qtables)):
                q_update(qtables[i], states[i], actions[i],
                         rewards[i], next_states[i], alpha, gamma)
                ep_reward[i] += rewards[i]

            states = next_states
            step  += 1
            if done:
                break

        total_r = sum(ep_reward)
        history.append((ep, total_r, step, epsilon))

        if verbose and ((ep + 1) % 500 == 0 or ep == n_episodes - 1):
            elapsed = time.time() - t0
            print(f"  ep {ep+1:>5}/{n_episodes}  reward={total_r:+.3f}"
                  f"  eps={epsilon:.4f}  t={elapsed:.1f}s")

    return qtables[0] if not two_device else qtables, history


# ── Save helpers ───────────────────────────────────────────────────────────────

def save_qtable(qtable: np.ndarray, data_dir: str, device_id: int = 0) -> str:
    ensure_dir(data_dir)
    path = os.path.join(data_dir, f"q_table{'_dev'+str(device_id) if device_id else ''}.npy")
    np.save(path, qtable)
    return path


def save_reward_history(history: list, data_dir: str) -> str:
    ensure_dir(data_dir)
    df  = pd.DataFrame(history, columns=["episode", "total_reward", "steps", "epsilon"])
    csv = os.path.join(data_dir, "reward_history.csv")
    df.to_csv(csv, index=False)
    return csv


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    cfg      = load_config()
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, cfg.get("data_dir", "data"))
    mode     = cfg.get("mode", "single")

    train_seeds = cfg.get("train_seeds", [cfg.get("seed", 42)])
    print(f"\n[{mode}] Training for {cfg['episodes']} episodes across {len(train_seeds)} seeds...")
    
    all_qtables = []
    
    for s in train_seeds:
        cfg["seed"] = s
        print(f"  Training seed {s} ...")
        qtable_or_list, history = train_qtable(cfg, verbose=False)
        all_qtables.append(qtable_or_list)

    # Average the Q-tables
    if mode == "two_device":
        avg_qtables = [np.mean([qt[i] for qt in all_qtables], axis=0) for i in range(2)]
        for i, qt in enumerate(avg_qtables):
            path = save_qtable(qt, data_dir, device_id=i)
            print(f"Averaged Q-table dev{i} -> {path}")
    else:
        avg_qtable = np.mean(all_qtables, axis=0)
        path = save_qtable(avg_qtable, data_dir, device_id=0)
        print(f"\nAveraged Q-table saved -> {path}")

    csv = save_reward_history(history, data_dir) # Just saves the last seed's history for simplicity
    print(f"Reward history (last seed) -> {csv}")
    print("Training complete.")


if __name__ == "__main__":
    main()
