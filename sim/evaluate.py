"""
evaluate.py -- Multi-seed Evaluation & Heatmaps (Phase 1)
===========================================================
Compares:
  - Trained Q-Learning
  - Always-Local
  - Always-Edge
  - Always-Cloud
  - Static-Threshold

Over multiple seeds to compute mean and std of rewards.
Produces:
  - results/reward_curve.png
  - results/bar_chart_metrics.png
  - results/policy_heatmaps.png
  - results/eval_summary.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from offload_env import (OffloadEnv, N_ACTIONS, ACTION_LOCAL, ACTION_EDGE, ACTION_CLOUD,
                         N_BATTERY_BINS, N_NETWORKS, N_TASKS, N_EDGE_QUEUE_BINS,
                         index_to_state, state_to_index)
from train_qlearning import train_qtable, choose_action, init_qtable
from utils import load_config, ensure_dir

# ── Baseline Policies ──────────────────────────────────────────────────────────

def policy_always_local(state: int, env: OffloadEnv) -> int:
    return ACTION_LOCAL

def policy_always_edge(state: int, env: OffloadEnv) -> int:
    return ACTION_EDGE

def policy_always_cloud(state: int, env: OffloadEnv) -> int:
    return ACTION_CLOUD

def policy_static_threshold(state: int, env: OffloadEnv) -> int:
    """
    Static heuristic:
    - If battery < 20% (bin 0), try to offload.
      - If network poor, CLOUD is risky/expensive, EDGE might be bad. But LOCAL has penalty.
      - If network good, EDGE or CLOUD.
    - If task is heavy and battery is fine:
      - If network is poor -> LOCAL.
      - If network is good and queue is short -> EDGE.
      - If queue is long -> CLOUD.
    Let's make it simple:
    """
    bat, net, task, q, dev = index_to_state(state)
    
    if bat == 0:
        return ACTION_CLOUD if net > 0 else ACTION_LOCAL
    
    if task == 2: # heavy
        if net == 0:
            return ACTION_LOCAL
        elif q >= 3:
            return ACTION_CLOUD
        else:
            return ACTION_EDGE
    elif task == 0: # light
        return ACTION_LOCAL
    else: # medium
        if q < 2 and net > 0:
            return ACTION_EDGE
        else:
            return ACTION_LOCAL

# ── Runner ─────────────────────────────────────────────────────────────────────

def run_policy(env: OffloadEnv, policy_fn, n_episodes: int, rng: np.random.Generator) -> dict:
    rewards = []
    energy = []
    latency = []
    accuracy = []
    
    for _ in range(n_episodes):
        state = env.reset()
        ep_r = 0.0
        while True:
            action = policy_fn(state, env)
            next_state, r, done, info = env.step(action)
            ep_r += r
            energy.append(info["energy"][0])
            latency.append(info["latency"][0])
            accuracy.append(info["accuracy"][0])
            state = next_state
            if done:
                break
        rewards.append(ep_r)
        
    return {
        "rewards": np.array(rewards),
        "energy": np.array(energy),
        "latency": np.array(latency),
        "accuracy": np.array(accuracy)
    }

# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_heatmaps(qtable: np.ndarray, results_dir: str):
    """
    Plots policy heatmaps for a single Q-table.
    Axes: Battery Bin (Y) vs Network (X).
    Panels for different (Task, Queue) combinations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        (0, 0, "Light Task, Empty Queue"),
        (2, 0, "Heavy Task, Empty Queue"),
        (2, 4, "Heavy Task, Full Queue"),
        (1, 2, "Medium Task, Med Queue")
    ]
    
    net_labels = ["Poor", "Medium", "Good"]
    bat_labels = ["<20%", "20-40%", "40-60%", "60-80%", ">80%"]
    cmap = matplotlib.colors.ListedColormap(["#E07B54", "#5B8ED6", "#6ABF84"]) # Local, Edge, Cloud
    
    for ax, (task, q, title) in zip(axes.flatten(), panels):
        img = np.zeros((N_BATTERY_BINS, N_NETWORKS))
        for bat in range(N_BATTERY_BINS):
            for net in range(N_NETWORKS):
                idx = state_to_index(bat, net, task, q, 0)
                best_action = np.argmax(qtable[idx])
                img[bat, net] = best_action
                
        sns.heatmap(img, ax=ax, cmap=cmap, vmin=0, vmax=2, 
                    xticklabels=net_labels, yticklabels=bat_labels,
                    cbar=False, annot=True, fmt=".0f", annot_kws={"size": 14})
        ax.set_title(title)
        ax.set_xlabel("Network Quality")
        ax.set_ylabel("Battery Level")
        ax.invert_yaxis()
        
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#E07B54", label="LOCAL (0)"),
                       Patch(facecolor="#5B8ED6", label="EDGE (1)"),
                       Patch(facecolor="#6ABF84", label="CLOUD (2)")]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
    
    plt.tight_layout()
    out = os.path.join(results_dir, "policy_heatmaps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

def plot_bar_metrics(summary_df: pd.DataFrame, results_dir: str):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = [
        ("mean_reward", "Mean Reward (Higher is better)"),
        ("mean_energy", "Mean Energy (Lower is better)"),
        ("mean_latency", "Mean Latency (Lower is better)"),
        ("mean_accuracy", "Mean Accuracy (Higher is better)")
    ]
    
    for ax, (col, title) in zip(axes, metrics):
        sns.barplot(data=summary_df, x="policy", y=col, ax=ax, palette="Set2")
        ax.set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("")
        if col == "mean_reward":
            ax.errorbar(x=range(len(summary_df)), y=summary_df[col], 
                        yerr=summary_df["std_reward"], fmt='none', c='black', capsize=5)
            
    plt.tight_layout()
    out = os.path.join(results_dir, "bar_chart_metrics.png")
    fig.savefig(out, dpi=150)
    plt.close()

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    results_dir = os.path.join(os.path.dirname(__file__), cfg.get("results_dir", "results"))
    ensure_dir(results_dir)
    
    seeds = cfg.get("eval_seeds", [42, 123, 456, 789, 1337])
    n_eval = cfg.get("eval_episodes", 200)
    
    policies = ["Q-Learning", "Always-Local", "Always-Edge", "Always-Cloud", "Static-Threshold"]
    agg_results = {p: {"rewards": [], "energy": [], "latency": [], "accuracy": []} for p in policies}
    
    all_qtables = []
    train_histories = []
    
    print(f"Starting Multi-seed Evaluation ({len(seeds)} seeds, {n_eval} episodes/seed) ...")
    for seed in seeds:
        print(f"  Running seed {seed} ...")
        # 1. Train Q-Learning for this seed
        cfg["seed"] = seed
        qtable, hist = train_qtable(cfg, verbose=False)
        train_histories.append(hist)
        all_qtables.append(qtable)
            
        env = OffloadEnv(cfg)
        rng = np.random.default_rng(seed)
        
        # Q-Learning Policy wrapper
        def ql_policy(state, env_obj):
            return choose_action(qtable, state, epsilon=0.0, rng=rng)
            
        pol_fns = {
            "Q-Learning": ql_policy,
            "Always-Local": policy_always_local,
            "Always-Edge": policy_always_edge,
            "Always-Cloud": policy_always_cloud,
            "Static-Threshold": policy_static_threshold
        }
        
        for p_name, p_fn in pol_fns.items():
            res = run_policy(env, p_fn, n_eval, rng)
            agg_results[p_name]["rewards"].extend(res["rewards"])
            agg_results[p_name]["energy"].extend(res["energy"])
            agg_results[p_name]["latency"].extend(res["latency"])
            agg_results[p_name]["accuracy"].extend(res["accuracy"])

    # ── Summarise ──────────────────────────────────────────────────────────────
    rows = []
    for p_name in policies:
        r = np.array(agg_results[p_name]["rewards"])
        e = np.array(agg_results[p_name]["energy"])
        l = np.array(agg_results[p_name]["latency"])
        a = np.array(agg_results[p_name]["accuracy"])
        rows.append({
            "policy": p_name,
            "mean_reward": float(r.mean()),
            "std_reward": float(r.std()),
            "mean_energy": float(e.mean()),
            "mean_latency": float(l.mean()),
            "mean_accuracy": float(a.mean())
        })
        
    df_summary = pd.DataFrame(rows)
    csv_out = os.path.join(results_dir, "eval_summary.csv")
    df_summary.to_csv(csv_out, index=False)
    
    print("\n--- Final Results ---")
    print(df_summary.to_string(index=False))
    
    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_bar_metrics(df_summary, results_dir)
    
    avg_qtable = np.mean(all_qtables, axis=0)
    plot_heatmaps(avg_qtable, results_dir)
    
    # Training curve (average across seeds)
    fig, ax = plt.subplots(figsize=(8, 4))
    for hist in train_histories:
        episodes = [x[0] for x in hist]
        rewards = [x[1] for x in hist]
        ax.plot(episodes, rewards, alpha=0.2, color="#4C72B0")
    
    avg_rewards = np.mean([[x[1] for x in hist] for hist in train_histories], axis=0)
    smoothed = pd.Series(avg_rewards).rolling(50, min_periods=1).mean()
    ax.plot(episodes, smoothed, color="#4C72B0", linewidth=2.0, label="Mean Smoothed")
    ax.set_title("Training Reward Curve (Multiple Seeds)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(results_dir, "reward_curve.png"), dpi=150)
    plt.close()
    
    print(f"\nAll evaluation plots and data saved to {results_dir}")

if __name__ == "__main__":
    main()
