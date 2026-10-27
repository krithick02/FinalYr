import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from offload_env import OffloadEnv, N_BATTERY_BINS, N_NETWORKS, state_to_index
from evaluate import policy_static_threshold, run_policy, choose_action
from train_qlearning import train_qtable
from utils import load_config, ensure_dir

def load_avg_qtable(data_dir):
    return np.load(os.path.join(data_dir, "q_table.npy"))

# ── Task 1: 15 seeds Baseline Comparison ──────────────────────────────────────
def task1_baseline_15seeds(cfg, results_dir):
    print("\n--- TASK 1: 15-Seed Baseline Comparison ---")
    data_dir = os.path.join(os.path.dirname(__file__), cfg.get("data_dir", "data"))
    qtable = load_avg_qtable(data_dir)
    
    seeds = np.random.randint(0, 100000, size=15).tolist()
    n_eval = 200
    
    ql_rewards, st_rewards = [], []
    ql_energy, st_energy = [], []
    ql_latency, st_latency = [], []
    
    for s in seeds:
        env = OffloadEnv(cfg)
        rng = np.random.default_rng(s)
        
        def ql_policy(state, env_obj):
            return choose_action(qtable, state, epsilon=0.0, rng=rng)
            
        res_ql = run_policy(env, ql_policy, n_eval, rng)
        res_st = run_policy(env, policy_static_threshold, n_eval, rng)
        
        ql_rewards.extend(res_ql["rewards"])
        st_rewards.extend(res_st["rewards"])
        
        ql_energy.extend(res_ql["energy"])
        st_energy.extend(res_st["energy"])
        
        ql_latency.extend(res_ql["latency"])
        st_latency.extend(res_st["latency"])
        
    ql_r, st_r = np.array(ql_rewards), np.array(st_rewards)
    
    # Welch t-test
    t_stat, p_val = stats.ttest_ind(ql_r, st_r, equal_var=False)
    
    # 95% CI for the difference of means
    mean_diff = np.mean(ql_r) - np.mean(st_r)
    se_diff = np.sqrt(np.var(ql_r, ddof=1)/len(ql_r) + np.var(st_r, ddof=1)/len(st_r))
    ci_low = mean_diff - 1.96 * se_diff
    ci_high = mean_diff + 1.96 * se_diff
    
    print(f"Evaluated on {len(seeds)} seeds * {n_eval} episodes = {len(ql_r)} episodes total.")
    print("Q-Learning vs Static-Threshold:")
    print(f"  QL Mean Reward: {np.mean(ql_r):.2f} ± {np.std(ql_r):.2f}")
    print(f"  ST Mean Reward: {np.mean(st_r):.2f} ± {np.std(st_r):.2f}")
    print(f"  QL Energy: {np.mean(ql_energy)*1000:.2f} mWh, Latency: {np.mean(ql_latency)*1000:.2f} ms")
    print(f"  ST Energy: {np.mean(st_energy)*1000:.2f} mWh, Latency: {np.mean(st_latency)*1000:.2f} ms")
    print(f"  95% CI of Difference (QL - ST): [{ci_low:.2f}, {ci_high:.2f}]")
    sig = "Significant" if p_val < 0.05 else "Not Significant"
    print(f"  Welch t-test p-value: {p_val:.4e} ({sig})")


# ── Task 2: Heatmaps with unvisited states greyed out ─────────────────────────
def task2_heatmaps_grey(cfg, results_dir):
    print("\n--- TASK 2: Heatmaps (Unvisited states greyed out) ---")
    data_dir = os.path.join(os.path.dirname(__file__), cfg.get("data_dir", "data"))
    qtable = load_avg_qtable(data_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        (0, 0, "Light Task, Empty Queue"),
        (2, 0, "Heavy Task, Empty Queue"),
        (2, 4, "Heavy Task, Full Queue"),
        (1, 2, "Medium Task, Med Queue")
    ]
    
    net_labels = ["Poor", "Medium", "Good"]
    bat_labels = ["<20%", "20-40%", "40-60%", "60-80%", ">80%"]
    
    # Custom colormap
    base_cmap = matplotlib.colors.ListedColormap(["#E07B54", "#5B8ED6", "#6ABF84"])
    base_cmap.set_bad('grey')
    
    for ax, (task, q, title) in zip(axes.flatten(), panels):
        img = np.zeros((N_BATTERY_BINS, N_NETWORKS))
        for bat in range(N_BATTERY_BINS):
            for net in range(N_NETWORKS):
                idx = state_to_index(bat, net, task, q, 0)
                if np.all(qtable[idx] == 0):
                    img[bat, net] = np.nan
                else:
                    img[bat, net] = np.argmax(qtable[idx])
                
        sns.heatmap(img, ax=ax, cmap=base_cmap, vmin=0, vmax=2, 
                    xticklabels=net_labels, yticklabels=bat_labels,
                    cbar=False, annot=True, fmt=".0f", annot_kws={"size": 14})
        ax.set_title(title)
        ax.set_xlabel("Network Quality")
        ax.set_ylabel("Battery Level")
        ax.invert_yaxis()
        
    plt.tight_layout()
    out = os.path.join(results_dir, "policy_heatmaps_grey.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {out}")

# ── Task 3: Two-Device Mode (Fairness vs No Fairness) ─────────────────────────
def train_and_eval_two_device(cfg, fairness_penalty):
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["mode"] = "two_device"
    cfg_copy["penalties"]["fairness"] = fairness_penalty
    cfg_copy["episodes"] = 4000 # Enough for comparison
    cfg_copy["train_seeds"] = [42] # Single seed to save time
    
    qtables, _ = train_qtable(cfg_copy, verbose=False) # Returns list of 2 qtables
    
    env = OffloadEnv(cfg_copy)
    rng = np.random.default_rng(123)
    
    n_eval = 200
    e0, e1 = [], []
    l0, l1 = [], []
    
    for _ in range(n_eval):
        states = env.reset()
        while True:
            a0 = choose_action(qtables[0], states[0], 0.0, rng)
            a1 = choose_action(qtables[1], states[1], 0.0, rng)
            ns, rs, done, info = env.step([a0, a1])
            e0.append(info["energy"][0])
            e1.append(info["energy"][1])
            l0.append(info["latency"][0])
            l1.append(info["latency"][1])
            states = ns
            if done:
                break
                
    return np.mean(e0)*1000, np.mean(e1)*1000, np.mean(l0)*1000, np.mean(l1)*1000

def task3_two_device_fairness(cfg):
    print("\n--- TASK 3: Two-Device Mode (Fairness) ---")
    print("Training with Fairness = 1.0 (Fair)...")
    fe0, fe1, fl0, fl1 = train_and_eval_two_device(cfg, 1.0)
    print("Training with Fairness = 0.0 (Unfair)...")
    ue0, ue1, ul0, ul1 = train_and_eval_two_device(cfg, 0.0)
    
    print("Fairness ON:")
    print(f"  Dev0 Energy: {fe0:.2f} mWh, Dev1 Energy: {fe1:.2f} mWh (Var: {np.var([fe0, fe1]):.4f})")
    print(f"  Dev0 Latency: {fl0:.2f} ms, Dev1 Latency: {fl1:.2f} ms (Var: {np.var([fl0, fl1]):.4f})")
    
    print("Fairness OFF:")
    print(f"  Dev0 Energy: {ue0:.2f} mWh, Dev1 Energy: {ue1:.2f} mWh (Var: {np.var([ue0, ue1]):.4f})")
    print(f"  Dev0 Latency: {ul0:.2f} ms, Dev1 Latency: {ul1:.2f} ms (Var: {np.var([ul0, ul1]):.4f})")

# ── Task 4: Adaptation Experiment ─────────────────────────────────────────────
def task4_adaptation(cfg, results_dir):
    print("\n--- TASK 4: Adaptation Experiment ---")
    data_dir = os.path.join(os.path.dirname(__file__), cfg.get("data_dir", "data"))
    base_qtable = load_avg_qtable(data_dir)
    
    qtable_frozen = np.copy(base_qtable)
    qtable_adapt = np.copy(base_qtable)
    
    env_cfg = copy.deepcopy(cfg)
    env = OffloadEnv(env_cfg)
    rng = np.random.default_rng(999)
    
    total_episodes = 1000
    midpoint = 500
    
    frozen_rewards = []
    adapt_rewards = []
    
    alpha = 0.02
    gamma = cfg["gamma"]
    epsilon = 0.05
    
    for ep in range(total_episodes):
        if ep == midpoint:
            # Degrade environment
            env.max_queue = min(env.max_queue * 2, 40)
            env.q_arrive = min(env.q_arrive * 2.0, 0.95)
            # Shift network to poor (60% poor, 30% med, 10% good) for all states
            for dev in env.devices:
                dev.cfg["network_transition"]["poor"] = [0.8, 0.15, 0.05]
                dev.cfg["network_transition"]["medium"] = [0.6, 0.3, 0.1]
                dev.cfg["network_transition"]["good"] = [0.5, 0.3, 0.2]
                
        # ── Frozen Agent ──
        env.rng = np.random.default_rng(ep) # Sync RNG
        s = env.reset()
        ep_r_frozen = 0
        while True:
            a = choose_action(qtable_frozen, s, 0.0, env.rng)
            ns, r, done, _ = env.step(a)
            ep_r_frozen += r
            s = ns
            if done: break
        frozen_rewards.append(ep_r_frozen)
        
        # ── Adaptive Agent ──
        env.rng = np.random.default_rng(ep) # Sync RNG
        s = env.reset()
        ep_r_adapt = 0
        while True:
            a = choose_action(qtable_adapt, s, epsilon, env.rng)
            ns, r, done, _ = env.step(a)
            
            # Update
            td_target = r + gamma * np.max(qtable_adapt[ns])
            qtable_adapt[s, a] += alpha * (td_target - qtable_adapt[s, a])
            
            ep_r_adapt += r
            s = ns
            if done: break
        adapt_rewards.append(ep_r_adapt)

    # Plotting
    f_smooth = pd.Series(frozen_rewards).rolling(50, min_periods=1).mean()
    a_smooth = pd.Series(adapt_rewards).rolling(50, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(f_smooth, label="Frozen Q-Table", color="#E07B54")
    ax.plot(a_smooth, label="Adaptive Q-Table (α=0.02)", color="#5B8ED6")
    ax.axvline(x=midpoint, color='red', linestyle='--', label='Environment Degraded')
    ax.set_title("Adaptation Experiment (Moving Avg Reward)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    
    out_plot = os.path.join(results_dir, "adaptation_experiment.png")
    fig.savefig(out_plot, dpi=150)
    plt.close()
    
    out_qtable = os.path.join(results_dir, "adaptive_q_table.npy")
    np.save(out_qtable, qtable_adapt)
    
    print(f"Adaptation plot saved to {out_plot}")
    print(f"Adaptive Q-table saved to {out_qtable}")
    print(f"Frozen Avg Reward (Ep 500-1000): {np.mean(frozen_rewards[500:]):.2f}")
    print(f"Adaptive Avg Reward (Ep 500-1000): {np.mean(adapt_rewards[500:]):.2f}")

def main():
    cfg = load_config()
    results_dir = os.path.join(os.path.dirname(__file__), cfg.get("results_dir", "results"))
    ensure_dir(results_dir)
    
    task1_baseline_15seeds(cfg, results_dir)
    task2_heatmaps_grey(cfg, results_dir)
    task3_two_device_fairness(cfg)
    task4_adaptation(cfg, results_dir)

if __name__ == "__main__":
    main()
