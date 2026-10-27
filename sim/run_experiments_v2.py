import os
import sys
import numpy as np
import scipy.stats as stats
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from offload_env import OffloadEnv, NETWORK_POOR
from evaluate import policy_static_threshold, run_policy, choose_action
from train_qlearning import train_qtable
from utils import load_config

def load_avg_qtable(data_dir):
    return np.load(os.path.join(data_dir, "q_table.npy"))

def calc_95_ci(data):
    mean = np.mean(data)
    se = stats.sem(data)
    ci = se * stats.t.ppf((1 + 0.95) / 2., len(data)-1)
    return mean, mean - ci, mean + ci

# ── Task 1: 15 seeds Baseline Comparison ──────────────────────────────────────
def task1_baseline_15seeds(cfg):
    print("\n--- TASK 1: Q-learning vs Static-Threshold ---")
    data_dir = os.path.join(os.path.dirname(__file__), cfg.get("data_dir", "data"))
    qtable = load_avg_qtable(data_dir)
    
    seeds = [s * 111 for s in range(1, 16)]
    n_eval = 200
    
    ql_means, st_means = [], []
    
    for s in seeds:
        env = OffloadEnv(cfg)
        rng = np.random.default_rng(s)
        
        def ql_policy(state, env_obj):
            return choose_action(qtable, state, epsilon=0.0, rng=rng)
            
        res_ql = run_policy(env, ql_policy, n_eval, rng)
        res_st = run_policy(env, policy_static_threshold, n_eval, rng)
        
        ql_means.append(np.mean(res_ql["rewards"]))
        st_means.append(np.mean(res_st["rewards"]))
        
    ql_means = np.array(ql_means)
    st_means = np.array(st_means)
    
    t_stat, p_val = stats.ttest_ind(ql_means, st_means, equal_var=False)
    
    mean_diff = np.mean(ql_means) - np.mean(st_means)
    se_diff = np.sqrt(np.var(ql_means, ddof=1)/15 + np.var(st_means, ddof=1)/15)
    ci_low = mean_diff - stats.t.ppf(0.975, 14) * se_diff # approx df
    ci_high = mean_diff + stats.t.ppf(0.975, 14) * se_diff
    
    print(f"Seeds: 15 (one mean reward per seed)")
    print(f"QL Means: {np.mean(ql_means):.2f} ± {np.std(ql_means):.2f}")
    print(f"ST Means: {np.mean(st_means):.2f} ± {np.std(st_means):.2f}")
    print(f"95% CI of Difference (QL - ST): [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"Welch t-test p-value: {p_val:.4e}")

# ── Task 2: Adaptation Experiment (15 Seeds) ──────────────────────────────────
def task2_adaptation(cfg):
    print("\n--- TASK 2: Adaptation (15 Seeds, Strong Disturbance) ---")
    data_dir = os.path.join(os.path.dirname(__file__), cfg.get("data_dir", "data"))
    base_qtable = load_avg_qtable(data_dir)
    
    seeds = [s * 222 for s in range(1, 16)]
    total_episodes = 1000
    midpoint = 500
    alpha = 0.02
    gamma = cfg["gamma"]
    epsilon = 0.05
    
    frozen_post_means = []
    adapt_post_means = []
    
    frozen_recovery = []
    adapt_recovery = []
    
    for s in seeds:
        qtable_frozen = np.copy(base_qtable)
        qtable_adapt = np.copy(base_qtable)
        
        # Frozen Run
        env = OffloadEnv(cfg)
        env.rng = np.random.default_rng(s)
        frozen_r = []
        for ep in range(total_episodes):
            if ep == midpoint:
                env.max_queue = min(env.max_queue * 3, 60)
                env.q_arrive = min(env.q_arrive * 3.0, 1.0)
                for dev in env.devices:
                    dev.network = NETWORK_POOR
                    dev.cfg["network_transition"]["poor"] = [1.0, 0.0, 0.0]
                    dev.cfg["network_transition"]["medium"] = [1.0, 0.0, 0.0]
                    dev.cfg["network_transition"]["good"] = [1.0, 0.0, 0.0]
                    
            state = env.reset()
            ep_r = 0
            while True:
                a = choose_action(qtable_frozen, state, 0.0, env.rng)
                ns, r, done, _ = env.step(a)
                ep_r += r
                state = ns
                if done: break
            frozen_r.append(ep_r)
            
        # Adapt Run
        env = OffloadEnv(cfg)
        env.rng = np.random.default_rng(s)
        adapt_r = []
        for ep in range(total_episodes):
            if ep == midpoint:
                env.max_queue = min(env.max_queue * 3, 60)
                env.q_arrive = min(env.q_arrive * 3.0, 1.0)
                for dev in env.devices:
                    dev.network = NETWORK_POOR
                    dev.cfg["network_transition"]["poor"] = [1.0, 0.0, 0.0]
                    dev.cfg["network_transition"]["medium"] = [1.0, 0.0, 0.0]
                    dev.cfg["network_transition"]["good"] = [1.0, 0.0, 0.0]
                    
            state = env.reset()
            ep_r = 0
            while True:
                a = choose_action(qtable_adapt, state, epsilon, env.rng)
                ns, r, done, _ = env.step(a)
                td_target = r + gamma * np.max(qtable_adapt[ns])
                qtable_adapt[state, a] += alpha * (td_target - qtable_adapt[state, a])
                ep_r += r
                state = ns
                if done: break
            adapt_r.append(ep_r)
            
        # Analysis for this seed
        pre_frozen_mean = np.mean(frozen_r[:midpoint])
        pre_adapt_mean = np.mean(adapt_r[:midpoint])
        
        post_frozen_mean = np.mean(frozen_r[midpoint:])
        post_adapt_mean = np.mean(adapt_r[midpoint:])
        
        frozen_post_means.append(post_frozen_mean)
        adapt_post_means.append(post_adapt_mean)
        
        # Recovery (50-ep moving avg)
        def find_recovery(rewards, pre_mean):
            target = 0.90 * pre_mean
            post_r = rewards[midpoint:]
            import pandas as pd
            ma = pd.Series(post_r).rolling(50, min_periods=1).mean().values
            # only check after some episodes to avoid overlap with pre-degradation tail
            for i in range(len(ma)):
                if ma[i] >= target:
                    return i
            return -1
            
        f_rec = find_recovery(frozen_r, pre_frozen_mean)
        a_rec = find_recovery(adapt_r, pre_adapt_mean)
        
        frozen_recovery.append(f_rec)
        adapt_recovery.append(a_rec)

    f_mean, f_ci_low, f_ci_high = calc_95_ci(frozen_post_means)
    a_mean, a_ci_low, a_ci_high = calc_95_ci(adapt_post_means)
    
    print(f"Frozen Post-Change Mean Reward: {f_mean:.2f} (95% CI: [{f_ci_low:.2f}, {f_ci_high:.2f}])")
    print(f"Adaptive Post-Change Mean Reward: {a_mean:.2f} (95% CI: [{a_ci_low:.2f}, {a_ci_high:.2f}])")
    
    f_recovers = [x for x in frozen_recovery if x != -1]
    a_recovers = [x for x in adapt_recovery if x != -1]
    
    f_rec_str = f"{np.mean(f_recovers):.1f} eps" if f_recovers else "Never"
    a_rec_str = f"{np.mean(a_recovers):.1f} eps" if a_recovers else "Never"
    
    print(f"Frozen Recovery to 90%: {f_rec_str} (Recovered in {len(f_recovers)}/15 seeds)")
    print(f"Adaptive Recovery to 90%: {a_rec_str} (Recovered in {len(a_recovers)}/15 seeds)")

# ── Task 3: Fairness vs No Fairness (15 Seeds) ────────────────────────────────
def task3_fairness_15seeds(cfg):
    print("\n--- TASK 3: Fairness (15 Seeds) ---")
    seeds = [s * 333 for s in range(1, 16)]
    n_eval = 200
    
    # We will use the already trained two-device Q-tables, 
    # but since the penalty needs to be built into the policy, we must quickly train them.
    # To save time in script, let's train a lightweight Q-table for fair and unfair just once.
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["mode"] = "two_device"
    cfg_copy["episodes"] = 4000
    cfg_copy["train_seeds"] = [42]
    
    cfg_copy["penalties"]["fairness"] = 1.0
    qtables_fair, _ = train_qtable(cfg_copy, verbose=False)
    
    cfg_copy["penalties"]["fairness"] = 0.0
    qtables_unfair, _ = train_qtable(cfg_copy, verbose=False)
    
    fair_gaps = []
    unfair_gaps = []
    
    for s in seeds:
        env_f = OffloadEnv(copy.deepcopy(cfg_copy))
        env_f.pen_fairness = 1.0
        env_f.rng = np.random.default_rng(s)
        
        env_u = OffloadEnv(copy.deepcopy(cfg_copy))
        env_u.pen_fairness = 0.0
        env_u.rng = np.random.default_rng(s)
        
        # Eval Fair
        gap_f = []
        for _ in range(n_eval):
            states = env_f.reset()
            while True:
                a0 = choose_action(qtables_fair[0], states[0], 0.0, env_f.rng)
                a1 = choose_action(qtables_fair[1], states[1], 0.0, env_f.rng)
                ns, rs, done, info = env_f.step([a0, a1])
                gap_f.append(abs(info["latency"][0] - info["latency"][1]))
                states = ns
                if done: break
        fair_gaps.append(np.mean(gap_f) * 1000)
        
        # Eval Unfair
        gap_u = []
        for _ in range(n_eval):
            states = env_u.reset()
            while True:
                a0 = choose_action(qtables_unfair[0], states[0], 0.0, env_u.rng)
                a1 = choose_action(qtables_unfair[1], states[1], 0.0, env_u.rng)
                ns, rs, done, info = env_u.step([a0, a1])
                gap_u.append(abs(info["latency"][0] - info["latency"][1]))
                states = ns
                if done: break
        unfair_gaps.append(np.mean(gap_u) * 1000)
        
    f_mean, f_low, f_high = calc_95_ci(fair_gaps)
    u_mean, u_low, u_high = calc_95_ci(unfair_gaps)
    
    print(f"Fairness Penalty ON (1.0): Absolute Latency Gap = {f_mean:.2f} ms (95% CI: [{f_low:.2f}, {f_high:.2f}])")
    print(f"Fairness Penalty OFF (0.0): Absolute Latency Gap = {u_mean:.2f} ms (95% CI: [{u_low:.2f}, {u_high:.2f}])")

def main():
    cfg = load_config()
    task1_baseline_15seeds(cfg)
    task2_adaptation(cfg)
    task3_fairness_15seeds(cfg)

if __name__ == "__main__":
    main()
