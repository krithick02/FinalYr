import os
import sys
import numpy as np
import scipy.stats as stats
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from offload_env import OffloadEnv
from evaluate import policy_static_threshold, run_policy, choose_action
from train_qlearning import train_qtable
from utils import load_config

def calc_95_ci(data):
    mean = np.mean(data)
    if np.std(data) == 0:
        return mean, mean, mean
    se = stats.sem(data)
    ci = se * stats.t.ppf((1 + 0.95) / 2., len(data)-1)
    return mean, mean - ci, mean + ci

# ── Task 1: 15 Independent Q-Tables ───────────────────────────────────────────
def task1_15_independent_qtables(cfg):
    print("\n--- TASK 1: Q-learning vs Static-Threshold (15 Independent Seeds) ---")
    
    train_seeds = [100 + i for i in range(15)]
    eval_seeds = [500 + i for i in range(15)]
    n_eval = 200
    
    ql_means = []
    st_means = []
    
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["episodes"] = 4000 # To keep runtime reasonable
    
    for ts, es in zip(train_seeds, eval_seeds):
        # Train
        cfg_copy["seed"] = ts
        qtable, _ = train_qtable(cfg_copy, verbose=False)
        
        # Eval
        eval_cfg = copy.deepcopy(cfg)
        eval_cfg["seed"] = es
        env = OffloadEnv(eval_cfg)
        rng = np.random.default_rng(es)
        
        def ql_policy(state, env_obj):
            return choose_action(qtable, state, epsilon=0.0, rng=rng)
            
        res_ql = run_policy(env, ql_policy, n_eval, rng)
        res_st = run_policy(env, policy_static_threshold, n_eval, rng)
        
        ql_means.append(np.mean(res_ql["rewards"]))
        st_means.append(np.mean(res_st["rewards"]))
        
    print(f"Per-seed QL mean rewards: {', '.join([f'{x:.2f}' for x in ql_means])}")
    print(f"Per-seed ST mean rewards: {', '.join([f'{x:.2f}' for x in st_means])}")
    
    ql_means = np.array(ql_means)
    st_means = np.array(st_means)
    
    t_stat, p_val = stats.ttest_ind(ql_means, st_means, equal_var=False)
    
    mean_diff = np.mean(ql_means) - np.mean(st_means)
    se_diff = np.sqrt(np.var(ql_means, ddof=1)/15 + np.var(st_means, ddof=1)/15)
    ci_low = mean_diff - stats.t.ppf(0.975, 14) * se_diff 
    ci_high = mean_diff + stats.t.ppf(0.975, 14) * se_diff
    
    print(f"QL Means: {np.mean(ql_means):.2f} ± {np.std(ql_means):.2f}")
    print(f"ST Means: {np.mean(st_means):.2f} ± {np.std(st_means):.2f}")
    print(f"95% CI of Difference (QL - ST): [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"Welch t-test p-value: {p_val:.4e}")

# ── Task 2: Adaptation Experiment (15 Seeds, 4x Edge Latency) ─────────────────
def task2_adaptation(cfg):
    print("\n--- TASK 2: Adaptation (15 Seeds, 4x Edge Latency) ---")
    
    train_cfg = copy.deepcopy(cfg)
    train_cfg["episodes"] = 4000
    train_cfg["seed"] = 42
    base_qtable, _ = train_qtable(train_cfg, verbose=False)
    
    seeds = [1000 + i for i in range(15)]
    total_episodes = 1000
    midpoint = 500
    alpha = 0.02
    gamma = cfg["gamma"]
    epsilon = 0.02 # Action noise for both
    
    frozen_post_means = []
    adapt_post_means = []
    
    for s in seeds:
        qtable_frozen = np.copy(base_qtable)
        qtable_adapt = np.copy(base_qtable)
        
        # Frozen Run
        env = OffloadEnv(copy.deepcopy(cfg))
        env.rng = np.random.default_rng(s)
        frozen_r = []
        for ep in range(total_episodes):
            if ep == midpoint:
                env._net_mult_edge *= 4.0 # 4x slower edge latency permanently
                    
            state = env.reset()
            ep_r = 0
            while True:
                a = choose_action(qtable_frozen, state, epsilon, env.rng)
                ns, r, done, _ = env.step(a)
                ep_r += r
                state = ns
                if done: break
            frozen_r.append(ep_r)
            
        # Adapt Run
        env = OffloadEnv(copy.deepcopy(cfg))
        env.rng = np.random.default_rng(s)
        adapt_r = []
        for ep in range(total_episodes):
            if ep == midpoint:
                env._net_mult_edge *= 4.0 # 4x slower edge latency permanently
                    
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
            
        frozen_post_means.append(np.mean(frozen_r[midpoint:]))
        adapt_post_means.append(np.mean(adapt_r[midpoint:]))

    f_mean, f_ci_low, f_ci_high = calc_95_ci(frozen_post_means)
    a_mean, a_ci_low, a_ci_high = calc_95_ci(adapt_post_means)
    
    print(f"Frozen Post-Change Mean Reward: {f_mean:.2f} (95% CI: [{f_ci_low:.2f}, {f_ci_high:.2f}])")
    print(f"Adaptive Post-Change Mean Reward: {a_mean:.2f} (95% CI: [{a_ci_low:.2f}, {a_ci_high:.2f}])")

# ── Task 3: Fairness vs No Fairness (15 Seeds, Abs Diff of Mean Latencies) ────
def task3_fairness_15seeds(cfg):
    print("\n--- TASK 3: Fairness (15 Seeds) ---")
    seeds = [3000 + i for i in range(15)]
    n_eval = 200
    
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["mode"] = "two_device"
    cfg_copy["episodes"] = 4000
    
    cfg_copy["penalties"]["fairness"] = 1.0
    cfg_copy["seed"] = 42
    qtables_fair, _ = train_qtable(cfg_copy, verbose=False)
    
    cfg_copy["penalties"]["fairness"] = 0.0
    cfg_copy["seed"] = 42
    qtables_unfair, _ = train_qtable(cfg_copy, verbose=False)
    
    fair_gaps = []
    unfair_gaps = []
    
    for s in seeds:
        # Eval Fair
        env_f = OffloadEnv(copy.deepcopy(cfg_copy))
        env_f.pen_fairness = 1.0
        env_f.rng = np.random.default_rng(s)
        
        dev0_lat_f = []
        dev1_lat_f = []
        for _ in range(n_eval):
            states = env_f.reset()
            while True:
                a0 = choose_action(qtables_fair[0], states[0], 0.0, env_f.rng)
                a1 = choose_action(qtables_fair[1], states[1], 0.0, env_f.rng)
                ns, rs, done, info = env_f.step([a0, a1])
                dev0_lat_f.append(info["latency"][0])
                dev1_lat_f.append(info["latency"][1])
                states = ns
                if done: break
        # absolute difference of the mean latency of device 0 and device 1
        gap_f = abs(np.mean(dev0_lat_f) - np.mean(dev1_lat_f)) * 1000 # ms
        fair_gaps.append(gap_f)
        
        # Eval Unfair
        env_u = OffloadEnv(copy.deepcopy(cfg_copy))
        env_u.pen_fairness = 0.0
        env_u.rng = np.random.default_rng(s)
        
        dev0_lat_u = []
        dev1_lat_u = []
        for _ in range(n_eval):
            states = env_u.reset()
            while True:
                a0 = choose_action(qtables_unfair[0], states[0], 0.0, env_u.rng)
                a1 = choose_action(qtables_unfair[1], states[1], 0.0, env_u.rng)
                ns, rs, done, info = env_u.step([a0, a1])
                dev0_lat_u.append(info["latency"][0])
                dev1_lat_u.append(info["latency"][1])
                states = ns
                if done: break
        gap_u = abs(np.mean(dev0_lat_u) - np.mean(dev1_lat_u)) * 1000 # ms
        unfair_gaps.append(gap_u)
        
    f_mean, f_low, f_high = calc_95_ci(fair_gaps)
    u_mean, u_low, u_high = calc_95_ci(unfair_gaps)
    
    print(f"Fairness Penalty ON (1.0): Absolute Latency Gap = {f_mean:.2f} ms (95% CI: [{f_low:.2f}, {f_high:.2f}])")
    print(f"Fairness Penalty OFF (0.0): Absolute Latency Gap = {u_mean:.2f} ms (95% CI: [{u_low:.2f}, {u_high:.2f}])")
    
    if np.std(fair_gaps) == 0:
        print("Note: Fairness ON std is zero because all runs yielded identical gap values.")
    if np.std(unfair_gaps) == 0:
        print("Note: Fairness OFF std is zero because all runs yielded identical gap values.")

def main():
    cfg = load_config()
    task1_15_independent_qtables(cfg)
    task2_adaptation(cfg)
    task3_fairness_15seeds(cfg)

if __name__ == "__main__":
    main()
