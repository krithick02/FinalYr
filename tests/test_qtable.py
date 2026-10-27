"""
tests/test_qtable.py -- Q-Learning Logic and Policy Validation (Phase 1)
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))

from train_qlearning import train_qtable
from evaluate import (policy_always_local, policy_always_edge, 
                      policy_always_cloud, policy_static_threshold, run_policy, choose_action)
from offload_env import OffloadEnv
from utils import load_config

@pytest.fixture(scope="module")
def cfg():
    c = load_config(os.path.join(os.path.dirname(__file__), "..", "sim", "config.yaml"))
    # for tests, we can use fewer episodes to speed up, but enough to learn a decent policy
    # 500 is enough to beat naive baselines.
    c["episodes"] = 500
    return c

def test_learned_policy_not_constant(cfg):
    """(a) the learned policy is not constant across states"""
    # Use fixed seed for reproducibility
    cfg["seed"] = 42
    qtable, _ = train_qtable(cfg, verbose=False)
    
    # Extract greedy actions for all states
    best_actions = np.argmax(qtable, axis=1)
    
    unique_actions = np.unique(best_actions)
    # The agent should learn to use at least 2 different actions across the 450 states
    assert len(unique_actions) > 1, f"Policy is constant! Always choosing action {unique_actions}"

def test_trained_agent_beats_baselines(cfg):
    """(b) the trained agent beats every baseline on mean reward"""
    cfg["seed"] = 123
    cfg["episodes"] = 1500 # Give it enough time to learn well for this test
    qtable, _ = train_qtable(cfg, verbose=False)
    
    env = OffloadEnv(cfg)
    rng = np.random.default_rng(123)
    n_eval = 50
    
    def ql_policy(state, env_obj):
        return choose_action(qtable, state, epsilon=0.0, rng=rng)
        
    res_ql = run_policy(env, ql_policy, n_eval, rng)
    mean_ql = res_ql["rewards"].mean()
    
    baselines = [
        ("Always-Local", policy_always_local),
        ("Always-Edge", policy_always_edge),
        ("Always-Cloud", policy_always_cloud),
        ("Static-Threshold", policy_static_threshold)
    ]
    
    for name, policy_fn in baselines:
        res_base = run_policy(env, policy_fn, n_eval, rng)
        mean_base = res_base["rewards"].mean()
        assert mean_ql > mean_base, f"Q-Learning ({mean_ql}) failed to beat {name} ({mean_base})"
