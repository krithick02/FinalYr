"""
tests/test_env.py -- Environment shape and behaviour validation
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))

from offload_env import (
    OffloadEnv,
    state_to_index,
    index_to_state,
    STATE_SIZE,
    N_ACTIONS,
    ACTION_LOCAL,
    ACTION_EDGE,
    ACTION_CLOUD,
    NETWORK_POOR, NETWORK_MEDIUM, NETWORK_GOOD,
    TASK_LIGHT, TASK_MEDIUM, TASK_HEAVY,
    N_BATTERY_BINS, N_EDGE_QUEUE_BINS, N_NETWORKS, N_TASKS, N_DEVICES
)
from utils import load_config

@pytest.fixture(scope="module")
def cfg():
    return load_config(os.path.join(os.path.dirname(__file__), "..", "sim", "config.yaml"))

@pytest.fixture
def env(cfg):
    return OffloadEnv(cfg)

# ── Env Shapes & State Mappings ────────────────────────────────────────────────

def test_state_mapping_reversible():
    for bat in range(N_BATTERY_BINS):
        for net in range(N_NETWORKS):
            for task in range(N_TASKS):
                for q in range(N_EDGE_QUEUE_BINS):
                    for dev in range(2):
                        idx = state_to_index(bat, net, task, q, dev)
                        assert 0 <= idx < STATE_SIZE
                        rb, rn, rt, rq, rd = index_to_state(idx)
                        assert (bat, net, task, q, dev) == (rb, rn, rt, rq, rd)

def test_env_shapes_are_correct(env):
    assert env.state_size == STATE_SIZE
    assert env.n_actions == N_ACTIONS

    state = env.reset()
    assert isinstance(state, int)
    assert 0 <= state < STATE_SIZE
    
    ns, r, done, info = env.step(ACTION_LOCAL)
    assert isinstance(ns, int)
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert "energy" in info and "latency" in info and "accuracy" in info

# ── Env Behavioural Validation (Phase 1 conditions) ───────────────────────────

def test_condition_poor_net_light_task_local_best(env):
    """poor network + light task -> local is best"""
    # Try all actions, compute deterministic reward
    r_local = env.deterministic_reward(battery_bin=4, network=NETWORK_POOR, task=TASK_LIGHT, queue_bin=0, action=ACTION_LOCAL)
    r_edge  = env.deterministic_reward(battery_bin=4, network=NETWORK_POOR, task=TASK_LIGHT, queue_bin=0, action=ACTION_EDGE)
    r_cloud = env.deterministic_reward(battery_bin=4, network=NETWORK_POOR, task=TASK_LIGHT, queue_bin=0, action=ACTION_CLOUD)
    
    assert r_local > r_edge
    assert r_local > r_cloud

def test_condition_heavy_task_good_net_short_queue_edge_best(env):
    """heavy task + good network + short queue -> edge is best"""
    r_local = env.deterministic_reward(battery_bin=4, network=NETWORK_GOOD, task=TASK_HEAVY, queue_bin=0, action=ACTION_LOCAL)
    r_edge  = env.deterministic_reward(battery_bin=4, network=NETWORK_GOOD, task=TASK_HEAVY, queue_bin=0, action=ACTION_EDGE)
    r_cloud = env.deterministic_reward(battery_bin=4, network=NETWORK_GOOD, task=TASK_HEAVY, queue_bin=0, action=ACTION_CLOUD)
    
    assert r_edge > r_local
    assert r_edge > r_cloud

def test_condition_long_edge_queue(env):
    """long edge queue -> cloud or local beats edge"""
    r_local = env.deterministic_reward(battery_bin=4, network=NETWORK_GOOD, task=TASK_HEAVY, queue_bin=4, action=ACTION_LOCAL)
    r_edge  = env.deterministic_reward(battery_bin=4, network=NETWORK_GOOD, task=TASK_HEAVY, queue_bin=4, action=ACTION_EDGE)
    r_cloud = env.deterministic_reward(battery_bin=4, network=NETWORK_GOOD, task=TASK_HEAVY, queue_bin=4, action=ACTION_CLOUD)
    
    # either cloud > edge or local > edge
    assert (r_cloud > r_edge) or (r_local > r_edge)

def test_condition_low_battery_decent_net(env):
    """low battery + decent network -> offloading beats local"""
    # battery_bin = 0 (< 20%) triggers low battery penalty for LOCAL
    r_local = env.deterministic_reward(battery_bin=0, network=NETWORK_MEDIUM, task=TASK_MEDIUM, queue_bin=0, action=ACTION_LOCAL)
    r_edge  = env.deterministic_reward(battery_bin=0, network=NETWORK_MEDIUM, task=TASK_MEDIUM, queue_bin=0, action=ACTION_EDGE)
    r_cloud = env.deterministic_reward(battery_bin=0, network=NETWORK_MEDIUM, task=TASK_MEDIUM, queue_bin=0, action=ACTION_CLOUD)
    
    assert max(r_edge, r_cloud) > r_local

def test_condition_heavy_accuracy_cloud_when_latency_allows(env):
    """heavy, accuracy-critical task -> cloud when latency allows (e.g. good network, long edge queue)"""
    r_edge  = env.deterministic_reward(battery_bin=4, network=NETWORK_GOOD, task=TASK_HEAVY, queue_bin=4, action=ACTION_EDGE)
    r_cloud = env.deterministic_reward(battery_bin=4, network=NETWORK_GOOD, task=TASK_HEAVY, queue_bin=4, action=ACTION_CLOUD)
    assert r_cloud > r_edge
