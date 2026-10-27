"""
offload_env.py -- IoT Task-Offloading Environment  (Phase 1)
=============================================================
Behavioural guarantees (verified analytically in config comments):
  1. poor network + light task  -> LOCAL best
  2. heavy task + good network + short queue  -> EDGE best
  3. long edge queue  -> CLOUD or LOCAL beats EDGE
  4. low battery + decent network  -> offloading beats LOCAL
  5. heavy accuracy-critical task  -> CLOUD when EDGE queue congested

State tuple  (device-local):
    (battery_bin, network, task_type, edge_queue_bin, device_id)

No gym dependency.  Pure NumPy + standard-library only.
"""

import numpy as np

# ── Action constants ───────────────────────────────────────────────────────────
ACTION_LOCAL = 0
ACTION_EDGE  = 1
ACTION_CLOUD = 2
N_ACTIONS    = 3

# ── State dimension constants ─────────────────────────────────────────────────
NETWORK_POOR   = 0
NETWORK_MEDIUM = 1
NETWORK_GOOD   = 2
N_NETWORKS     = 3

TASK_LIGHT  = 0
TASK_MEDIUM = 1
TASK_HEAVY  = 2
N_TASKS     = 3

N_BATTERY_BINS    = 5
N_EDGE_QUEUE_BINS = 5
N_DEVICES         = 2   # upper bound; single-device mode uses device_id=0 only

# Flat state space:  5 * 3 * 3 * 5 * 2 = 450
STATE_SIZE = (N_BATTERY_BINS * N_NETWORKS * N_TASKS
              * N_EDGE_QUEUE_BINS * N_DEVICES)


# ── Binning helpers ────────────────────────────────────────────────────────────

def _bin_battery(battery: float, n_bins: int = N_BATTERY_BINS) -> int:
    """Map continuous battery [0, 100] to bin index [0, n_bins-1]."""
    battery = float(np.clip(battery, 0.0, 100.0))
    if battery >= 100.0:
        return n_bins - 1
    return int(battery / (100.0 / n_bins))


def _bin_queue(queue_depth: int, max_queue: int,
               n_bins: int = N_EDGE_QUEUE_BINS) -> int:
    """Map queue depth [0, max_queue] to bin index [0, n_bins-1]."""
    queue_depth = int(np.clip(queue_depth, 0, max_queue))
    if queue_depth >= max_queue:
        return n_bins - 1
    return int(queue_depth / (max_queue / n_bins))


def state_to_index(battery_bin: int, network: int, task: int,
                   queue_bin: int, device_id: int) -> int:
    """Convert state tuple to a unique flat integer index."""
    idx = (battery_bin * (N_NETWORKS * N_TASKS * N_EDGE_QUEUE_BINS * N_DEVICES)
           + network   * (N_TASKS * N_EDGE_QUEUE_BINS * N_DEVICES)
           + task      * (N_EDGE_QUEUE_BINS * N_DEVICES)
           + queue_bin * N_DEVICES
           + device_id)
    return int(idx)


def index_to_state(idx: int):
    """Reverse mapping: flat index -> (battery_bin, network, task, queue_bin, device_id)."""
    device_id   = idx % N_DEVICES;           idx //= N_DEVICES
    queue_bin   = idx % N_EDGE_QUEUE_BINS;   idx //= N_EDGE_QUEUE_BINS
    task        = idx % N_TASKS;             idx //= N_TASKS
    network     = idx % N_NETWORKS;          idx //= N_NETWORKS
    battery_bin = idx
    return battery_bin, network, task, queue_bin, device_id


# ── Per-device mutable state ───────────────────────────────────────────────────

class DeviceState:
    """Continuous state for a single ESP32 device."""

    def __init__(self, device_id: int, rng: np.random.Generator, cfg: dict):
        self.device_id = device_id
        self.rng       = rng
        self.cfg       = cfg
        self.battery: float = float(rng.uniform(50.0, 100.0))
        self.network: int   = int(rng.integers(0, N_NETWORKS))
        self.task_type: int = int(rng.integers(0, N_TASKS))

    def sample_task(self) -> None:
        self.task_type = int(self.rng.integers(0, N_TASKS))

    def advance_network(self) -> None:
        net_names = ["poor", "medium", "good"]
        probs     = self.cfg["network_transition"][net_names[self.network]]
        self.network = int(self.rng.choice(N_NETWORKS, p=probs))

    def get_state_tuple(self, queue_depth: int, max_queue: int):
        return (
            _bin_battery(self.battery),
            self.network,
            self.task_type,
            _bin_queue(queue_depth, max_queue),
            self.device_id,
        )

    def get_state_index(self, queue_depth: int, max_queue: int) -> int:
        return state_to_index(*self.get_state_tuple(queue_depth, max_queue))


# ── Main environment class ─────────────────────────────────────────────────────

class OffloadEnv:
    """
    Single-device or two-device IoT task-offloading environment.

    Single-device
    -------------
        env   = OffloadEnv(cfg)
        state = env.reset()                        # int
        ns, r, done, info = env.step(action)       # action in {0,1,2}

    Two-device
    ----------
        env    = OffloadEnv(cfg)
        states = env.reset()                       # list[int]
        ns, rs, done, info = env.step([a0, a1])
    """

    def __init__(self, cfg: dict):
        self.cfg        = cfg
        self.two_device = cfg.get("mode", "single") == "two_device"
        self.n_devices  = 2 if self.two_device else 1
        self.max_queue  = int(cfg.get("max_edge_queue", 20))

        # Build lookup arrays: shape (3 tasks, 3 actions)
        _t = ["light", "medium", "heavy"]
        self._energy   = np.array([cfg["energy"][t]   for t in _t], dtype=float)
        self._latency  = np.array([cfg["latency"][t]  for t in _t], dtype=float)
        self._accuracy = np.array([cfg["accuracy"][t] for t in _t], dtype=float)

        # Network latency multipliers
        self._net_mult_edge  = np.array(cfg["network_latency_mult"]["edge"],  dtype=float)
        self._net_mult_cloud = np.array(cfg["network_latency_mult"]["cloud"], dtype=float)

        # Queue latency multipliers  (indexed by queue_bin 0..4)
        self._queue_mult = np.array(cfg["queue_latency_mult"], dtype=float)

        # Poor-network energy penalty multiplier
        self._poor_energy_mult = float(cfg.get("poor_network_energy_mult", 1.4))

        # Reward weights
        self.w1 = float(cfg["weights"]["w1"])
        self.w2 = float(cfg["weights"]["w2"])
        self.w3 = float(cfg["weights"]["w3"])

        # Penalties
        self.pen_low_bat  = float(cfg["penalties"]["low_battery"])
        self.pen_death    = float(cfg["penalties"]["death"])
        self.pen_fairness = float(cfg["penalties"]["fairness"])

        # Queue dynamics
        self.q_arrive = float(cfg.get("queue_arrival_prob",   0.35))
        self.q_depart = float(cfg.get("queue_departure_prob", 0.55))

        # Gaussian noise std on lookup values (5 % of value)
        self._noise_std = 0.05

        self.edge_queue: int  = 0
        self._step_count: int = 0
        self._max_steps: int  = int(cfg.get("steps_per_episode", 50))

        seed = cfg.get("seed", None)
        self.rng = np.random.default_rng(seed)

        self.devices: list = [
            DeviceState(i, self.rng, cfg) for i in range(self.n_devices)
        ]

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self):
        self.edge_queue  = int(self.rng.integers(0, self.max_queue // 4 + 1))
        self._step_count = 0
        for dev in self.devices:
            dev.battery   = float(self.rng.uniform(50.0, 100.0))
            dev.network   = int(self.rng.integers(0, N_NETWORKS))
            dev.sample_task()
        if self.two_device:
            return [d.get_state_index(self.edge_queue, self.max_queue)
                    for d in self.devices]
        return self.devices[0].get_state_index(self.edge_queue, self.max_queue)

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, actions):
        if not self.two_device:
            actions = [actions]
        actions = [int(a) for a in actions]
        self._step_count += 1

        rewards  = []
        all_done = False
        info     = {"energy": [], "latency": [], "accuracy": []}

        for i, dev in enumerate(self.devices):
            r, done_i, si = self._device_step(dev, actions[i])
            rewards.append(r)
            if done_i:
                all_done = True
            info["energy"].append(si["energy"])
            info["latency"].append(si["latency"])
            info["accuracy"].append(si["accuracy"])

        # Fairness penalty (two-device)
        if self.two_device:
            edge_use = [1 if a == ACTION_EDGE else 0 for a in actions]
            fp = self.pen_fairness * abs(edge_use[0] - edge_use[1])
            rewards = [r - fp for r in rewards]
            info["fairness_penalty"] = fp
        else:
            info["fairness_penalty"] = 0.0

        # Update shared edge queue
        edge_jobs = sum(1 for a in actions if a == ACTION_EDGE)
        self.edge_queue = min(self.edge_queue + edge_jobs, self.max_queue)
        if self.rng.random() < self.q_arrive:
            self.edge_queue = min(self.edge_queue + 1, self.max_queue)
        if self.rng.random() < self.q_depart and self.edge_queue > 0:
            self.edge_queue -= 1

        # Advance per-device dynamics
        for dev in self.devices:
            dev.advance_network()
            dev.sample_task()

        if self._step_count >= self._max_steps:
            all_done = True

        next_states = [d.get_state_index(self.edge_queue, self.max_queue)
                       for d in self.devices]

        if self.two_device:
            return next_states, rewards, all_done, info
        return next_states[0], rewards[0], all_done, info

    # ── Internal per-device step ───────────────────────────────────────────────
    def _device_step(self, dev: DeviceState, action: int):
        task = dev.task_type
        net  = dev.network

        # ── Base values from lookup table ──────────────────────────────────
        base_energy   = float(self._energy[task, action])
        base_latency  = float(self._latency[task, action])
        base_accuracy = float(self._accuracy[task, action])

        # ── Gaussian noise (5 %) ───────────────────────────────────────────
        energy   = float(np.clip(base_energy   * (1 + self.rng.normal(0, self._noise_std)), 1e-6, None))
        latency  = float(np.clip(base_latency  * (1 + self.rng.normal(0, self._noise_std)), 1e-6, None))
        accuracy = float(np.clip(base_accuracy * (1 + self.rng.normal(0, self._noise_std)), 0.0,  1.0))

        # ── Network latency multiplier (EDGE and CLOUD only) ───────────────
        if action == ACTION_EDGE:
            latency *= float(self._net_mult_edge[net])
            # Queue-depth multiplier (EDGE only) -- KEY for scenario 3 & 5
            q_bin    = _bin_queue(dev.cfg.get("edge_queue_proxy", self.edge_queue)
                                  if hasattr(dev.cfg, "get") else self.edge_queue,
                                  self.max_queue)
            q_bin    = _bin_queue(self.edge_queue, self.max_queue)
            latency *= float(self._queue_mult[q_bin])

        elif action == ACTION_CLOUD:
            latency *= float(self._net_mult_cloud[net])

        # ── Poor-network energy overhead (retransmissions) ─────────────────
        if action in (ACTION_EDGE, ACTION_CLOUD) and net == NETWORK_POOR:
            energy *= self._poor_energy_mult

        # ── Battery drain ──────────────────────────────────────────────────
        dev.battery = max(0.0, dev.battery - energy * 100.0)

        # ── Reward ─────────────────────────────────────────────────────────
        reward = (
            - self.w1 * energy
            + self.w2 * (1.0 / latency)
            + self.w3 * accuracy
        )

        done = False

        # Low-battery penalty (LOCAL chosen while battery < 20 %)
        if action == ACTION_LOCAL and dev.battery < 20.0:
            reward -= self.pen_low_bat

        # Battery death
        if dev.battery <= 0.0:
            reward -= self.pen_death
            done    = True

        return reward, done, {"energy": energy, "latency": latency, "accuracy": accuracy}

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def state_size(self) -> int:
        return STATE_SIZE

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    def get_current_states(self) -> list:
        return [d.get_state_index(self.edge_queue, self.max_queue)
                for d in self.devices]

    def deterministic_reward(self, battery_bin: int, network: int,
                              task: int, queue_bin: int, action: int) -> float:
        """
        Compute reward deterministically (no noise, no battery drain) for a
        given state and action.  Useful for verifying scenario requirements
        and unit tests.
        """
        energy   = float(self._energy[task, action])
        latency  = float(self._latency[task, action])
        accuracy = float(self._accuracy[task, action])

        if action == ACTION_EDGE:
            latency *= float(self._net_mult_edge[network])
            latency *= float(self._queue_mult[queue_bin])
        elif action == ACTION_CLOUD:
            latency *= float(self._net_mult_cloud[network])

        if action in (ACTION_EDGE, ACTION_CLOUD) and network == NETWORK_POOR:
            energy *= self._poor_energy_mult

        reward = -self.w1 * energy + self.w2 * (1.0 / latency) + self.w3 * accuracy

        # Simulate low-battery penalty for battery_bin == 0 (<20%)
        if action == ACTION_LOCAL and battery_bin == 0:
            reward -= self.pen_low_bat

        return reward
