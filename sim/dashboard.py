"""
dashboard.py - IoT Task-Offloading Visual Dashboard (Premium UI)
Run with: streamlit run sim/dashboard.py
"""
from __future__ import annotations
import os, sys, time, copy, json
from typing import Callable
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)

from offload_env import (
    OffloadEnv, STATE_SIZE, N_ACTIONS,
    ACTION_LOCAL, ACTION_EDGE, ACTION_CLOUD,
    N_BATTERY_BINS, N_NETWORKS, N_TASKS, N_EDGE_QUEUE_BINS,
    index_to_state, state_to_index,
    NETWORK_POOR, NETWORK_MEDIUM, NETWORK_GOOD,
    TASK_LIGHT, TASK_MEDIUM, TASK_HEAVY,
)
from utils import load_config

# ── constants ────────────────────────────────────────────────────────────────
ACTION_NAMES  = ["LOCAL", "EDGE", "CLOUD"]
ACTION_COLORS = {"LOCAL": "#FF6B35", "EDGE": "#4FC3F7", "CLOUD": "#69FF47"}
ACTION_GLOW   = {"LOCAL": "#FF6B3540", "EDGE": "#4FC3F740", "CLOUD": "#69FF4740"}
NET_NAMES     = ["Poor", "Medium", "Good"]
TASK_NAMES    = ["Light", "Medium", "Heavy"]
BAT_LABELS    = ["<20%", "20-40%", "40-60%", "60-80%", ">80%"]
POLICY_NAMES  = [
    "Trained Q-Learning", "Always-Local", "Always-Edge",
    "Always-Cloud", "Static-Threshold",
]

# ── theme ─────────────────────────────────────────────────────────────────────
BG      = "#050810"
SURFACE = "#0D1526"
SURF2   = "#111D35"
TEXT    = "#E8F4FD"
MUTED   = "#7B9BB8"
ACCENT  = "#4FC3F7"
ACCENT2 = "#9C7FFF"
GREEN   = "#69FF47"
ORANGE  = "#FF6B35"

# ── CSS ───────────────────────────────────────────────────────────────────────
def _css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif; box-sizing: border-box; }

.stApp {
  background: radial-gradient(ellipse 80% 60% at 15% 10%, #0A1628 0%, #050810 55%);
  color: #E8F4FD;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2rem 2rem 2rem !important; max-width: 1500px; }

/* ── HEADER ─────────────────────────────────────────────── */
.dash-header {
  background: linear-gradient(135deg, rgba(79,195,247,0.1) 0%, rgba(156,127,255,0.07) 100%);
  border: 1px solid rgba(79,195,247,0.18);
  border-radius: 18px; padding: 22px 30px; margin-bottom: 22px;
  backdrop-filter: blur(20px); position: relative; overflow: hidden;
}
.dash-header::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent 0%, #4FC3F7 40%, #9C7FFF 70%, transparent 100%);
}
.dash-header::after {
  content: ""; position: absolute; bottom: 0; right: 0;
  width: 300px; height: 300px; border-radius: 50%;
  background: radial-gradient(circle, rgba(156,127,255,0.06) 0%, transparent 70%);
  pointer-events: none;
}
.dash-title {
  font-size: 1.8rem; font-weight: 800; margin: 0 0 6px 0;
  background: linear-gradient(135deg, #E8F4FD 0%, #4FC3F7 45%, #9C7FFF 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.dash-sub { color: #7B9BB8; font-size: 0.83rem; margin: 0; font-weight: 400; }
.stat-pill {
  display: inline-block; background: rgba(79,195,247,0.1);
  border: 1px solid rgba(79,195,247,0.2); border-radius: 20px;
  padding: 3px 12px; font-size: 0.75rem; color: #4FC3F7;
  font-family: 'JetBrains Mono', monospace; margin-top: 8px; margin-right: 6px;
}
.pulse-dot {
  width: 9px; height: 9px; border-radius: 50%; background: #69FF47;
  display: inline-block; margin-right: 8px; vertical-align: middle;
  box-shadow: 0 0 6px #69FF47, 0 0 14px rgba(105,255,71,0.5);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.45; transform: scale(1.4); }
}

/* ── TABS ──────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 5px; background: rgba(13,21,38,0.85); border-radius: 13px;
  padding: 5px; border: 1px solid rgba(255,255,255,0.06); backdrop-filter: blur(12px);
}
.stTabs [data-baseweb="tab"] {
  background: transparent; color: #7B9BB8; border-radius: 10px;
  padding: 9px 22px; font-weight: 500; font-size: 0.85rem;
  border: none; transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #E8F4FD; background: rgba(255,255,255,0.04); }
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, #4FC3F7 0%, #9C7FFF 100%) !important;
  color: #fff !important; box-shadow: 0 4px 18px rgba(79,195,247,0.3);
}

/* ── METRIC CARDS ─────────────────────────────────────── */
[data-testid="stMetric"] {
  background: rgba(13,21,38,0.75); border: 1px solid rgba(79,195,247,0.12);
  border-radius: 13px; padding: 16px 18px; backdrop-filter: blur(12px);
  transition: all 0.2s ease; position: relative; overflow: hidden;
}
[data-testid="stMetric"]:hover {
  border-color: rgba(79,195,247,0.35);
  box-shadow: 0 0 22px rgba(79,195,247,0.08); transform: translateY(-2px);
}
[data-testid="stMetric"]::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, #4FC3F7, #9C7FFF);
}
[data-testid="stMetricLabel"] {
  font-size: 0.68rem !important; color: #7B9BB8 !important;
  font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em;
}
[data-testid="stMetricValue"] {
  font-size: 1.45rem !important; font-weight: 800; color: #E8F4FD !important;
  font-family: 'JetBrains Mono', monospace !important;
}

/* ── BUTTONS ───────────────────────────────────────────── */
.stButton > button {
  background: rgba(17,29,53,0.9); color: #E8F4FD;
  border: 1px solid rgba(79,195,247,0.22); border-radius: 10px;
  font-weight: 600; font-size: 0.85rem; padding: 9px 16px;
  transition: all 0.2s ease; backdrop-filter: blur(6px);
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(79,195,247,0.18), rgba(156,127,255,0.18));
  border-color: rgba(79,195,247,0.55);
  box-shadow: 0 0 18px rgba(79,195,247,0.2); color: #fff;
}

/* ── SIDEBAR ───────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: rgba(5,8,16,0.97) !important;
  border-right: 1px solid rgba(79,195,247,0.08);
}
[data-testid="stSidebar"] .block-container { padding: 1.2rem 1rem !important; }

/* ── PROGRESS ──────────────────────────────────────────── */
.stProgress > div > div {
  background: linear-gradient(90deg, #4FC3F7, #9C7FFF) !important;
  border-radius: 4px; box-shadow: 0 0 8px rgba(79,195,247,0.35);
}
.stProgress > div { background: rgba(255,255,255,0.05) !important; border-radius: 4px; }

/* ── MISC ─────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.06); margin: 12px 0; }

.sec {
  font-size: 0.67rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.13em; color: #7B9BB8; margin: 16px 0 7px 0;
}

.gcard {
  background: rgba(13,21,38,0.6); border: 1px solid rgba(255,255,255,0.05);
  border-radius: 14px; padding: 18px 20px; backdrop-filter: blur(14px);
}

.action-badge {
  border-radius: 12px; padding: 18px 24px; text-align: center;
  font-size: 1.25rem; font-weight: 800; margin-top: 14px;
  letter-spacing: 0.06em; text-shadow: 0 0 20px currentColor;
  font-family: 'JetBrains Mono', monospace;
  animation: fadeup 0.3s ease;
}
@keyframes fadeup {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

.legend-row {
  display: flex; gap: 16px; margin-top: 12px; font-size: 0.78rem;
  color: #7B9BB8; align-items: center; flex-wrap: wrap;
}
.legend-dot {
  width: 10px; height: 10px; border-radius: 3px;
  display: inline-block; margin-right: 5px;
}
</style>
"""


# ── resource loading ──────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    cfg = load_config(os.path.join(SIM_DIR, "config.yaml"))
    data_path    = os.path.join(SIM_DIR, "data",    "q_table.npy")
    results_path = os.path.join(SIM_DIR, "results", "qtable_dev0.npy")
    if os.path.isfile(data_path):
        qtable = np.load(data_path)
    elif os.path.isfile(results_path):
        qtable = np.load(results_path)
    else:
        st.error("No trained Q-table found. Run train_qlearning.py first.")
        st.stop()
    return cfg, qtable


# ── policy functions ──────────────────────────────────────────────────────────
def policy_qlearning(state, qtable):       return int(np.argmax(qtable[state]))
def policy_always_local(state, _q):        return ACTION_LOCAL
def policy_always_edge(state, _q):         return ACTION_EDGE
def policy_always_cloud(state, _q):        return ACTION_CLOUD

def policy_static_threshold(state, _q):
    """Rule-based heuristic matching evaluate.py."""
    bat, net, task, q, _ = index_to_state(state)
    if bat == 0:
        return ACTION_CLOUD if net > 0 else ACTION_LOCAL
    if task == TASK_HEAVY:
        if net == NETWORK_POOR: return ACTION_LOCAL
        elif q >= 3:            return ACTION_CLOUD
        else:                   return ACTION_EDGE
    elif task == TASK_LIGHT:    return ACTION_LOCAL
    else:
        if q < 2 and net > 0:   return ACTION_EDGE
        return ACTION_LOCAL

POLICY_FNS: dict[str, Callable[[int, np.ndarray], int]] = {
    "Trained Q-Learning": policy_qlearning,
    "Always-Local":       policy_always_local,
    "Always-Edge":        policy_always_edge,
    "Always-Cloud":       policy_always_cloud,
    "Static-Threshold":   policy_static_threshold,
}


# ── helpers ───────────────────────────────────────────────────────────────────
def make_fresh_env(cfg, seed=42):
    c = copy.deepcopy(cfg); c["seed"] = seed; return OffloadEnv(c)

def visited_mask(qtable):
    return (qtable != 0.0).any(axis=1)

def _chart(height=210):
    """Common dark-transparent Plotly layout."""
    return dict(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, family="Inter, sans-serif", size=11),
        margin=dict(l=4, r=4, t=34, b=4),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, color=MUTED),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, color=MUTED),
    )


# ── network routing diagram ───────────────────────────────────────────────────
def render_network_diagram(action: int) -> go.Figure:
    """ESP32 → Local / Edge / Cloud routing diagram."""
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(range=[0, 10], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 6],  showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8),
        height=235,
    )
    # ESP32 node
    fig.add_shape(type="rect", x0=0.2, y0=2.2, x1=2.2, y1=3.8,
                  line=dict(color=ACCENT, width=2),
                  fillcolor="rgba(79,195,247,0.07)")
    fig.add_annotation(x=1.2, y=3.0, text="<b>ESP32</b>",
                       showarrow=False, font=dict(color=TEXT, size=13))

    boxes = [
        (7.3, 4.9, 4.0, "LOCAL",  ACTION_COLORS["LOCAL"],  ACTION_GLOW["LOCAL"]),
        (7.3, 3.5, 2.6, "EDGE",   ACTION_COLORS["EDGE"],   ACTION_GLOW["EDGE"]),
        (7.3, 2.1, 1.2, "CLOUD",  ACTION_COLORS["CLOUD"],  ACTION_GLOW["CLOUD"]),
    ]
    arrow_y = {ACTION_LOCAL: 4.45, ACTION_EDGE: 3.05, ACTION_CLOUD: 1.65}

    for x, y_top, y_bot, label, col, glow in boxes:
        a_idx  = ACTION_NAMES.index(label)
        chosen = a_idx == action
        fig.add_shape(
            type="rect", x0=x, y0=y_bot, x1=x+1.7, y1=y_top,
            line=dict(color=col if chosen else "rgba(255,255,255,0.07)",
                      width=2.5 if chosen else 1),
            fillcolor=glow if chosen else "rgba(13,21,38,0.5)",
        )
        fig.add_annotation(
            x=x+0.85, y=(y_top+y_bot)/2,
            text=f"<b>{label}</b>", showarrow=False,
            font=dict(color="white" if chosen else MUTED, size=11),
        )

    for a_idx in range(N_ACTIONS):
        chosen = a_idx == action
        col    = ACTION_COLORS[ACTION_NAMES[a_idx]]
        ay     = arrow_y[a_idx]
        fig.add_annotation(
            x=7.3, y=ay, ax=2.2, ay=3.0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.1,
            arrowwidth=3.5 if chosen else 0.8,
            arrowcolor=col if chosen else "rgba(255,255,255,0.05)",
            opacity=1.0 if chosen else 0.12,
        )
    return fig


# ── simulation state helpers ──────────────────────────────────────────────────
def _init_sim_state(cfg, seed=42):
    """Build the initial mutable simulation state dict."""
    env = make_fresh_env(cfg, seed); state = env.reset()
    return dict(
        env=env, state=state, step=0, action=None,
        cum_energy=0.0, cum_reward=0.0,
        latencies=[], battery_hist=[env.devices[0].battery],
        reward_hist=[0.0], action_counts=[0, 0, 0], done=False,
    )

def _do_step(sim, qtable, policy_name):
    """Advance the sim dict by one step in-place."""
    if sim["done"]: return
    fn = POLICY_FNS[policy_name]
    action = fn(sim["state"], qtable)
    next_state, reward, done, info = sim["env"].step(action)
    sim["state"]       = next_state
    sim["action"]      = action
    sim["step"]       += 1
    sim["cum_energy"] += info["energy"][0] * 1000.0
    sim["cum_reward"] += reward
    sim["latencies"].append(info["latency"][0] * 1000.0)
    sim["battery_hist"].append(sim["env"].devices[0].battery)
    sim["reward_hist"].append(sim["cum_reward"])
    sim["action_counts"][action] += 1
    sim["done"] = done


# ── Tab 1: Live Simulation ────────────────────────────────────────────────────
def tab_live(cfg, qtable):
    # ── sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<p class='sec'>Policy</p>", unsafe_allow_html=True)
        policy_name = st.selectbox(
            "", POLICY_NAMES, key="live_policy", label_visibility="collapsed")
        st.markdown("<p class='sec'>Playback Speed</p>", unsafe_allow_html=True)
        speed = st.slider("", 0.5, 10.0, 2.0, 0.5,
                          key="live_speed", label_visibility="collapsed")
        st.markdown("<hr/>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        play_btn  = b1.button("▶ Play",  key="live_play",  use_container_width=True)
        pause_btn = b2.button("⏸ Pause", key="live_pause", use_container_width=True)
        b3, b4 = st.columns(2)
        step_btn  = b3.button("⏭ Step",  key="live_step",  use_container_width=True)
        reset_btn = b4.button("↺ Reset", key="live_reset", use_container_width=True)
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size:0.7rem;color:{MUTED};"
            "font-family:JetBrains Mono,monospace;line-height:1.8'>"
            f"Q-table<br/>{STATE_SIZE} states<br/>{N_ACTIONS} actions</p>",
            unsafe_allow_html=True,
        )

    # ── session state ─────────────────────────────────────────────────────────
    if "sim" not in st.session_state or reset_btn:
        st.session_state["sim"]     = _init_sim_state(cfg)
        st.session_state["playing"] = False
    if play_btn:  st.session_state["playing"] = True
    if pause_btn: st.session_state["playing"] = False

    sim     = st.session_state["sim"]
    playing = st.session_state.get("playing", False)

    if step_btn: _do_step(sim, qtable, policy_name)
    if playing and not sim["done"]:
        _do_step(sim, qtable, policy_name)
        time.sleep(1.0 / speed)
        st.rerun()

    sim = st.session_state["sim"]

    # ── read env state ────────────────────────────────────────────────────────
    dev     = sim["env"].devices[0]
    battery = dev.battery
    network = dev.network
    task    = dev.task_type
    q_depth = sim["env"].edge_queue
    action  = sim["action"] if sim["action"] is not None else -1

    # ── layout ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([13, 9], gap="large")

    with col_left:
        # Metrics row
        st.markdown("<p class='sec'>Live Metrics</p>", unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Step",          sim["step"])
        m2.metric("Last Action",   ACTION_NAMES[action] if action >= 0 else "—")
        m3.metric("Energy (mWh)",  f"{sim['cum_energy']:.3f}")
        m4.metric("Avg Lat ms",
                  f"{np.mean(sim['latencies']):.1f}" if sim["latencies"] else "—")
        m5.metric("Cum Reward",    f"{sim['cum_reward']:.2f}")

        # Device gauges
        st.markdown("<p class='sec'>Device State</p>", unsafe_allow_html=True)
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            bat_pct = max(0, min(100, int(battery)))
            st.markdown(
                f"<p style='font-size:0.67rem;color:{MUTED};font-weight:700;"
                "text-transform:uppercase;letter-spacing:.12em;margin-bottom:7px'>"
                "Battery</p>", unsafe_allow_html=True)
            st.progress(bat_pct / 100, text=f"{bat_pct}%")
        with g2:
            net_icons = ["🔴 Poor", "🟡 Medium", "🟢 Good"]
            st.metric("Network", net_icons[network])
        with g3:
            st.metric("Task", TASK_NAMES[task])
        with g4:
            q_max = sim["env"].max_queue
            st.markdown(
                f"<p style='font-size:0.67rem;color:{MUTED};font-weight:700;"
                "text-transform:uppercase;letter-spacing:.12em;margin-bottom:7px'>"
                "Edge Queue</p>", unsafe_allow_html=True)
            st.progress(q_depth / q_max, text=f"{q_depth}/{q_max}")

        # Rolling charts
        st.markdown("<p class='sec'>History</p>", unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)

        with rc1:
            x = list(range(len(sim["battery_hist"])))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=sim["battery_hist"], mode="lines",
                line=dict(color=ACTION_COLORS["CLOUD"], width=2.5),
                fill="tozeroy", fillcolor="rgba(105,255,71,0.07)",
            ))
            fig.update_layout(
                title=dict(text="Battery %", font=dict(color=TEXT, size=12)),
                **_chart(215))
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

        with rc2:
            x = list(range(len(sim["reward_hist"])))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=sim["reward_hist"], mode="lines",
                line=dict(color=ACCENT2, width=2.5),
                fill="tozeroy", fillcolor="rgba(156,127,255,0.07)",
            ))
            fig.update_layout(
                title=dict(text="Cum Reward", font=dict(color=TEXT, size=12)),
                **_chart(215))
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

        with rc3:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ACTION_NAMES, y=sim["action_counts"],
                marker_color=[ACTION_COLORS[n] for n in ACTION_NAMES],
                marker_line_width=0,
                marker=dict(opacity=0.88),
            ))
            fig.update_layout(
                title=dict(text="Actions", font=dict(color=TEXT, size=12)),
                **_chart(215))
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

    with col_right:
        st.markdown("<p class='sec'>Task Routing</p>", unsafe_allow_html=True)
        if action >= 0:
            st.plotly_chart(render_network_diagram(action),
                            use_container_width=True,
                            config={"displayModeBar": False})
            col_b = ACTION_COLORS[ACTION_NAMES[action]]
            col_g = ACTION_GLOW[ACTION_NAMES[action]]
            st.markdown(
                f"<div class='action-badge' style='background:{col_g};"
                f"border:2px solid {col_b};color:{col_b};'>"
                f"{ACTION_NAMES[action]}</div>",
                unsafe_allow_html=True,
            )
            # Donut: action split
            if sum(sim["action_counts"]) > 0:
                fig_d = go.Figure(go.Pie(
                    labels=ACTION_NAMES,
                    values=sim["action_counts"],
                    hole=0.68,
                    marker=dict(
                        colors=[ACTION_COLORS[n] for n in ACTION_NAMES],
                        line=dict(color="rgba(0,0,0,0)", width=0),
                    ),
                    textinfo="none",
                ))
                fig_d.update_layout(
                    height=155, showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=0, b=0),
                    annotations=[dict(
                        text="<b>split</b>",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=11, color=MUTED),
                    )],
                )
                st.plotly_chart(fig_d, use_container_width=True,
                                config={"displayModeBar": False})
        else:
            st.markdown(
                f"<div class='gcard' style='text-align:center;color:{MUTED};"
                "margin-top:40px;padding:40px 20px'>"
                "<span style='font-size:2rem'>📡</span><br/><br/>"
                "Press <b>▶ Play</b> or <b>⏭ Step</b><br/>to begin</div>",
                unsafe_allow_html=True,
            )

    if sim["done"]:
        st.success("🎉 Episode complete — press ↺ Reset to run again.")


# ── Tab 2: Compare Policies ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Running all 5 policies…")
def compare_policies(cfg_json: str, seed: int = 42) -> pd.DataFrame:
    """
    Run each of the 5 policies on a single deterministic env seed.
    Returns a summary DataFrame.
    """
    cfg_base = json.loads(cfg_json)
    _, qtable = load_resources()
    rows = []
    for pname in POLICY_NAMES:
        c = copy.deepcopy(cfg_base); c["seed"] = seed
        env = make_fresh_env(c, seed); state = env.reset()
        fn = POLICY_FNS[pname]
        energies, latencies, accuracies, rewards = [], [], [], []
        done = False
        while not done:
            action = fn(state, qtable)
            next_state, rew, done, info = env.step(action)
            energies.append(info["energy"][0])
            latencies.append(info["latency"][0])
            accuracies.append(info["accuracy"][0])
            rewards.append(rew)
            state = next_state
        rows.append({
            "Policy":        pname,
            "Energy (mWh)":  round(float(np.mean(energies)) * 1000, 4),
            "Latency (ms)":  round(float(np.mean(latencies)) * 1000, 2),
            "Accuracy":      round(float(np.mean(accuracies)), 4),
            "Total Reward":  round(float(np.sum(rewards)), 3),
        })
    return pd.DataFrame(rows)


def tab_compare(cfg):
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.85rem;margin-bottom:14px'>"
        "All five policies run on the <b>same</b> environment seed — "
        "results from real simulation steps, no hard-coded values.</p>",
        unsafe_allow_html=True,
    )
    sc, _ = st.columns([2, 6])
    with sc:
        seed = st.number_input("Seed", value=42, min_value=0, step=1, key="cmp_seed")

    df = compare_policies(json.dumps(cfg), seed=int(seed))

    st.markdown("<p class='sec'>Summary Table</p>", unsafe_allow_html=True)
    st.dataframe(
        df.style
          .highlight_max(subset=["Accuracy", "Total Reward"],  color="#0d3321")
          .highlight_min(subset=["Energy (mWh)", "Latency (ms)"], color="#0d3321"),
        use_container_width=True,
    )

    st.markdown("<p class='sec'>Visual Comparison</p>", unsafe_allow_html=True)
    palette = ["#FF6B35", "#4FC3F7", "#69FF47", "#FF4E8A", "#9C7FFF"]
    metrics = ["Energy (mWh)", "Latency (ms)", "Accuracy", "Total Reward"]
    fig = make_subplots(rows=1, cols=4, subplot_titles=metrics,
                        horizontal_spacing=0.05)
    for ci, metric in enumerate(metrics, 1):
        for ri, row in df.iterrows():
            fig.add_trace(
                go.Bar(
                    name=row["Policy"],
                    x=[row["Policy"]], y=[row[metric]],
                    marker_color=palette[ri % len(palette)],
                    marker_line_width=0,
                    showlegend=(ci == 1),
                ),
                row=1, col=ci,
            )
    fig.update_layout(
        height=400, barmode="group",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, family="Inter, sans-serif"),
        legend=dict(orientation="h", y=-0.26, font=dict(size=10, color=MUTED)),
        margin=dict(l=8, r=8, t=52, b=8),
    )
    fig.update_xaxes(showticklabels=False, gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 3: Ask the Agent ──────────────────────────────────────────────────────
def tab_ask(qtable):
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.85rem;margin-bottom:14px'>"
        "Set a hypothetical device state and inspect what the Q-agent recommends.</p>",
        unsafe_allow_html=True,
    )
    col_sl, col_out = st.columns([1, 2], gap="large")

    with col_sl:
        st.markdown("<p class='sec'>State Parameters</p>", unsafe_allow_html=True)
        bat_bin  = st.slider("Battery bin  (0=<20%  4=>80%)",
                             0, N_BATTERY_BINS - 1, 2, key="ask_bat")
        net_val  = st.select_slider("Network quality",
                                    options=["Poor", "Medium", "Good"],
                                    value="Medium", key="ask_net")
        task_val = st.select_slider("Task type",
                                    options=["Light", "Medium", "Heavy"],
                                    value="Medium", key="ask_task")
        q_bin    = st.slider("Queue bin  (0=empty  4=full)",
                             0, N_EDGE_QUEUE_BINS - 1, 1, key="ask_q")

    net_idx  = ["Poor", "Medium", "Good"].index(net_val)
    task_idx = ["Light", "Medium", "Heavy"].index(task_val)
    state    = state_to_index(bat_bin, net_idx, task_idx, q_bin, 0)
    q_vals   = qtable[state]
    is_vis   = visited_mask(qtable)[state]
    best_act = int(np.argmax(q_vals))

    with col_out:
        st.markdown("<p class='sec'>Q-Values</p>", unsafe_allow_html=True)
        if not is_vis:
            st.warning("⚠️ State never visited during training — Q-values are unreliable.")

        colors = ([ACTION_COLORS[n] for n in ACTION_NAMES]
                  if is_vis else ["#333", "#333", "#333"])
        fig = go.Figure(go.Bar(
            x=ACTION_NAMES, y=q_vals,
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.4f}" for v in q_vals], textposition="outside",
            textfont=dict(color=MUTED, size=11),
        ))
        fig.update_layout(**_chart(280), yaxis_title="Q-value")
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

        if is_vis:
            cb = ACTION_COLORS[ACTION_NAMES[best_act]]
            cg = ACTION_GLOW[ACTION_NAMES[best_act]]
            st.markdown(
                f"<div class='action-badge' style='background:{cg};"
                f"border:2px solid {cb};color:{cb};'>"
                f"✅ Recommended: {ACTION_NAMES[best_act]}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='action-badge' style='background:rgba(255,255,255,0.03);"
                f"border:2px solid rgba(255,255,255,0.1);color:{MUTED};'>"
                "❓ No reliable recommendation</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            f"Index {state} | bat={bat_bin} net={net_val} "
            f"task={task_val} q={q_bin} visited={is_vis}"
        )


# ── Tab 4: Policy Heatmaps ────────────────────────────────────────────────────
def _build_heatmap(qtable, vm, task, q, title) -> go.Figure:
    """Return a Plotly Battery×Network action heatmap."""
    color_map = {
        ACTION_LOCAL:  ACTION_COLORS["LOCAL"],
        ACTION_EDGE:   ACTION_COLORS["EDGE"],
        ACTION_CLOUD:  ACTION_COLORS["CLOUD"],
    }
    fig = go.Figure()
    for bat in range(N_BATTERY_BINS):
        for net in range(N_NETWORKS):
            idx = state_to_index(bat, net, task, q, 0)
            if vm[idx]:
                act   = int(np.argmax(qtable[idx]))
                fill  = color_map[act]
                label = ACTION_NAMES[act][0]
            else:
                fill  = "rgba(17,29,53,0.8)"
                label = "?"
            fig.add_shape(
                type="rect",
                x0=net - 0.5, y0=bat - 0.5,
                x1=net + 0.5, y1=bat + 0.5,
                fillcolor=fill,
                line=dict(color="rgba(5,8,16,0.9)", width=2.5),
            )
            fig.add_annotation(
                x=net, y=bat, text=f"<b>{label}</b>",
                showarrow=False, font=dict(color="white", size=15),
            )
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=11)),
        xaxis=dict(
            tickvals=list(range(N_NETWORKS)), ticktext=NET_NAMES,
            title="Network Quality", color=MUTED,
            showgrid=False, zeroline=False, range=[-0.5, N_NETWORKS - 0.5],
        ),
        yaxis=dict(
            tickvals=list(range(N_BATTERY_BINS)), ticktext=BAT_LABELS,
            title="Battery Level", color=MUTED,
            showgrid=False, zeroline=False, range=[-0.5, N_BATTERY_BINS - 0.5],
        ),
        height=290,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, family="Inter, sans-serif"),
        margin=dict(l=8, r=8, t=44, b=8),
    )
    return fig


def tab_heatmaps(qtable):
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.85rem;margin-bottom:14px'>"
        "Greedy actions chosen by the Q-agent per state. "
        "<b>Dark/?</b> = state never visited during training.</p>",
        unsafe_allow_html=True,
    )
    vm = visited_mask(qtable)

    fc1, fc2, _ = st.columns([2, 2, 4])
    with fc1:
        task_opt = st.selectbox(
            "Task panel",
            ["All 4 preset panels", "Light", "Medium", "Heavy"],
            key="hm_task",
        )
    with fc2:
        q_val = st.slider("Queue bin", 0, N_EDGE_QUEUE_BINS - 1, 0, key="hm_q")

    PRESETS = [
        (TASK_LIGHT,  0,                    "Light Task | Empty Queue"),
        (TASK_HEAVY,  0,                    "Heavy Task | Empty Queue"),
        (TASK_HEAVY,  N_EDGE_QUEUE_BINS-1,  "Heavy Task | Full Queue"),
        (TASK_MEDIUM, N_EDGE_QUEUE_BINS//2, "Medium Task | Medium Queue"),
    ]
    st.markdown("<p class='sec'>Heatmaps</p>", unsafe_allow_html=True)

    if task_opt == "All 4 preset panels":
        ra = st.columns(2, gap="medium")
        rb = st.columns(2, gap="medium")
        for i, (t, q, ttl) in enumerate(PRESETS):
            c = ra[i] if i < 2 else rb[i - 2]
            with c:
                st.plotly_chart(
                    _build_heatmap(qtable, vm, t, q, ttl),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
    else:
        tm = {"Light": TASK_LIGHT, "Medium": TASK_MEDIUM, "Heavy": TASK_HEAVY}
        st.plotly_chart(
            _build_heatmap(qtable, vm, tm[task_opt], q_val,
                           f"{task_opt} Task | Queue bin={q_val}"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # Legend
    cl = ACTION_COLORS["LOCAL"]
    ce = ACTION_COLORS["EDGE"]
    cc = ACTION_COLORS["CLOUD"]
    st.markdown(
        f"<div class='legend-row'>"
        f"<span><span class='legend-dot' style='background:{cl}'></span>LOCAL</span>"
        f"<span><span class='legend-dot' style='background:{ce}'></span>EDGE</span>"
        f"<span><span class='legend-dot' style='background:{cc}'></span>CLOUD</span>"
        f"<span><span class='legend-dot' style='background:rgba(17,29,53,.9);"
        "border:1px solid rgba(255,255,255,.1)'></span>Unvisited</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="IoT Offloading Dashboard",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_css(), unsafe_allow_html=True)

    # Animated header
    playing = st.session_state.get("playing", False)
    dot     = "<span class='pulse-dot'></span>" if playing else ""
    cfg, qtable = load_resources()

    st.markdown(
        f"<div class='dash-header'>"
        f"<h1 class='dash-title'>{dot}IoT Task-Offloading Dashboard</h1>"
        f"<p class='dash-sub'>"
        f"ESP32 &rarr; Local / Edge / Cloud &nbsp;·&nbsp; "
        f"Tabular Q-Learning &nbsp;·&nbsp; "
        f"{STATE_SIZE}&nbsp;states &times; {N_ACTIONS}&nbsp;actions"
        f"</p>"
        f"<span class='stat-pill'>{STATE_SIZE} States</span>"
        f"<span class='stat-pill'>{N_ACTIONS} Actions</span>"
        f"<span class='stat-pill'>{len(POLICY_NAMES)} Policies</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "▶  Live Simulation",
        "📊  Compare Policies",
        "🔍  Ask the Agent",
        "🗺️  Policy Heatmaps",
    ])
    with tab1: tab_live(cfg, qtable)
    with tab2: tab_compare(cfg)
    with tab3: tab_ask(qtable)
    with tab4: tab_heatmaps(qtable)


if __name__ == "__main__":
    main()
