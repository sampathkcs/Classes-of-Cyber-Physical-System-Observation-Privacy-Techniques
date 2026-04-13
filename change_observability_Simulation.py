#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

# Final Unified Simulation – JSD for Unobservable + Attacker Observable + Combined Path Figure

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import copy
from collections import Counter
from scipy.spatial.distance import jensenshannon
from matplotlib.lines import Line2D

from pprint import pprint

# ---------- Font size constants (aligned with card-swap heatmap) ----------
TITLE_FONTSIZE  = 15   # heatmap titles
LABEL_FONTSIZE  = 13   # axis labels
TICK_FONTSIZE   = 11   # tick labels
CELL_FONTSIZE   = 10   # numbers inside heatmap cells
LEGEND_FONTSIZE = 10   # legend text size

# Do not generate Type 3 fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# --- Setup ---
rng = np.random.default_rng()  # fixed seed for consistency
NUM_USERS = 10
JOURNEY_LENGTH = 10
NUM_EXPERIMENTS = 10000
nodes = [f"V{i}" for i in range(1, 15)]

# --- Graph setup ---
edges = [
    ("V1", "V2"), ("V1", "V5"),
    ("V2", "V1"), ("V2", "V3"), ("V2", "V6"),
    ("V3", "V2"), ("V3", "V4"), ("V3", "V7"),
    ("V4", "V3"), ("V4", "V8"),
    ("V5", "V6"), ("V5", "V9"),
    ("V6", "V5"), ("V6", "V2"), ("V6", "V7"), ("V6", "V10"),
    ("V7", "V6"), ("V7", "V3"), ("V7", "V8"), ("V7", "V11"),
    ("V8", "V7"), ("V8", "V4"), ("V8", "V12"),
    ("V9", "V10"),
    ("V10", "V6"), ("V10", "V9"), ("V10", "V11"), ("V10", "V13"),
    ("V11", "V10"), ("V11", "V7"), ("V11", "V12"), ("V11", "V14"),
    ("V12", "V8"), ("V12", "V11"),
    ("V13", "V10"), ("V13", "V14"),
    ("V14", "V11"), ("V14", "V13"), ("V14", "V12")
]
G: nx.DiGraph[str] = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# --- Start locations ---
def make_user_starts():
    # U0..U1 at V1, U2..U4 at V2, others random
    return ({i: "V1" for i in range(2)} |
            {i: "V2" for i in range(2, 5)} |
            {i: rng.choice(nodes) for i in range(5, 10)})

USER_START_LOCATION = make_user_starts()

print("Start locations")
print(USER_START_LOCATION)

# --- Journey simulation ---
def decide_stop(current: str) -> str:
    neighbours = list(G.successors(current))
    if not neighbours:
        neighbours = [n for n in nodes if n != current]
    return rng.choice(neighbours)

def simulate_journeys() -> list[list[str]]:
    journeys: list[list[str]] = []
    for uid in range(NUM_USERS):
        path = [USER_START_LOCATION[uid]]
        current = path[0]
        for _t in range(1, JOURNEY_LENGTH):
            current = decide_stop(current)
            path.append(current)
        journeys.append(path)
    return journeys

# --- Run base simulation for visualisation (single run) ---
true_journeys = simulate_journeys()

print("True journeys")
pprint(true_journeys)

# --- Observability mask ---
observability_mask = np.full((NUM_USERS, JOURNEY_LENGTH), True, dtype=bool)

for uid in range(NUM_USERS):
    # user 0 fully observable, so skip it
    if uid == 0:
        continue

    hidden = rng.choice(JOURNEY_LENGTH, size=3, replace=False)

    print(uid, hidden)

    observability_mask[uid, hidden] = False

print(observability_mask)

# --- Masked journeys (for visualisation only, using "Unknown") ---
masked_journeys_visual = copy.deepcopy(true_journeys)
for uid in range(NUM_USERS):
    for t in range(JOURNEY_LENGTH):
        if not observability_mask[uid, t]:
            masked_journeys_visual[uid][t] = "Unknown"

# --- Full experiments (for distributions) ---
def simulate_full_experiments():
    return [simulate_journeys() for _ in range(NUM_EXPERIMENTS)]

journeys_full = simulate_full_experiments()

# --- Masked version for unobservable JSD (random replacement) ---
def simulate_masked_for_jsd(journeys_full, obs_mask) -> list[list[list[str]]]:
    """Random replacement for hidden steps for statistical masking."""
    journeys_masked = copy.deepcopy(journeys_full)
    for exp in journeys_masked:
        for uid in range(NUM_USERS):
            if uid == 0:
                continue

            for t in range(JOURNEY_LENGTH):
                if not obs_mask[uid, t]:
                    exp[uid][t] = "Unknown"  # rng.choice(nodes)
    return journeys_masked

journeys_masked = simulate_masked_for_jsd(journeys_full, observability_mask)

# --- Counting and prob conversion helpers ---
def journeys_to_counts_full(journeys: list[list[list[str]]]) -> list[list[Counter[str]]]:
    """Counts for full-length (JOURNEY_LENGTH) journeys."""
    users_dist = [
        [
            Counter({n: 0 for n in nodes + ["Unknown"]})
            for _ in range(JOURNEY_LENGTH)
        ]
        for _ in range(NUM_USERS)
    ]
    for experiment in journeys:
        for uid, path in enumerate(experiment):
            for t, node in enumerate(path):
                users_dist[uid][t][node] += 1
    return users_dist

def counts_to_probs(counts: list[list[Counter[str]]]) -> list[list[dict[str, float]]]:
    return [
        [
            {k: v / NUM_EXPERIMENTS for k, v in counter.items()}
            for counter in user
        ]
        for user in counts
    ]

# --- True and unobservable (random-mask) distributions ---
dist_full   = counts_to_probs(journeys_to_counts_full(journeys_full))
dist_masked = counts_to_probs(journeys_to_counts_full(journeys_masked))

# --- JSD matrix for unobservable model (original one) ---
jsd_unobs = np.full((NUM_USERS, JOURNEY_LENGTH), np.nan)
for uid in range(NUM_USERS):
    for t in range(JOURNEY_LENGTH):
        p = np.array(list(dist_full[uid][t].values()))
        q = np.array(list(dist_masked[uid][t].values()))

        assert np.isclose(p.sum(), 1.0), f"Bad p {uid} {t} {p.sum()} {p}"
        assert np.isclose(q.sum(), 1.0), f"Bad q {uid} {t} {q.sum()} {q}"

        print(uid, t)
        print("p", p)
        print("q", q)

        jsd_unobs[uid, t] = jensenshannon(p, q, base=2)

print(jsd_unobs)

#  FIGURE 1: JSD HEATMAP – True vs Unobservable (random mask)

plt.figure(figsize=(12, 6))
ax1 = sns.heatmap(
    jsd_unobs,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    annot_kws={"fontsize": CELL_FONTSIZE, "fontweight": "bold"},
    cbar_kws={'label': 'JSD'}
)

ax1.set_title(
    "JSD between the journey taken by the user\n"
    "and the observations made when the adversary knows if a stop occurred",
    fontsize=TITLE_FONTSIZE,
    pad=10
)
ax1.set_xlabel("Time Step", fontsize=LABEL_FONTSIZE)
ax1.set_ylabel("User ID", fontsize=LABEL_FONTSIZE)

ax1.set_xticks(np.arange(JOURNEY_LENGTH) + 0.5)
ax1.set_xticklabels(
    np.arange(1, JOURNEY_LENGTH + 1),
    fontsize=TICK_FONTSIZE,
    fontweight='bold'
)

ax1.set_yticks(np.arange(NUM_USERS) + 0.5)
ax1.set_yticklabels(
    [f"U{i}" for i in range(NUM_USERS)],
    fontsize=TICK_FONTSIZE,
    fontweight='bold'
)

# Colourbar font sizes (no bold for label now)
cbar1 = ax1.collections[0].colorbar
cbar1.ax.set_ylabel("JSD", rotation=-90, va="bottom",
                    fontsize=LABEL_FONTSIZE)
cbar1.ax.tick_params(labelsize=TICK_FONTSIZE)

ax1.invert_yaxis()

plt.tight_layout()
plt.savefig("jsd_unobservable_F5.pdf", bbox_inches='tight')
plt.close()


#  Build attacker observable compressed journeys (red model)
#  Attacker sees only observable steps, reindexed to 7 steps.

OBS_STEPS = JOURNEY_LENGTH - 3  # attacker effectively sees 7 time steps

def build_attacker_observable_experiments(journeys_full: list[list[list[str]]], obs_mask, obs_steps: int) -> list[list[list[str]]]:
    """
    For each experiment and user, build a compressed journey of length obs_steps:
      compressed[k] = node at the k-th observable time step.
    """
    experiments: list[list[list[str]]] = []
    for exp in journeys_full:
        attacker_exp: list[list[str]] = []
        for uid in range(NUM_USERS):

            obs_indices = np.where(obs_mask[uid])[0]

            # Check if sorted
            assert np.all(obs_indices[:-1] <= obs_indices[1:])

            # Trim extra steps
            # For example, user 0 will have no steps removed due to the mask, so we remove here.
            obs_indices = obs_indices[:obs_steps]

            compressed: list[str] = [exp[uid][t] for t in obs_indices]
            attacker_exp.append(compressed)
        experiments.append(attacker_exp)
    return experiments

attacker_experiments = build_attacker_observable_experiments(
    journeys_full, observability_mask, OBS_STEPS
)

dist_attacker = counts_to_probs(journeys_to_counts_full(attacker_experiments))

# --- JSD matrix for attacker observable model ---
jsd_attacker = np.full((NUM_USERS, OBS_STEPS), np.nan)

for uid in range(NUM_USERS):
    for k in range(OBS_STEPS):
        # True timeline distribution at time t = k
        p = np.array([dist_full[uid][k][n] for n in nodes])
        # Attacker compressed distribution at index k
        q = np.array([dist_attacker[uid][k][n] for n in nodes])

        assert np.isclose(p.sum(), 1.0), f"Bad p {uid} {k} {p.sum()} {p}"
        assert np.isclose(q.sum(), 1.0), f"Bad q {uid} {k} {q.sum()} {q}"

        print(uid, k)
        print("p", p)
        print("q", q)

        jsd_attacker[uid, k] = jensenshannon(p, q, base=2)

print(jsd_attacker)

#  FIGURE 2: JSD HEATMAP – True vs Attacker Observable (7 steps)

plt.figure(figsize=(12, 6))
ax2 = sns.heatmap(
    jsd_attacker,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    annot_kws={"fontsize": CELL_FONTSIZE, "fontweight": "bold"},
    cbar_kws={'label': 'JSD'}
)

ax2.set_title(
    "JSD between the journey taken by the user\n"
    "and the observations made when the adversary does not know if a stop occurred",
    fontsize=TITLE_FONTSIZE,
    pad=10
)
ax2.set_xlabel("Adversary Time Step", fontsize=LABEL_FONTSIZE)
ax2.set_ylabel("User ID", fontsize=LABEL_FONTSIZE)

ax2.set_xticks(np.arange(OBS_STEPS) + 0.5)
ax2.set_xticklabels(
    np.arange(1, OBS_STEPS + 1),
    fontsize=TICK_FONTSIZE,
    fontweight='bold'
)

ax2.set_yticks(np.arange(NUM_USERS) + 0.5)
ax2.set_yticklabels(
    [f"U{i}" for i in range(NUM_USERS)],
    fontsize=TICK_FONTSIZE,
    fontweight='bold'
)

cbar2 = ax2.collections[0].colorbar
cbar2.ax.set_ylabel("JSD", rotation=-90, va="bottom",
                    fontsize=LABEL_FONTSIZE)
cbar2.ax.tick_params(labelsize=TICK_FONTSIZE)

ax2.invert_yaxis()

plt.tight_layout()
plt.savefig("jsd_Adversary_observable_F5.pdf", bbox_inches='tight')
plt.close()


#  FIGURE 3: COMBINED PATH VIEW (true / masked / attacker-visible)

fig, ax3 = plt.subplots(figsize=(14, 14))

# Within-user spacing
LAYER_GAP = 1.1
ROW_SPACE = 4.0

y_centres = []
user_labels = []

for uid in range(NUM_USERS):
    y_red    = uid * ROW_SPACE
    y_orange = y_red + LAYER_GAP
    y_green  = y_red + 2 * LAYER_GAP

    x_vals = np.arange(1, JOURNEY_LENGTH + 1)
    y_centres.append((y_green + y_red) / 2.0)
    user_labels.append(f"U{uid}")

    # TRUE PATH (GREEN)
    ax3.plot(x_vals, [y_green] * JOURNEY_LENGTH, color='green', linewidth=2.2)
    for t in range(JOURNEY_LENGTH):
        ax3.text(
            x_vals[t], y_green,
            true_journeys[uid][t],
            ha='center', va='center',
            fontsize=10, color='black',
            bbox=dict(boxstyle="round,pad=0.25", edgecolor='black', facecolor='white')
        )

    # OBS-KNOWN PATH (ORANGE)
    ax3.plot(x_vals, [y_orange] * JOURNEY_LENGTH, color='orange', linewidth=2.2)
    for t in range(JOURNEY_LENGTH):
        ax3.text(
            x_vals[t], y_orange,
            masked_journeys_visual[uid][t],
            ha='center', va='center',
            fontsize=10, color='black',
            bbox=dict(boxstyle="round,pad=0.25", edgecolor='black', facecolor='white')
        )

    # ATTACKER OBSERVABLE (RED)
    obs_indices = np.where(observability_mask[uid])[0]
    obs_indices = np.sort(obs_indices)[:OBS_STEPS]
    if len(obs_indices) > 1:
        obs_x = np.arange(1, len(obs_indices) + 1)
        ax3.plot(obs_x, [y_red] * len(obs_x), color='red', linewidth=2.2)
        for k, t in enumerate(obs_indices):
            ax3.text(
                obs_x[k], y_red,
                true_journeys[uid][t],
                ha='center', va='center',
                fontsize=10, color='black',
                bbox=dict(boxstyle="round,pad=0.25", edgecolor='black', facecolor='white')
            )

# Axis labels (already non-bold)
ax3.set_xticks(np.arange(1, JOURNEY_LENGTH + 1))
ax3.set_xlabel("Time Step", fontsize=12)

ax3.set_yticks(y_centres)
ax3.set_yticklabels(user_labels, fontsize=11)
ax3.set_ylabel("User ID", fontsize=12)

ax3.set_title("User Journeys and Adversary Observations", fontsize=14, pad=6)

ax3.grid(True, linestyle='--', axis='x', alpha=0.3)

# ---- Legend (inside, small gap, bold text) ----
legend_lines = [
    ("green",  "True journey taken by user"),
    ("orange", "Observed journey when adversary knows if a stop occurred"),
    ("red",    "Observed journey when adversary only sees observable taps"),
]

line_handles = [Line2D([0], [0], color=c, linewidth=3) for c,_ in legend_lines]
line_labels  = [txt for _,txt in legend_lines]

extra_handles = [Line2D([], [], linestyle='None'),
                 Line2D([], [], linestyle='None')]
extra_labels = [
    "V1-V14 are nodes where the user taps the card",
    '"Unknown" marks time steps not observable to the adversary'
]

leg = fig.legend(
    line_handles + extra_handles,
    line_labels + extra_labels,
    loc='lower center',
    ncol=2,
    fontsize=LEGEND_FONTSIZE,
    frameon=False,
    handletextpad=1,
    labelspacing=0.5,
    borderpad=0.2
)

plt.subplots_adjust(bottom=0.08)   # tighter gap to the legend
plt.savefig("Journeys_F5.pdf", bbox_inches='tight')
plt.close()

print("\nGenerated figures:")
print("   - jsd_unobservable_F5.pdf")
print("   - jsd_Adversary_observable_F5.pdf")
print("   - Journeys_F5.pdf")
