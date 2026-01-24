# -*- coding: utf-8 -*-

############################### Final Simulation  (No Forced Return)################################
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import jensenshannon
from collections import Counter
import random
import seaborn as sns
import pandas as pd
from pprint import pprint
import copy

# Define the graph using adjacency list for 14-node transit network
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

rng = random.Random(0)
G = nx.DiGraph()
G.add_edges_from(edges)

NUM_EXPERIMENTS = 1000
JOURNEY_LENGTH = 10
NUM_USERS = 10

def make_user_starts(g: nx.DiGraph, start: int, count: int, location: str) -> dict[int, str]:
    nodes = list(g.nodes())
    return {
        i: location if location != "random" else rng.choice(nodes)
        for i in range(start, start+count)
    }

USER_START_LOCATION = (
    make_user_starts(G, 0, 2, "V1") |
    make_user_starts(G, 2, 3, "V2") |
    make_user_starts(G, 5, 5, "random")
)

assert len(USER_START_LOCATION) == NUM_USERS

def decide_stop(g, uid, t, current):
    node_choices = list(set(g.nodes()) - {USER_START_LOCATION[uid]})
    return rng.choice(node_choices)

def simulate_journeys(G):
    users_path = []
    for uid in range(NUM_USERS):
        start = USER_START_LOCATION[uid]
        path = [start]
        current = start
        for t in range(1, JOURNEY_LENGTH):
            current = decide_stop(G, uid, t, current)
            path.append(current)
        users_path.append(path)
    return users_path

def simulate_swaps(g, users_path):
    users_path_swapped = copy.deepcopy(users_path)
    def swap_at_t(t, node, uids):
        available = list(uids)
        while len(available) >= 2:
            uid1, uid2 = rng.sample(available, 2)
            available.remove(uid1); available.remove(uid2)
            users_path_swapped[uid1][t:], users_path_swapped[uid2][t:] = \
                users_path_swapped[uid2][t:], users_path_swapped[uid1][t:]
        return available

    for t in range(JOURNEY_LENGTH):
        user_locations = {
            node: {uid for uid in range(NUM_USERS)
                   if users_path[uid][t] == node and uid != 0}
            for node in g.nodes()
        }
        for node, uids in user_locations.items():
            if len(uids) > 1:
                swap_at_t(t, node, uids)
    return users_path_swapped

def journeys_to_counts(g, experiments):
    users_dist = [
        [Counter({n: 0 for n in g.nodes()}) for _ in range(JOURNEY_LENGTH)]
        for _ in range(NUM_USERS)
    ]
    for experiment in experiments:
        for uid, journey in enumerate(experiment):
            for i in range(JOURNEY_LENGTH):
                users_dist[uid][i][journey[i]] += 1
    return users_dist

def counts_to_dist(journeys_counts):
    return [
        [{k: v / NUM_EXPERIMENTS for (k, v) in counts.items()} for counts in j]
        for j in journeys_counts
    ]

# Run
users_journeys = [simulate_journeys(G) for _ in range(NUM_EXPERIMENTS)]
users_swap = [simulate_swaps(G, uj) for uj in users_journeys]

dist = counts_to_dist(journeys_to_counts(G, users_journeys))
dist_swap = counts_to_dist(journeys_to_counts(G, users_swap))

jsds = np.zeros((NUM_USERS, JOURNEY_LENGTH))
for uid in range(NUM_USERS):
    for t in range(JOURNEY_LENGTH):
        p = np.array(list(dist[uid][t].values()))
        q = np.array(list(dist_swap[uid][t].values()))
        jsds[uid, t] = jensenshannon(p, q, base=2)

# ================= PLOT (NO BOLD AXIS TITLES/LABELS) =================
fig = plt.figure(figsize=(9, 6))
ax = fig.gca()

im = ax.imshow(jsds, interpolation='nearest', origin='lower', cmap='viridis')

TITLE_FONTSIZE = 15
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 11
CELL_FONTSIZE = 10

ax.set_title("Effect of Card-Swap (JSD)", fontsize=TITLE_FONTSIZE)
ax.set_xlabel("Time step", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("User", fontsize=LABEL_FONTSIZE)

cbar = fig.colorbar(im, ax=ax)
cbar.ax.set_ylabel("JSD", rotation=-90, va="bottom", fontsize=LABEL_FONTSIZE)
cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

ax.set_xticks(range(JOURNEY_LENGTH))
ax.set_xticklabels(range(JOURNEY_LENGTH), fontsize=TICK_FONTSIZE)

ax.set_yticks(range(NUM_USERS))
ax.set_yticklabels(range(NUM_USERS), fontsize=TICK_FONTSIZE)

# dynamic text
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=jsds.min(), vmax=jsds.max())

for i in range(NUM_USERS):
    for j in range(JOURNEY_LENGTH):
        v = jsds[i, j]
        r, g, b, _ = cmap(norm(v))
        lum = 0.299*r + 0.587*g + 0.114*b
        col = "black" if lum > 0.5 else "white"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color=col, fontsize=CELL_FONTSIZE)

# Top scenario header (kept bold)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("jsd_cardswap_heatmap.pdf", bbox_inches='tight')
plt.show()

print("Saved: jsd_cardswap_heatmap.pdf")