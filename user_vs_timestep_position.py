#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

# Final Correct Visualisation: Pre-Swap State per Time Step

import random
import string
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, patheffects as pe
from matplotlib.patches import Patch
 
# Do not generate Type 3 fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Graph
edges = [
    ("V1","V2"),("V1","V5"),("V2","V1"),("V2","V3"),("V2","V6"),
    ("V3","V2"),("V3","V4"),("V3","V7"),("V4","V3"),("V4","V8"),
    ("V5","V6"),("V5","V9"),("V6","V5"),("V6","V2"),("V6","V7"),("V6","V10"),
    ("V7","V6"),("V7","V3"),("V7","V8"),("V7","V11"),
    ("V8","V7"),("V8","V4"),("V8","V12"),
    ("V9","V10"),
    ("V10","V6"),("V10","V9"),("V10","V11"),("V10","V13"),
    ("V11","V10"),("V11","V7"),("V11","V12"),("V11","V14"),
    ("V12","V8"),("V12","V11"),
    ("V13","V10"),("V13","V14"),
    ("V14","V11"),("V14","V13"),("V14","V12")
]

G: nx.DiGraph[str] = nx.DiGraph()
G.add_edges_from(edges)
 
# Parameters
rng = random.Random(42)
NUM_USERS, JOURNEY_LENGTH = 10, 10
ID_LETTERS = list(string.ascii_uppercase[:NUM_USERS])
 
def make_user_starts(g: nx.DiGraph[str], start: str, count: int, loc: str):
    nodes = list(g.nodes())
    return {i: (loc if loc != "random" else rng.choice(nodes)) for i in range(start, start + count)}
 
USER_START_LOCATION = (
    make_user_starts(G, 0, 2, "V1") |
    make_user_starts(G, 2, 3, "V2") |
    make_user_starts(G, 5, 5, "random")
)
 
#  Journey Simulation
def decide_stop(g, uid, t, current):
    nxt = list(g.successors(current)) or list(g.nodes())
    return rng.choice(nxt)
 
def simulate_journeys(g: nx.DiGraph[str]):
    paths=[]
    for uid in range(NUM_USERS):
        cur=USER_START_LOCATION[uid]
        path=[cur]
        for t in range(1,JOURNEY_LENGTH):
            cur=decide_stop(g,uid,t,cur)
            path.append(cur)
        paths.append(path)
    return paths
 
# Strict Same-Node Single-Swap Model
def simulate_with_strict_swaps(g: nx.DiGraph[str]):
    users_path = simulate_journeys(g)
    current_id = {u: ID_LETTERS[u] for u in range(NUM_USERS)}
    id_timeline = [[None]*JOURNEY_LENGTH for _ in range(NUM_USERS)]
    swap_events=[]
 
    for t in range(JOURNEY_LENGTH):
        # record state BEFORE swap
        for u in range(NUM_USERS):
            id_timeline[u][t] = current_id[u]
 
        # now perform swaps (apply after logging pre-swap state)
        node_users={}
        for u in range(NUM_USERS):
            node=users_path[u][t]
            if u!=0:
                node_users.setdefault(node, []).append(u)
 
        swapped=set()
        new_id=current_id.copy()
        for node,uids in node_users.items():
            candidates=[u for u in uids if u not in swapped]
            if len(candidates)>=2:
                u1,u2=rng.sample(candidates,2)
                new_id[u1],new_id[u2] = current_id[u2], current_id[u1]
                swapped.update({u1,u2})
                swap_events.append((t,node,u1,u2,current_id[u1],current_id[u2]))
        current_id=new_id
 
    return users_path, id_timeline, swap_events
 
# Run Simulation
users_path,id_timeline,swaps=simulate_with_strict_swaps(G)
 
print("Swap Log:")
for t,node,u1,u2,id1,id2 in swaps:
    print(f"t={t:2d} node={node:>3} users {u1:<2d}<->{u2:<2d}  IDs {id1}↔{id2}")
 
# Plot (correct pre-swap view)
def col(L: str):
    return list(cm.tab20.colors)[ID_LETTERS.index(L)%20]
 
fig, ax = plt.subplots(figsize=(12,6))
ax.set_facecolor("white")
 
# coloured squares + labels (state before swaps)
for u in range(NUM_USERS):
    for t in range(JOURNEY_LENGTH):
        L = id_timeline[u][t]
        node = users_path[u][t]
        ax.scatter(t,u,s=300,marker='s',edgecolor='k',facecolor=col(L),zorder=3)
        ax.text(t,u+0.1,L,ha='center',va='center',fontsize=11,
                path_effects=[pe.withStroke(linewidth=2,foreground='white')])
        ax.text(t,u-0.38,node,ha='center',va='center',fontsize=9,
                bbox=dict(boxstyle='round,pad=0.25',fc='white',ec='0.3'))
 
# Lines showing post-swap connections (t → t+1)
for L in ID_LETTERS:
    xs = []
    ys = []
    for t in range(JOURNEY_LENGTH-1):
        holder_t = [u for u in range(NUM_USERS) if id_timeline[u][t]==L][0]
        holder_next = [u for u in range(NUM_USERS) if id_timeline[u][t+1]==L][0]
        xs.extend([t,t+1])
        ys.extend([holder_t,holder_next])
    ax.plot(xs,ys,color=col(L),linewidth=2,alpha=0.95,zorder=1)
 
# Axes / legend
ax.set_title("Card Identity per Physical User Over Time")
ax.set_xlabel("Time Step ")
ax.set_ylabel("User")
ax.set_xticks(range(JOURNEY_LENGTH))
ax.set_yticks(range(NUM_USERS))
ax.set_yticklabels([f"{u}" for u in range(NUM_USERS)])
ax.grid(True,axis='x',linestyle=':',alpha=0.35)
 
ax.legend(
    handles=[Patch(facecolor=col(L),edgecolor='k',label=f"ID {L}") for L in ID_LETTERS],
    title="Starting IDs",
    bbox_to_anchor=(1.02,1),
    loc='upper left'
)
 
plt.tight_layout()
plt.savefig("user_vs_timestep_position_V5.pdf",bbox_inches='tight')
print("\n Saved: user_vs_timestep_position_V5.pdf")
