# Apurva Badithela
# File to synthesize test cases for Easter Egg Hunt Grid framework shown in Easter_Egg_Hunt.png and Easter_Egg_Hunt_grid.png

import numpy as np
import random
from random import randrange
import importlib
import itertools
import pickle
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os
import math
import networkx as nx
import pdb
import inspect
from matplotlib import animation

import gridworld_class as gc
import Player_class as pc
import Game_graph_class as gg
import General_Game_Graph_class as ggg
import test_run_configuration as trc
from time import gmtime, strftime

# % ============== Configuring File Names to Save Data: ============== %
file_path = "Easter_Egg_Hunt/"
fname_matrix = file_path + "static_obstacle_matrix_#.dat"
fname_grid_w_prop = file_path + "initial_grid_#.png"
fname_grid_w_static_obs = file_path + "grid_static_obs_#.png"
fname_propositions = file_path + "propositions_#.dat"
fname_goal = file_path + "goal_#.dat"
fname_ani = file_path + "test_ani_#.avi"
fname_matrix = fname_matrix.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
fname_propositions = fname_propositions.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
fname_goal = fname_goal.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
fname_ani = fname_ani.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
fname_grid_w_prop = fname_grid_w_prop.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
fname_grid_w_static_obs = fname_grid_w_static_obs.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))

pkl_matrix = open(fname_matrix,"wb")
pkl_prop = open(fname_propositions, "wb")
pkl_goal = open(fname_goal,"wb")

### ================ Configuring the static test environment ========================== % ###
# % ================ Setting initial variables for static nodes ======================== %
M = 6 # No. of rows
N = 6 # No. of columns
Nprop = 1 # No. of propositions
Nmax_prop = 2 # Maximum number of nodes in each proposition

# Setting up goals and propositions:
goal_row = M # Row counted starting from 1 top to bottom
goal_col = N # Column counted starting from 1 left to right
sys_reach = [N*(goal_row-1) + goal_col]
nNodes = M*N
rand = "no"
if rand == "yes":
    nprop_to_cover = [random.choice(range(1,Nmax_prop+1)) for ii in range(Nprop)] # A list of numbers with each element between 1 and 3
    nNodes_to_cover = [[random.choice(range(1,nNodes+1)) for ii in range(nprop_to_cover[jj])] for jj in range(len(nprop_to_cover))] # Randomly choosing the vertices to cover
else:
    cell2node = lambda r, c: N*(r-1) + c
    nprop_to_cover = [2]
    cover_cells = [[3,6], [6,3]]
    nNodes_to_cover = [[cell2node(cell[0],cell[1]) for cell in cover_cells]]

# % =============== Construct gridworld states and transitions: ============= %
GW = gc.GridWorld([M,N])
GW.grid_transitions()
test_config = trc.test_run_configuration(GW) # Setting up a test configuration

print("System reachability goal: ")
print(*sys_reach)
print("Propositions to cover in testing: ")
for prop in nNodes_to_cover:
    print(*prop)
pickle.dump(nNodes_to_cover, pkl_prop)
pickle.dump(sys_reach, pkl_goal)
pkl_prop.close()
pkl_goal.close()

# % =============== Synthesizing static obstacles ========================= %
# Add propositions:
test_config.set_final_reach_goal(sys_reach)
test_config.set_propositions(nNodes_to_cover)

# Create Base plot of goal and proposition locations:
fig_num = 1
fig, ax, im = GW.base_plot(fig_num)
fig, ax, im = test_config.base_plot(fig, ax, im, nNodes_to_cover, sys_reach)
fig.savefig(fname_grid_w_prop, dpi=fig.dpi)
fig_num+=1

fig2, ax2, im2 = GW.base_plot(fig_num)
fig_num += 1
# Creating static obstacles and plotting them:
via = "Labels"
if via == "ValueFunc":
    cut_transitions, static_obstacles = test_config.generate_static_obstacles(nNodes_to_cover, sys_reach)
if via == "Labels":
    static_obstacles = test_config.generate_static_obstacles_via_labels(nNodes_to_cover, sys_reach)
test_matrix = test_config.update_matrix(static_obstacles)
fig2, ax2, im2 = test_config.base_plot(fig2, ax2, im2, nNodes_to_cover, sys_reach) # Constructing the base plot
fig2, ax2, im2 = test_config.static_obstacle_plot(fig2, ax2, im2, test_matrix[-1]) # Plotting static obstacles on the grid
fig2.savefig(fname_grid_w_static_obs, dpi=fig.dpi) # Save figure

# Files to save animation and test matrix:
pickle.dump(test_matrix[-1], open(fname_matrix,"wb"))
pkl_matrix.close()
# plt.show()

### ================== End of configuring the static test environment =================== ###
# ------------------------------------------------------------------------------------------#
### ================== Synthesizing Dynamic Test Strategy =============================== ###
# ToDo: Extend it to multiple coverage goals. For now, we can only do this for one adversary
# Environment specification comprises of environment behavior: agent can be present/absent in its cell
# ===============Initializing dynamic agent transitions ======================= #
switch_test_loc = nNodes_to_cover[0].copy()
switch_test_loc = [24, 34]
ntest_agents = len(switch_test_loc) # Number of dynamic agents that only appear and disappear at the nodes that have to be covered
states = [1,2] # 1: test agent is absent from its cell, 2: test agent is active in its cell

# Written by hand but this should be automatically configured somehow:
t1 = [[1,2], [1,2]] # State transition list for test agent in prop = 24
nt1 = len(t1)
t2 = [[1,2], [2]] # State transition list for test agent in prop = 34
nt2 = len(t2)

# state 1: both open; state 2: t2 = 2, t1 = 1; state 3= t2 =1, t1 = 2; state 4= t1 = 2 and t2 = 2
test_state_func = lambda t1_loc, t2_loc: len(states)*(t1_loc-1) + t2_loc # Function converting individual test agent states to test states
test_env_transitions = []
for st1 in range(1, nt1+1):
    for st2 in range(1, nt2+1):
        start = test_state_func(st1, st2)
        end_st = [test_state_func(e_st1, e_st2) for e_st2 in t2[st2-1] for e_st1 in t1[st1-1]]
        not_good_state = test_state_func(nt1, nt2)
        while not_good_state in end_st:
            end_st.remove(not_good_state)
        if (st1 == nt1 and st2 == nt2):
            end_st = [start]
        test_env_transitions.append(end_st)
print("Transitions of dynamic obstacle")
print(test_env_transitions)
# Comments: Ideally, these transitions should be derived from specifications of the dynamic agent
# Configuring the gridworld and adding agent transitions:
static_obstacles_cell = [GW.node2cell[s] for s in static_obstacles[0]]
GW.add_static_obstacles(static_obstacles_cell)

# Adding players and transitions:
GW.grid_transitions()
Tsys_list = [(1,[1,2,7]), (2,[1,2,3,8]), (3,[2,3,4,9]), (4,[3,4,5,10]), (5,[4,5,6,11]), (6,[5,6,12]), 
(7,[1,7,8,13]), (8,[2,7,8,9,14]), (9,[3,8,9,10,15]), (10,[4,9,10,11,16]), (11,[5,10,11,12,17]), (12,[6,11,12,18]), 
(13,[7,13,14, 19]), (14,[8,13,14,15,20]), (15,[9,14,15,16,21]), (16,[10,15,16, 17, 22]), (17, [11,16,17,18,23]), (18, [12,17,18,24]),
(19,[13,19,20,25]), (20,[14,19,20,21,26]), (21, [15,20,21,22,27]), (22,[16,21,22,23,28]), (23, [17,22,23,24,29]), (24,[18,23,24,30]),
(25, [19,25,26,31]), (26, [20, 25, 26,27, 32]), (27, [21,26,27,28,33]), (28, [22,27,28,29,34]), (29,[23,28,29,30,35]), (30, [24,29,30,36]),
(31, [25,31,32]), (32, [26,31,32,33]), (33, [27,32,33,34]), (34, [28,33,34,35]), (35, [29,34,35,36]), (36,[30,35,36])] # System transitions
Tsys = dict(Tsys_list)
sys_transitions = []
env_transitions = []
for key, val in Tsys.items():
    sys_transitions.append(val)
Tenv_list = [(1, [1,3,2]), (2,[2]), (3,[1,3]), (4, [4])]
Tenv = dict(Tenv_list)
for key, val in Tenv.items():
    env_transitions.append(val)
unsafe = [[3, 24], [4, 24], [2, 34], [4, 34]]
Ns = len(sys_transitions)
Ne = len(env_transitions)
## --------------------------------------- Setting game graph: -------------------------------------------------- ##
Game_Graph = ggg.GeneralGameGraph(static_obstacles[0], sys_transitions, env_transitions, unsafe)

# Removing some edges:
start1 = "v1_"+str(Game_Graph.state(Ns, Ne, 34, 1))
end1 = "v2_"+str(Game_Graph.state(Ns, Ne, 34, 2))
start2 = "v1_"+str(Game_Graph.state(Ns, Ne, 24, 1))
end2 = "v2_"+str(Game_Graph.state(Ns, Ne, 24, 3))
rem_edges = [[start1, end1], [start2, end2]]
Game_Graph.remove_edges(rem_edges)

# Unsafe Nodes:
U_nodes= []
for v in Game_Graph.U:
    p, ns, ne = Game_Graph.unpack_vertex(v)
    U_nodes.append([ne,ns])

# Finding the system winning set and value function:
sys_goal_nodes = [[1, 36], [2, 36], [3, 36]]
goal_vertices = Game_Graph.set_win_states(sys_goal_nodes)

# Robust Pre for system winning set computation:
quant_env = 'forall'
quant_sys = 'exists'
win_agent = 's'
Wsys2 = Game_Graph.win_reach(win_agent, goal_vertices, quant_env, quant_sys)
Val_sys = Game_Graph.get_value_function(win_agent) # System Value Function
# pdb.set_trace()

# Finding environment winning set and value function:
env_goal_nodes = [[1, 33], [2, 33], [3, 33], [1, 18], [2, 18], [3, 18]]
goal_vertices = Game_Graph.set_win_states(env_goal_nodes)
# Robust Pre for system winning set computation:
quant_env = 'exists'
quant_sys = 'exists'
win_agent = 'e'

# Setting weights:
coverage_props_list = [(lambda ns, ne: ns==33 or ns==18, 1)]
coverage_props_list = [(lambda ns, ne: ns==33, 1), (lambda ns, ne: ns==18, 1)]
coverage_props = dict(coverage_props_list)
Game_Graph.set_vertex_weight(coverage_props)

Wenv = Game_Graph.win_reach(win_agent, goal_vertices, quant_env, quant_sys)
Val_env = Game_Graph.get_value_function(win_agent) # Environment Value Function

# Looking at the nodes:
Wenv_max = Wenv[-1]
Wsys_max = Wsys2[-1]
Wenv_sys_nodes = []
Wenv_env_nodes = []
Wsys_sys_nodes = []
Wsys_env_nodes = []
for v in Wenv_max[0]:
    state = int(v[3:])
    ne, ns = Game_Graph.state2node(Ns, Ne, state)
    Wenv_env_nodes.append([ne,ns])
for v in Wenv_max[1]:
    state = int(v[3:])
    ne, ns = Game_Graph.state2node(Ns, Ne, state)
    Wenv_sys_nodes.append([ne,ns])
for v in Wsys_max[0]:
    state = int(v[3:])
    ne, ns = Game_Graph.state2node(Ns, Ne, state)
    Wsys_env_nodes.append([ne,ns])
for v in Wsys_max[1]:
    state = int(v[3:])
    ne, ns = Game_Graph.state2node(Ns, Ne, state)
    Wsys_sys_nodes.append([ne,ns])

# pdb.set_trace()
# ------------------------------------------------------------------------------------------------------------------------------- #
# Choosing initial condition:
ne_auto_0 = 2
ns_0 = 1
ne_0 = 1

q0 = "v1_"+str(Game_Graph.state(Ns, Ne, ns_0, ne_0))
# pdb.set_trace()
test_run = [q0]
qcur = q0
test_length = 70
next_turn = 'e'
NTOTAL = []
NCORRECT = []
cones = []
for t in range(1, test_length):
    q = test_run[t-1]
    turn = next_turn
    if turn == 'e':
        qn = Game_Graph.test_policy(q, Wsys2)
        test_run.append(qn)
        next_turn = 's'
    if turn == 's':
        qn, flag_win, ntotal, ncorrect = Game_Graph.agent_policy(q, cones,  Wsys2)
        NTOTAL.append(ntotal)
        NCORRECT.append(ncorrect)
        test_run.append(qn)
        next_turn = 'e'
        if flag_win: # If the game has been won stop simulation
            break

### ================== End of Synthesizing Dynamic Test Strategy =============================== ###
## ------------------------ Animating test run --------------------------------------------- ###
def get_scell(ns):
    if (ns%N == 0):
        col = N
        row = ns//N
    else:
        col = ns%N
        row = ns//N + 1
    return [row, col]

# Make this more general
def get_ecell(ne):
    if(ne == 1):
        row = []
        col = []
    if (ne == 2):
        cell = get_scell(24)
        row = cell[0]
        col = cell[1]
    if (ne==3):
        cell = get_scell(34)
        row = cell[0]
        col = cell[1]
    if (ne == 4):
        cell1 = get_scell(24)
        cell2 = get_scell(34)
        row = [cell1[0], cell2[0]]
        col = [cell1[1], cell2[1]]
    return [row, col]

skip_transitions = ['e']
test_run_cells = []
for v in test_run:
    state = int(v[3:])
    ne, ns = Game_Graph.state2node(Ns, Ne, state)
    scell = get_scell(ns)
    ecell = get_ecell(ne)
    test_state = [ecell, scell]
    test_run_cells.append(test_state)
pdb.set_trace()
anim = GW.animate_test_run_gg(test_run_cells, fig, ax, skip_transitions)
writervideo = animation.FFMpegWriter(fps=60)
anim.save(fname_ani, writer=writervideo)