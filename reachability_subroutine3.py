## % Apurva Badithela
## % 5/22/2020
## Optimal strategy synthesis for reachability goals of the tester where the tester maximizes the time taken to reach a goal while the agent tries to minimize it.
## Using dynamic programming to find the optimal strategy algorithm
# This reachability subroutine algorithm is used in the examples where the environment has no stopping strategies

import numpy as np 
import random
import networkx as nx
import pickle
import pdb 
import inspect
from gridworld1_modified import system_K
from grid_construct import plot_grid_world
from grid_construct import base_plot

# Filenames:
fn_GVp = "GVp_ex4.dat"
fn_GEdges = "GEdges_ex4.dat"
fn_sys_edge_info = "sys_edge_info_ex4.dat"
fn_env_edge_info = "env_edge_info_ex4.dat"
fn_env_win_set = "env_win_set_ex4.dat"
fn_sys_win_set = "sys_win_set_ex4.dat"
fn_W_V1 = "W_V1_ex4.dat"
fn_W_V2 = "W_V2_ex4.dat"
fn_W = "W_ex4.dat"
fn_W0 = "W0_ex4.dat"
fn_online_test_runs = "online_test_runs_ex4.dat"
fn_e_states = "e_states_ex4.dat"

# Load data:
GEdges = pickle.load(open(fn_GEdges, "rb"))

# W_V1 and W_V2 are winning sets for each of the players with respect to the specification of the system
W_V1 = pickle.load(open(fn_W_V1,"rb"))
W_V2 = pickle.load(open(fn_W_V2,"rb"))
env_win_set = pickle.load(open(fn_env_win_set, "rb"))
sys_win_set = pickle.load(open(fn_sys_win_set, "rb"))
WA = pickle.load(open(fn_W,"rb"))
V1 = [row[0] for row in W_V1]
V2 = [row[0] for row in W_V2]

# System variables:
sys_var = "Xr"
fn = "reach_subroutine2_grid10_prop_4.avi"

# Create graph for this example:
# Makes the graph for the gridworld example:
# Also returns a node_dict for which the input is a number and the output is the node name with the v1/v2 prefixes
# V1: Nodes from which tester takes an action
# V2: Nodes from which agent takes an action
def make_graph(V1, V2, GEdges):
    G = nx.Graph()
    V_list = V1 + V2
    node_list = []
    for ii in range(len(V_list)):
        node_list.append((ii, V_list[ii]))
    node_dict = dict(node_list)
    N = [ii for ii in range(len(V1) + len(V2))]
    E = []
    # Constructing a tuple of numbers:
    for e in GEdges:
        e0 = e[0]
        e1 = e[1]
        n0 = V_list.index(e0)
        n1 = V_list.index(e1)
        E.append((n0, n1))
    G.add_edges_from(E)
    G.add_nodes_from(N)
    return G, node_dict

# Defining Propositions:
def propositions(p):
    state_to_props = []
    states_marked = [] # Keep list of tester states that have already been assigned priorities
    for ii in range(len(p)):
        lambda_prop = p[ii]
        q_lambda_sat = [] # Keep track of set of states that satisfy the proposition lambda_prop
        # States to be covered are environment states because we want to prompt the system to cover them 
        for row in W_V1:
            v = row[1]
            xe = v[0]
            xs = v[1]
            if(lambda_prop(xs, xe) and (v not in states_marked)):
                q_lambda_sat.append(row[0])
                states_marked.append(row[0])
        state_to_props.append((ii, q_lambda_sat))
    P = dict(state_to_props)
    return P

## % Main contents of file:
# Create graph G and return node dictionary connecting a number with a vertex:
# That is, 0 == V1[0], 1==V1[1], ....
# If l = len(V1), l = V2[0], l+1 = V2[1], and so on ...
G, node_dict = make_graph(V1, V2, GEdges)
num_node_dict = {v: k for k, v in node_dict.items()}
example = 4 # Which example the test run should be synthesized for. example = 2 corresponds to the 4-by-4 grid. example = 3 corresponds to the 10-by-10 grid
if example == 2:
    p = [lambda x_s, x_e: (x_s == x_e + 1), lambda x_s, x_e: (x_s == x_e - 1), lambda x_s, x_e: (x_s == x_e + N), lambda x_s, x_e: (x_s == x_e - N), lambda x_s, x_e: (x_s == 4)]
if example == 3:
    p = [lambda x_s, x_e: (x_s == x_e + 1), lambda x_s, x_e: (x_s == x_e - 1), lambda x_s, x_e: (x_s == x_e + N), lambda x_s, x_e: (x_s == x_e - N), lambda x_s, x_e: (x_s == 4)]
if example == 4:
    M = 7
    N = 7
    bridge1 = N-2
    bridge2 = (M-1)*N  + (N-2)
    p = [lambda x_s, x_e: ((x_s <= x_e - 1) and (x_e == bridge1)), lambda x_s, x_e: ((x_s <= x_e - 1) and (x_e == bridge2 and x_s>=40))]
    static_obs = []
    mov_obs_col = N-2
    for ii in range(2, M):
        static_obs.append([ii, mov_obs_col-1])
        if ((ii == 2) or (ii == M-1)):
            static_obs.append([ii, mov_obs_col-2])
P = propositions(p)
W_A_T_max = [num_node_dict[w] for w in WA[-1][0]] # Maximum winning set with tester nodes for the agent's specification

# Define weight function for tester nodes depending on the proposition they hold:
def weight(s):
    w = 0 # Default; no label
    node = node_dict[s]
    for ii in range(len(p)):
        states_ii = P[ii]
        if(node in states_ii):
            w = ii
            break
    return w

# Each tester state has a cost c, a cost-to-go function V, and strategy of the environment pi_env
# G is the goal set that we're trying to reach: <>S
# pi_env is the strategy that optimizes: J_k(x) = min_{u in U} max_{v in V} {g(x,u,v) + J_(k-1)}
# Here we only deal with tester nodes:
# Maximal winning set: W = [W0, pre(W0), pre(pre(W0)), ...] The union of all the pre's is the winning set.
def compute_winning_set(S, G):
    nodes = G.nodes # Numbered 0 to ln(nodes)
    V = [[] for ii in range(len(nodes))] # Cost-to-go function for each node in the graph G
    pi_env = [[] for ii in range(len(nodes))] # Environment strategy for each tester node in the graph G
    for s in S:
        V[s] = weight(s)
        pi_env[s] = s
    W0 = S
    W = []
    preS, V = pre(W0, V, G) # Computes {s in V_T | \forall aa \in A_A, \exists at in A_T s.t. T(s, aa, at) \in W0}
    while preS != W0:
        W.append(W0)
        for s in preS:
            successor = max_successor(s, V, G) # returns successor with the most weight
            V[s] = weight(s) + V[successor]
            pi_env[s] = successor
        W0 = preS
        preS, V = pre(W0, V, G)
    return W, V, pi_env

def pre(W0, V, G):
    Edges = G.edges
    pre_agent = []
    preS = []
    Vnew = V.copy()
    # Computing the agent vertices that are predecessors to W0. Assuming only tester vertices are in W0:
    for s in W0:
        pre_s = [e[0] for e in Edges if e[1]==s]
        pre_s_succ = [[e[1] for e in Edges if e[0]==p] for p in pre_s]
        bool_pre_s = [set(ps_succ) <= set(W0) for ps_succ in pre_s_succ] # Check if all successors of agent node lead to the winning set of the environment
        for ii in range(len(pre_s)):
            if (bool_pre_s[ii] and (pre_s[ii] not in pre_agent)):
                Vnew[pre_s[ii]] = min([V[succ] for succ in pre_s_succ[ii]])
                pre_agent.append(pre_s[ii])
    
    # Computing tester vertices that are predecessors to the agent vertices:
    for s_a in pre_agent:
        # Updating the cost-to-go function:
        # a_successors = [e[1] for e in Edges if e[0]==s_a]
        # V[s_a] = max([V[a] for a in a_successors]) # Max. of the cost-to-go function of its successors
        a_pre = [e[0] for e in Edges if e[1]==s_a]
        for p in a_pre:
            if p not in preS:
                preS.append(p)
    W_A_preS = list(set(preS) & set(W_A_T_max))
    return W_A_preS, Vnew

# For a tester node, this function returns the minimum weight successor to s in the game graph G.
def max_successor(s, V, G):
    Edges = G.edges
    successors = [e[1] for e in Edges if e[0]==s]
    successor_weight = [V[s] for s in successors] # This is the cost-to-go function for all agent successor nodes of the current vertex which is a tester node
    feas_successor = [s for s in successor_weight if (s != [])] # We're only interested in successors that have a feasible cost-to-go; which are ofcourse, successors in the winning set
    if not successor_weight:
        print("s:")
        print(s)
        # pdb.set_trace()
    min_weight = max(feas_successor) # Finds the minimum weight successor from all feasible transitions. Feasible = transition that continues to be in the winning set
    successor = successors[successor_weight.index(min_weight)]
    return successor

# Given a set of winning nodes W_N in format "v1_20", return a set of states with numbers that correspond to the winning nodes
# Furthermore, it will return the set of winning states that are also in the winning set of the agent: <A>W^T_max
def win_set(W_N):
    W0_T = [num_node_dict[k] for k in W_N]
    S = list(set(W0_T) & set(W_A_T_max))
    return S

# Constructing the set of winning states for each proposition:
for ii in range(len(P)):
    S = win_set(P[ii])            
    W, V, pi_env = compute_winning_set(S, G)
    prp = p[ii]
    print("Set of propositions: ")
    print(prp)
    print("Corresponding maximal winning set for the tester: ")
    print(W)

## Identifying initial condition for the tester for the atomic proposition prp you want to cover:
def test_strategy(prp):
    S = win_set(prp)
    W, V, pi_env = compute_winning_set(S, G)
    max_winning_set = W[-1]
    nW = 0
    n_pi_env = 0
    for ii_pi_env in pi_env:
        if ii_pi_env:
            n_pi_env += 1
    # Taking the union of elements in Wi:
    Wunion = []
    for Wi in W:
        for w in Wi:
            if w not in Wunion:
                Wunion.append(w) 
    nW = len(Wunion)
    assert(nW == n_pi_env) # Assertion to check that the winning set and pi_env have the same number of vertices for which there should be a strategy
    # Find a starting vertex with the least cost-to-go:
    V_win = [V[ii] for ii in max_winning_set]
    max_win_index = V_win.index(max(V_win)) # Find the initial condition with the most cost-to-go
    v0 = max_winning_set[max_win_index]
    return v0, pi_env

# Get the vertex (ENV, SYS) representation from the node/number form:
def get_vertex(v0, p):
    v0_vertex = node_dict[v0]
    # if("v1" == v0_vertex[0:2]):
    #     p = 'e'
    # else:
    #     p = 's'
    if(p == 'e'):
        v0_v = [s[1] for s in W_V1 if s[0] == v0_vertex]
    if(p == 's'):
        v0_v = [s[1] for s in W_V2 if s[0] == v0_vertex]
    return [v0_v[0][0], v0_v[0][1]] # return list form

# Set the vertex representation from the node form to a number form:
def set_vertex(v0, player):
    if(player == 's'):
        for s in W_V2:
            v = s[1]
            if(v0[0] == v[0] and v0[1]==v[1]):
                q0_v = s[0]
    if(player == 'e'):
        for s in W_V1:
            v = s[1]
            if(v0[0] == v[0] and v0[1]==v[1]):
                q0_v = s[0]
    q = num_node_dict[q0_v]
    return q

# System controller:
def system_controller(start, sys_control, cone_locs):
    env_pos = start[0]
    sys_pos = start[1]
    u = system_K.move(sys_control, env_pos, cone_locs)
    sys_pos = u[sys_var]
    finish = [env_pos, sys_pos]
    return finish

# Checking if a proposition is satisfied by a pair of coordinates:
# Inputs to the propositions are (xs, xe) and the coordinates are in the form (xe, xs)
def check_sat_spec(prop, vert):
    return prop(vert[1], vert[0])

## Constructing system and tester strategies:
prp = P[1]
q0, pi_env = test_strategy(prp)
q0_vertex = get_vertex(q0, 'e')
pi_sys = system_K(q0_vertex)

## Constructing a test run:
# Variables: *q0: old vertex number
#            *q0_vertex: old vertex in a tuple form: (env, sys)
#            *q: new vertex number
#            *q_vertex: new vertex in tuple form: (env, sys)
T = 20 # Horizon. Each player gets T turns
step = 0
q = q0 # v0_v is the vertex corresponding to the initial vertex
q_vertex = q0_vertex # Number equivalent of the vertex
traj = [q0_vertex] # Keeping track of trajectory
player = 'e'
cone_locations = []
# The following list of propositions indicates the list of atomic propositions the tester and agent need to satisfy. If both of them satisfy it, then we can move on:
tester_sat_prop = lambda x_s, x_e: (x_s == x_e - 1)
agent_sat_prop = lambda x_s, x_e: (x_s == M*N)

# Booleans to check if the tester and the system have both satisfied their specifications:
# Specifications are reached in environment states
tester_sat_spec = 0
agent_sat_spec = 0
tester_sat_spec = check_sat_spec(tester_sat_prop, q0_vertex)
sat_spec = [[0,0] for ii in range(T)] # First coordinate tracks tester specification; second coordinate tracks agent specification 
while step < T:
    if(step>0):
        sat_spec[step] = sat_spec[step-1] # If a spec has been satisfied, then it has been satisfied. 
    if(player == 'e'):
        q = pi_env[q0]
        q_vertex = get_vertex(q, 's')
        q0 = q # v0_v is the vertex corresponding to the initial vertex
        q0_vertex = q_vertex # Number equivalent of the vertex
        player = 's'
        traj.append(q0_vertex)
    if(player == 's'):    
        cone_locs = [] # Cone locations are empty
        q_vertex = system_controller(q0_vertex, pi_sys, cone_locs)
        q = set_vertex(q_vertex, player)
        q0 = q
        q0_vertex = q_vertex
        player = 'e'
        traj.append(q0_vertex)
    agent_sat_spec = check_sat_spec(agent_sat_prop, q0_vertex)
    if (agent_sat_spec and ~sat_spec[step][1]):
        sat_spec[step][0] = 1

    tester_sat_spec = check_sat_spec(tester_sat_prop, q0_vertex)
    if (tester_sat_spec and ~sat_spec[step][0]):
        sat_spec[step][0] = 1
    # If all propositions have been satisfied, then break out of the loop
    if sat_spec[step]:
        break
    step = step + 1

## Simulating a test run using matplotlib:
env_traj = [v[0] for v in traj]
sys_traj = [v[1] for v in traj]
anim = plot_grid_world(env_traj, sys_traj, static_obs, M, N)
anim.save(fn)
