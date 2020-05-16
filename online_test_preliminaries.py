# Function to synthesize a single test strategy:
# Environment goes first and then the system acts
# Test run is a state-action sequence: [e1, s1, e2, ...]
# Wf: Final winning set
import numpy as np
from random import randrange
import importlib
import pickle
import os

# Unsafe set:
def unsafe(collision_indices):
    U = []
    for ii in collision_indices:
        s = state(ii,ii)
        U.extend(["v1_"+str(s)])
        U.extend(["v2_"+str(s)])
    return U
# Generate transition system:
# U is the set of sink vertices. These vertices have no outgoing edges
def trans_sys(V1, V2, Vp, T1, T2, Tp, sink):
    if not Vp:
        GVp = []
    GEdges=[]
    # First all environment action transitions:
    for env_state in range(1,Ne+1):
        for end_env_state in T1[env_state-1]:
            for sys_state in range(1,Ns):
                start_state = state(sys_state, env_state)
                start_vtx = "v1_"+str(start_state)
                if start_vtx in sink:
                    continue
                end_state = state(sys_state, end_env_state)
                end_vtx = "v2_"+str(end_state)
                edge = [start_vtx, end_vtx, "ae"]
                GEdges.append(edge)
    # Now, all system action transitions:
    for sys_state in range(1,Ns+1):
        for end_sys_state in T2[sys_state-1]:
            for env_state in range(1,Ne):
                start_state = state(sys_state, env_state)
                start_vtx = "v2_"+str(start_state)
                if start_vtx in sink:
                    continue
                end_state = state(end_sys_state, env_state)
                end_vtx = "v1_"+str(end_state)
                edge = [start_vtx, end_vtx, "as"]
                GEdges.append(edge)
    return GVp, GEdges

# Generate winning sets for 2 player games:
def state(sys_loc, env_loc):
    return Ne*(sys_loc) + env_loc

# Given number of system and environment transitions:
def vertices(Ns, Ne):
    Vp =[]
    V1 = []
    V2 = []
    for xs in range(1,Ns+1):
        for xe in range(1,Ne+1):
            s = state(xs, xe)
            V1.extend(["v1_"+str(s)]) # Environment action vertices
            V2.extend(["v2_"+str(s)]) # System action vertices
    return V1, V2, Vp
    
# Main function to synthesize winning sets:
# Returns winning set for n steps
# N: Maximum number of iterations for fixed-point iterations
# Win_ii_n: The entire winning set upto the nth fixed point iteration
# Pre_ii_n: Consists only of the Pre set of states computed at the nth iteration

def synt_winning_set(GVp, GEdges, U, W0):
    W = [W0] # Winning set with 0 iterations
    Pre_cur = W0.copy()
    Win_cur = W0.copy()
    fixpoint = False
    while not fixpoint:
        Pre_ii = pre(GVp, GEdges, Pre_cur, U, 1, 0, 1)
        Pre_ii_1 = Pre_ii[0].copy()
        Pre_ii_2 = Pre_ii[1].copy()
        Win_cur_1 = Win_cur[0].copy()
        Win_cur_2 = Win_cur[1].copy()
        if Pre_ii_1: # If it is not empty
            Win_cur_1.extend(Pre_ii_1)
            Win_cur_1 = list(dict.fromkeys(Win_cur_1)) # Removes duplicates
        if Pre_ii_2: # If it is not empty
            Win_cur_2.extend(Pre_ii_2)
            Win_cur_2 = list(dict.fromkeys(Win_cur_2)) # Removes duplicates
        Win_ii_1 = Win_cur_1.copy()
        Win_ii_2 = Win_cur_2.copy()
        Win_ii = [Win_ii_1, Win_ii_2]
        W.append(Win_ii)
        if(Win_cur == Win_ii):
            fixpoint = True
        Pre_cur = Pre_ii.copy()
        Win_cur = Win_ii.copy()

    return W

# Synthesizing test cases:
# Environment states are the first place of states
# System state is the second place of states
def synt_winning_set2(GVp, GEdges, U, W0):
    W = [W0] # Winning set with 0 iterations
    W_env = [W0[0]] # Winning set with environment states
    W_sys = [W0[1]] # Winning set with system states
    Pre_cur = W0.copy()
    Win_cur = W0.copy()
    N = 0
    fixpoint = False
    while not fixpoint:
        Pre_ii = pre(GVp, GEdges, Pre_cur, U, 1, 0, 1)
        Pre_ii_1 = Pre_ii[0].copy()
        Pre_ii_2 = Pre_ii[1].copy()
        Win_cur_1 = Win_cur[0].copy()
        Win_cur_2 = Win_cur[1].copy()
        if Pre_ii_1: # If it is not empty
            Win_cur_1.extend(Pre_ii_1)
            Win_cur_1 = list(dict.fromkeys(Win_cur_1)) # Removes duplicates
        if Pre_ii_2: # If it is not empty
            Win_cur_2.extend(Pre_ii_2)
            Win_cur_2 = list(dict.fromkeys(Win_cur_2)) # Removes duplicates

        Win_ii_1 = Win_cur_1.copy()
        Win_ii_2 = Win_cur_2.copy()
        Win_ii = [Win_ii_1, Win_ii_2]
        Win_prev = Win_cur.copy()
        if(Win_ii_1 != Win_prev[0]):
            W_env.append(Win_ii_1)
        if(Win_ii_2 != Win_prev[1]):
            W_sys.append(Win_ii_2)
        if(Win_cur == Win_ii):
            fixpoint = True
        W.append(Win_ii)
        N += 1
        Pre_cur = Pre_ii.copy()
        Win_cur = Win_ii.copy()

    return W, W_env, W_sys

# ToDo: Fix the notation of W0 here...
# Defining Predecessor operator for synthesizing winning sets: 
# Assume: Player 1 is the environment and Player 2 is the System
# Winning sets would only contain environment action states
# Pre(S):= {x \in V2| \forall   }
# U: Unsafe set of states
# Qualifier notations: there_exists: 0 and forall: 1
def pre(GVp, GEdges, W0, U, qual1, qual2, qual3):
    if not GVp: # 2-player game winning set
        # Simple backward reachability:
        env_W0 = W0[0].copy() # First row of W0 has env action nodes in winning set
        sys_W0 = W0[1].copy() # Second row of W0 has sys action nodes in winning set
        Win1 = [] # Winning set containing environment action states
        Win2 = [] # Winning set containing system action states
        Win = [] # Winning set containing env winning actions in the first row and sys winning actions in the second row

        # Backward reachability for winning set with environment action state
        for env_win in env_W0:
            end_node = [row[1] for row in GEdges]
            env_win_idx = [ii for ii, x in enumerate(end_node) if x==env_win]
            start_node = [row[0] for row in GEdges] # Extracting the first column in G.Edges
            env_nbr = [start_node[ii] for ii in env_win_idx]
            if env_nbr: # If list is not empty
                for env_nbr_elem in env_nbr:
                    if env_nbr_elem not in U:  # Not in unsafe set
                        Win2.append(env_nbr_elem)

        # Backward reachability for winning set with system action state. All environment actions must lead to a winning state
        for sys_win in sys_W0:
            end_node = [row[1] for row in GEdges]
            potential_sys_win_idx = [ii for ii, x in enumerate(end_node) if x==sys_win]
            start_node = [row[0] for row in GEdges] # Extracting the first column in G.Edges                
            potential_sys_nbr = [start_node[ii] for ii in potential_sys_win_idx]
            sys_nbr = []
            for potential_nbr in potential_sys_nbr:
                if potential_nbr not in U:
                    potential_nbr_idx = [ii for ii, x in enumerate(start_node) if x==potential_nbr]
                    potential_nbr_end_node = [end_node[ii] for ii in potential_nbr_idx]
                    if set(potential_nbr_end_node) <= set(sys_W0):
                        sys_nbr.extend([potential_nbr])
         
            Win1.extend(sys_nbr) 
        Win1 = list(dict.fromkeys(Win1)) # Removes duplicates
        Win2 = list(dict.fromkeys(Win2)) # Removes duplicates
        Win.append(Win1)
        Win.append(Win2)

    else: # Find sure, almost-sure and positive winning sets
        Win=[]
    return Win

def get_state(state):
    if state%Ne == 0:
        env_state = Ne
        sys_state = state/Ne - 1
    else:
        env_state = state%Ne
        sys_state = state//Ne
    return env_state, sys_state

# Retrieve states:
def retrieve_win_states(W, W_V1, W_V2):
    sys_win = []
    env_win = []
    # Counters to check that the number of environment and system winning vertices are correctly recorded in W_V1 and W_V2
    n_sys_win = 0
    n_env_win = 0
    for ii in range(0,N):
        W_ii = W[ii].copy()
        env_action_states = W_ii[0].copy()
        sys_action_states = W_ii[1].copy()
        sys_win_ii = []
        env_win_ii = []
        for ee in env_action_states:
            ee_idx = V1.index(ee)
            env_temp_ptr = W_V1[ee_idx]  # Temporary pointer to point to the env winning list
            if env_temp_ptr[2]==-1:      # If there is not a winning set assignment yet, make an assignment
                env_temp_ptr[2] = ii
                n_env_win += 1
            s = int(ee[3:])
            [env_st, sys_st] = get_state(s)
            env_win_ii.append([env_st, sys_st])
        for ss in sys_action_states:
            ss_idx = V2.index(ss)
            sys_temp_ptr = W_V2[ss_idx]   # Temporary pointer to the sys winning list
            if sys_temp_ptr[2]==-1:
                sys_temp_ptr[2] = ii
                n_sys_win += 1
            s = int(ss[3:])
            [env_st, sys_st] = get_state(s)
            sys_win_ii.append([env_st, sys_st])

        # Assertion to check that all winning states have been correctly stored in W_V1 and W_V2:
        assert(n_env_win == len(env_action_states))
        assert(n_sys_win == len(sys_action_states))

        sys_win.append(sys_win_ii)
        env_win.append(env_win_ii)

    assert(len(env_win[N-1]) == n_env_win)
    assert(len(sys_win[N-1]) == n_sys_win)
    return env_win, sys_win

# Retrieve states2:
# Same function as above but based on winning sets containing purely environment states and purely system states
def retrieve_win_states2(W_sys, W_env, W_V1, W_V2):
    sys_win = []
    env_win = []
    # Counters to check that the number of environment and system winning vertices are correctly recorded in W_V1 and W_V2
    n_sys_win = 0
    n_env_win = 0
    # N is the length of the winning set:
    for ii in range(0,len(W_env)):
        env_action_states = W_env[ii].copy()
        env_win_ii = []
        for ee in env_action_states:
            ee_idx = V1.index(ee)
            env_temp_ptr = W_V1[ee_idx]  # Temporary pointer to point to the env winning list
            if env_temp_ptr[2]==-1:      # If there is not a winning set assignment yet, make an assignment
                env_temp_ptr[2] = ii
                n_env_win += 1
            s = int(ee[3:])
            [env_st, sys_st] = get_state(s)
            env_win_ii.append([env_st, sys_st])
        env_win.append(env_win_ii)

    for ii in range(0, len(W_sys)):
        sys_action_states = W_sys[ii].copy()
        sys_win_ii = []
        for ss in sys_action_states:
            ss_idx = V2.index(ss)
            sys_temp_ptr = W_V2[ss_idx]   # Temporary pointer to the sys winning list
            if sys_temp_ptr[2]==-1:
                sys_temp_ptr[2] = ii
                n_sys_win += 1
            s = int(ss[3:])
            [env_st, sys_st] = get_state(s)
            sys_win_ii.append([env_st, sys_st])
        sys_win.append(sys_win_ii)

    assert(n_env_win == len(env_action_states))
    assert(n_sys_win == len(sys_action_states))
    return env_win, sys_win

# Winning sets : [W0_sys]
# ADDITION on 3/17/20:
# Edge information for test agent
# GEdges are the edges on the game graph
# N: No. of winning sets
# env_win: Env action states in winning sets
# sys_win: Sys action states in winning sets
# Returns edge information for all edges from an environment action state
# Edge weight of 0 for transition leaving winning set, one for staying in the same winning set, and 2 for moving into the next winning set
# ToDo: Make the data sructures more like dictionaries so that they can be easily read. Learn using pandas

def edge_info(GEdges, N, sys_win, env_win, W_V1, W_V2):
    env_edge_info = [GEdges[ii] for ii in range(len(GEdges)) if GEdges[ii][2]=='ae'] # Env. action states
    sys_edge_info = [GEdges[ii] for ii in range(len(GEdges)) if GEdges[ii][2]=='as'] # Sys. action states
    # env_edge_info_len = [0 for ii in range(len(env_edge_info))] # Size of each env_edge_info row
    # sys_edge_info_len = [0 for ii in range(len(sys_end_info))] # Size of each sys_edge_info row
    V1 = [row[0] for row in W_V1] # Winning env action vertices
    V2 = [row[0] for row in W_V2] # Winning system action vertices
    # Edges with environment as the first state and system as the successor
    for env_edge in env_edge_info:
        # Finding start and end nodes of the edge:
        # Add if successor vertex is in winning sets
        assert(len(env_edge)==3)
        start = env_edge[0]
        succ = env_edge[1]
        # Add number of transitions.
        # Initially the number of transitions is 0
        env_edge.append(0)
        start_idx = V1.index(start)
        succ_idx = V2.index(succ)
        env_win_set = W_V1[start_idx][2]
        sys_win_set = W_V2[succ_idx][2]
        # If the successor remains in the same winning set, the env edge information records this as 1
        if ((sys_win_set>-1) and (env_win_set>-1)):
            if(env_win_set > sys_win_set): # Better because you're in the smaller attractor
                env_edge.append(2)
            elif(env_win_set == sys_win_set):
                env_edge.append(1)
            else:                          # Bad because you're in a bigger attractor, need to have a different number than the previous case
                env_edge.append(1)         # To Do: This needs to change
        elif ((sys_win_set>-1) and not (env_win_set>-1)):
            env_edge.append(2)
        else:
            env_edge.append(0) # Action leading outside winning set
        # Testing code:
        if(len(env_edge)!=5):
            #print(env_edge)
            assert(len(env_edge)==5)
    for sys_edge in sys_edge_info:
        assert(len(sys_edge)==3)
        # Finding start and end nodes of the edge:
        # Add if successor vertex is in winning sets
        start = sys_edge[0]
        succ = sys_edge[1]
        # Add number of transitions.
        # Initially the number of transitions is 0
        sys_edge.append(0)
        start_idx = V2.index(start)
        succ_idx = V1.index(succ)
        env_win_set = W_V1[succ_idx][2]
        sys_win_set = W_V2[start_idx][2]
        # If the successor remains in the same winning set, the env edge information records this as 1
        if ((sys_win_set>-1) and (env_win_set>-1)):
            if(sys_win_set > env_win_set): # Better because you're in the smaller attractor
                sys_edge.append(2)
            elif(env_win_set == sys_win_set): # Ok because you're in the same attractor
                sys_edge.append(1)
            else:                          # Bad because you're in a bigger attractor
                sys_edge.append(1)         # To Do: This needs to change
        elif ((env_win_set>-1) and not (sys_win_set>-1)):
            sys_edge.append(2)
        else:
            sys_edge.append(0) # Action leading outside winning set
        # Testing code:
        if(len(sys_edge)!=5):
            #print(sys_edge)
            assert(len(sys_edge) == 5)
    return env_edge_info, sys_edge_info


# Return edges where v is the starting vertex. The vertex input is in the form of [e,s]
# Player is either the system or the environment
# Returns the edges starting from v and their indices in sys_edge_info
def edge(v, edge_info, player):
    v_edges = []
    st = set_state(v, player)
    v_edges = [edges for edges in edge_info if edges[0]==st]
    v_edges_idx = [ii for ii in range(len(edge_info)) if edge_info[ii][0]==st]
    return v_edges, v_edges_idx

# Input v: vertex containing system and environment information
#        player: 'e' or 's' for environment or system
# Output: Returns vertex

def set_state(v, player):
    x = state(v[1], v[0]) # Feed in the system location and environment location
    if(player == 'e'):
        st = "v1_"+str(x)
    elif(player == 's'):
        st = "v2_"+str(x)
    else:
        print("Error in set_state: Input either 's' (system) or 'e' (environment) for the player variable.")
        st = []
    return st


# Initialize the main graph game:
### Main components of the file:
# Runner blocker example


# Transition system
# Need to manually write transitions
# T1: Transitions of environment
# T2: Transitions of system
# Example 1: 4-by-4 gridworld
ex = 1
if ex == 1:
    Ns = 16
    Ne = 16
    V1, V2, Vp = vertices(Ns,Ne)
    T1 = [[1], [3,6], [2,7], [4], [5], [2,10], [3,11], [8], [9], [6,14], [7,15], [12], [13], [10,15], [11, 14], [16]] #Environment transitions
    T2 = [[1,2,5], [1,2,3,6], [2,3,4,7], [3,4,8], [1,5,6,9], [2,5,6,7,10], [3,6,7,8,11], [4,7,8,12], [5,9,10,13], [6,9,10,11,14], [7,10,11,12,15], [8,11,12,16], [9,13,14], [10,13,14,15], [11,14,15,16], [12,15,16]] # System transitions
    Tp = []
    # Unsafe set
    e_states = [2,3,6,7,10,11,14,15]
    collision_indices = e_states # Collision can occur in one of environment states
    U = unsafe(collision_indices)
# Example 2: Runner blocker:
elif ex == 2:
    Ns = 5
    Ne = 5
    V1, V2, Vp = vertices(Ns,Ne)
    T1 = [[1], [3], [2,4], [3], [5]]
    T2 = [[1,2,3,4],[1,2,3,5], [1,2,3,4,5], [1,3,4,5], [2,3,4,5]]
    Tp = []
    e_states = [2,3,4]
    collision_indices = e_states # Collision can occur in one of environment states
    U = unsafe(collision_indices)
else:
    print("Let example be either 1 or 2")

# Graph transition
GVp, GEdges = trans_sys(V1, V2, Vp, T1, T2, Tp, U)

# Initial Winning set:
sys_W0 = [] # Winning states from system action states
env_W0 = [] # Winning states from environment action states
# Goal states of system and environment
env_goal = e_states
sys_goal = [1]
for env_state in env_goal:
    for sys_state in sys_goal:
        w0 = state(sys_state, env_state)
        sys_W0.extend(["v2_"+str(w0)])
        env_W0.extend(["v1_"+str(w0)])
W0 = [env_W0, sys_W0]

# No. of winning sets
W = synt_winning_set(GVp, GEdges, U, W0)
N = len(W)

W, W_env, W_sys = synt_winning_set2(GVp, GEdges, U, W0)

# Reachability: 
# If this is a winning set, it is a winning set only ... 
# Create copies of vertices to store their winning set information:
# -1 denotes that the vertex has not yet been designated in a winning set
W_V1 = [[v1, get_state(int(v1[3:])), -1] for v1 in V1] # Env. action states
W_V2 = [[v2, get_state(int(v2[3:])), -1] for v2 in V2] # Sys. action states
W_Vp = [[vp, get_state(int(vp[3:])), -1] for vp in Vp] # Prob. action states

# env_win, sys_win = retrieve_win_states(W, W_V1, W_V2) # Retrieves winning sets of states in env winning set and system winning set
env_win, sys_win = retrieve_win_states2(W_sys, W_env, W_V1, W_V2) # Retrieves winning sets of states in env winning set and system winning set
# Finding edge information to then pass into synth_test_strategies:
env_edge_info, sys_edge_info = edge_info(GEdges, N, sys_win, env_win, W_V1, W_V2)
online_test_runs = []
# Save using pickle:
# Saving edge_info, env_win, sys_win, W, W_V1, W_V2, W0, ...
# Perhaps clears all variables:
os.system('CLS')
pickle.dump(GVp, open("GVp.dat","wb"))
pickle.dump(GEdges, open("GEdges.dat","wb"))
pickle.dump(env_edge_info, open("env_edge_info.dat","wb"))
pickle.dump(sys_edge_info, open("sys_edge_info.dat","wb"))
pickle.dump(env_win, open("env_win_set.dat","wb"))
pickle.dump(sys_win, open("sys_win_set.dat","wb"))
pickle.dump(W_V1, open("W_V1.dat","wb"))
pickle.dump(W_V2, open("W_V2.dat","wb"))
pickle.dump(W, open("W.dat","wb"))
pickle.dump(W0, open("W0.dat","wb"))
pickle.dump(e_states, open("e_states.dat","wb"))
pickle.dump(online_test_runs, open("online_test_runs.dat", "wb"))