# Import files:
import numpy as np
from random import randrange
from gridworld1 import system_K
import importlib
import pickle
import sys
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation

# Variable and file names:
# Filenames for stored system controllers and related variables:
sys_var = "Xr" # This is the system variable that goes in the runner function
fname = "gridworld1"
cname = "system_K"

# Import files:
env_edge_info = pickle.load(open("env_edge_info.dat", "rb"))
sys_edge_info = pickle.load(open("sys_edge_info.dat", "rb"))
online_test_runs = pickle.load(open("online_test_runs.dat", "rb"))
env_win = pickle.load(open("env_win_set.dat","rb"))
sys_win = pickle.load(open("sys_win_set.dat","rb"))
W_V1 = pickle.load(open("W_V1.dat","rb"))
W_V2 = pickle.load(open("W_V2.dat","rb"))
W = pickle.load(open("W.dat","rb"))
W0 = pickle.load(open("W0.dat","rb"))
e_states = pickle.load(open("e_states.dat","rb"))
N = len(W)
N_env = len(env_win)
N_sys = len(sys_win)

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
    for env_state in range(1,Ne):
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
    for sys_state in range(1,Ns):
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

def get_state(state):
    if state%Ne == 0:
        env_state = Ne
        sys_state = state/Ne - 1
    else:
        env_state = state%Ne
        sys_state = state//Ne
    return env_state, sys_state
### Main components of the file:
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

# Now, do test strategy synthesis:
# Randomly choosing a successor vertex from a given set of vertices
def random_succ(succ_vts):
    r = []
    if succ_vts:
        l = len(succ_vts)
        r = succ_vts[randrange(l)]
    else:
        print("Error in random_succ: The input list is empty.")
    return r

# Returns the smallest (initial) and largest winning sets for a given player:
def player_win_sets(player):
    if(player == 'e'):
        Wf = env_win[0]
        W = env_win[-1]
    elif(player=='s'):
        Wf = sys_win[0]
        W = sys_win[-1]
    else:
        print("Player needs to be 'e' or 's'")
    return W, Wf

def synth_test_strategy(env_edge_info, sys_edge_info, e0, env_win, sys_win):
    sys_control = system_K()
    test_run = [e0]
    start = e0
    start = system_controller(start, sys_control) # Just to initialize the test run in the right place
    player = 'e'
    W, Wf = player_win_sets(player)
    while start not in Wf: # Final winning set
        if (player=='e'):
            edge_information = env_edge_info.copy()
            succ_edge_information = sys_edge_info.copy()
            succ_player = 's'
        
            edges, edges_idx = edge(start, edge_information, player)
            succ = [edge[1] for edge in edges] # List of all system successor vertices
            succ_win = [edge[4] for edge in edges] # List containing whether successor is in winning set or not
            succ_not_win = [succ[ii] for ii in range(len(succ_win)) if not succ_win[ii]] # All successor vertices not in winning set
            excluded_succ_not_win = env_assumption_violate(start, succ_not_win) # Set of successor vertices to which if the environment transitions to, it violates it's assumptions. 
            
            # List containing winning status of successors not in winning set:
            succ_in_win = [succ_win[ii] for ii in range(len(edges)) if succ[ii] not in excluded_succ_not_win]
            
            # First check if start is in the maximal winning set, and if it is not, check if it is because of transitions that might violate env assumptions
            vts_not_win = [] # List of possible successors that don't violate env assumptions, but put system outside win set
            if start not in W:
                # We want to exclude these set of successors from the ones that the environment could transition to.
                vts_not_win = [s for s in succ_not_win if s not in excluded_succ_not_win]
            
            # If there are environment transtions that don't violate env assumptions and put the system outside the winning set, then take that action:
            if vts_not_win:
                fin = finish(vts_not_win, succ, edges_idx, player)
                test_run.append(fin)
                break
            # Once you're outside the winning set, there's no way back in
            # ToDo: Make the following modular in functions...
            else:
                assert all(succ_in_win) # All successor vertices are winning vertices
                succ_1 = [succ[ii] for ii in range(len(succ)) if (edges[ii][4] == 1)]
                succ_2 = [succ[ii] for ii in range(len(succ)) if (edges[ii][4] == 2)]

                # Check that no vertices are in the excluded zone:
                succ_1_excl = [s for s in succ_1 if s in excluded_succ_not_win]
                succ_2_excl = [s for s in succ_2 if s in excluded_succ_not_win]
                assert(not succ_1_excl)
                assert(not succ_2_excl)

                succ_1_idx = [edges_idx[ii] for ii in range(len(succ)) if edges[ii][4] == 1]
                succ_2_idx = [edges_idx[ii] for ii in range(len(succ)) if edges[ii][4] == 2]
                
                # Unvisited nodes
                unvisit_succ_1 = [succ[ii] for ii in range(len(succ)) if (edges[ii][4] == 1 and edges[ii][3] == 0)]
                unvisit_succ_2 = [succ[ii] for ii in range(len(succ)) if (edges[ii][4] == 2 and edges[ii][3] == 0)]
                # Indices of unvisited nodes:
                unvisit_succ_1_idx = [edges_idx[ii] for ii in range(len(succ)) if (edges[ii][4] == 1 and edges[ii][3] == 0)]
                unvisit_succ_2_idx = [edges_idx[ii] for ii in range(len(succ)) if (edges[ii][4] == 2 and edges[ii][3] == 0)]

                # Successor nodes in same and better winning set with some winning actions not taken
                action_succ_1 = []
                action_succ_2 = []
                action_succ_1_idx = []
                action_succ_2_idx = []
                action_succ_1_potential_cones = [] # Potential system transitions that need to be blocked to enforce strong fairness
                action_succ_2_potential_cones = [] # Potential system transitions that need to be blocked to enforce strong fairness
                for ii in range(len(succ_1)):
                    s1 = succ_1[ii]
                    s1_idx = succ_1_idx[ii]
                    s1_edges, s1_edge_idx = edge(s1, succ_edge_information, succ_player)
                    # Have the winning actions been taken?
                    untested_winning_transitions1 = [t for t in s1_edges if (s1_edges[3]==0 and s1_edges[4])]
                    action_succ_1_potential_cones = [t[1] for t in untested_winning_transitions1]
                    if untested_winning_transitions1:
                        if not action_succ_1:
                            action_succ_1 = [s1]
                            action_succ_1_idx = s1_idx
                        else:
                            action_succ_1.append[s1]
                            action_succ_1_idx.append[s1_idx]
                # Successor nodes in attractor winning set with some winning actions not taken:
                for ii in range(len(succ_2)):
                    s2 = succ_2[ii]
                    s2_idx = succ_2_idx[ii]
                    s2_edges, s2_edge_idx = edge(s2, succ_edge_information, succ_player)
                    # Have the winning actions been taken?
                    untested_winning_transitions2 = [t for t in s2_edges if (s2_edges[3]==0 and s2_edges[4])]
                    action_succ_2_potential_cones = [t[1] for t in untested_winning_transitions2]
                    if untested_winning_transitions2:
                        if not action_succ_2:
                            action_succ_2 = [s2]
                            action_succ_2_idx = s2_idx
                        else:
                            action_succ_2.append[s2]
                            action_succ_2_idx.append[s2_idx]

                # First, go through all unvisited vertices in the same winning set, before proceeding to the attractor set at the next level
                if unvisit_succ_1:
                    fin = finish(unvisit_succ_1, succ, edges_idx, player)
                elif unvisit_succ_2:
                    fin = finish(unvisit_succ_2, succ, edges_idx, player)
                elif action_succ_1:
                    fin = finish(action_succ_1, succ, edges_idx, player)
                elif action_succ_2:
                    fin = finish(action_succ_2, succ, edges_idx, player)
                else:
                    # Choose from any of the successors that do not violate environment assumptions:
                    succ_not_violate = [s for s in succ if s not in excluded_succ_not_win]
                    if succ_not_violate:
                        fin = finish(succ_not_violate, succ, edges_idx, player)
                        # place_cone = 1 # For now, don't place cone here
                    else:
                        print("Environment out of options: no transition comp. with this initial condition")
                        print("Test ends because environment cannot take an action anymore")
                        break
            # Then keep a loop until you reach the winning set
                test_run.append(fin)

            # Next iteration resetting:
            player = 's'
            W, Wf = player_win_sets(player)
            start = fin

        # Here, we use the correct-by-construction controller to respond to the environment changes
        elif (player == 's'):
            edge_information = sys_edge_info.copy()
            succ_edge_information = env_edge_info.copy()
            succ_player = 'e'

            # Correct-by-construction controller for system:
            fin = system_controller(start, sys_control)
            test_run.append(fin)

            # Update the number of times the edge has been visited:
            start_state = set_state(start, player)
            fin_state = set_state(fin, succ_player)
            start_edge, start_succ_idx = edge(start, sys_edge_info, player)
            start_succ= [s[1] for s in start_edge]
            fin_state_idx = start_succ_idx[start_succ.index(fin_state)] # Index of the successor state
            sys_edge_info[fin_state_idx][3]+=1 

            # Next iteration resetting:
            player = 'e'
            W, Wf = player_win_sets(player)
            start = fin
        else:
            print("Player must be 'e' or 's'")
        print(fin)
    return test_run                                             

# Returns the set of successor states to which transitioning to by the environment requires violation of its own specifications
# I might have to use TuLiP controllers for this. 
def env_assumption_violate(start, succ_not_win):
    U = unsafe(collision_indices) # Vertices that the environment should not transition to because it is a safety assumption violation
    unsafe_states = [get_state(int(u[3:])) for u in U] # Collecting unsafe states in a list form: [[env1, sys1], [env2, sys2],...]
    succ_not_win_states = [get_state(int(s[3:])) for s in succ_not_win] # Collecting coordinates of not winning successor states: [[env1, sys1], [env2, sys2], ...]
    violate_succ = [set_state(s,'s') for s in succ_not_win_states if s in unsafe_states]
    return violate_succ


# A function to take in the correct-by-construction controller for the system:
def system_controller(start, sys_control):
    env_pos = start[0]
    sys_pos = start[1]
    u = system_K.move(sys_control, env_pos)
    sys_pos = u[sys_var]
    finish = [env_pos, sys_pos]
    return finish

# Function to find the end_vertex randomly from a list of vertices vts, succ is the list of all successors, edge_idx is the index of all edges in the main player_edge_info list
# Function also takes care of updating the main env_Edge_info or sys_edge_info list
def finish(vts, succ, edges_idx, player):
    rand_fin = random_succ(vts) # Choosing final state
    rand_fin_idx = edges_idx[succ.index(rand_fin)]# Finding final state winning index in either sys_edge_info/env_edge_info

    # Increasing number of times successor has been visited
    if player == 'e':
        env_edge_info[rand_fin_idx][3] += 1
    elif player == 's':
        sys_edge_info[rand_fin_idx][3] += 1
    rand_fin_env, rand_fin_sys = get_state(int(rand_fin[3:]))
    fin = [rand_fin_env, rand_fin_sys]
    return fin


# Test suite generation:
def test_suite(Ntests, e0, env_win, sys_win):
    test_suite = []
    for ii in range(0, Ntests):
        print("Test #", str(len(test_suite) + 1))
        test = synth_test_strategy(env_edge_info, sys_edge_info, e0, env_win, sys_win)
        if test_suite:
            test_suite.append(test)
        else:
            test_suite = [test]
    return test_suite

# Plotting function
def plot_grid_world(env_traj, sys_traj):
    # Plotting tools
    alw = 0.75    # AxesLineWidth
    fsz = 12      # Fontsize
    lw = 2        # LineWidth
    msz = 6       # MarkerSize
    # Gridworld size: M-by-N gridworld
    M = 4
    N = 4
    # Plotting system trajectory
    Cr = env_traj.copy()    
    Xr = sys_traj.copy()
    
    fig, ax = base_plot()
    points_sys, = ax.plot([], [], marker='o', color='blue', markersize=2*msz, markerfacecolor='blue')
    points_env, = ax.plot([], [], marker='o', color='red',  markersize=2*msz, markerfacecolor='red')
    len_TRAJ = len(Xr)    
        
    jmax = 10
    
    MAX_FRAMES = jmax*len_TRAJ
    
    def animate(frame_idx):
        ii = frame_idx//jmax
        jj = frame_idx%jmax
        
        robot_x, robot_y, patrol_x, patrol_y = grid_position(Xr[ii], Cr[ii])
        # In the first iteration, the old_robot_pos is the same as
        # curr_robot_pos
        if ii == 0: 
            old_robot_x = robot_x
            old_robot_y = robot_y
            old_patrol_x = patrol_x
            old_patrol_y = patrol_y
        else:
            old_robot_x, old_robot_y, old_patrol_x, old_patrol_y = grid_position(Xr[ii-1], Cr[ii-1])
            
        int_robot_x = np.linspace(old_robot_x, robot_x, jmax)
        int_robot_y = np.linspace(old_robot_y, robot_y, jmax)
        int_patrol_x = np.linspace(old_patrol_x, patrol_x, jmax)
        int_patrol_y = np.linspace(old_patrol_y, patrol_y, jmax)
        
        points_sys.set_data(int_robot_x[jj],int_robot_y[jj])
        points_env.set_data(int_patrol_x[jj],int_patrol_y[jj])
        return [points_sys, points_env]
    
    # Takes in raw grid cell location numbers for system and environment and returns location coordinates for system and environment
    def grid_position(sys_raw, env_raw):
        sys_x = sys_raw%N
        if sys_x == 0:
            sys_x = N
        
        env_x = env_raw%N
        if env_x == 0:
            env_x = N
        
        sys_y = M - math.floor((sys_raw-1)/M)
        env_y = M - math.floor((env_raw-1)/M)
        return sys_x, sys_y, env_x, env_y
    
    ani = animation.FuncAnimation(fig, animate, frames=MAX_FRAMES, interval = 100, blit=True)
    return ani


def base_plot():
    msz = 6       # MarkerSize
    goal_cell_x = 1
    goal_cell_y = 4
    # home_cell_x = 4
    # home_cell_y = 1
    # refuel_cell_x = 4
    # refuel_cell_y = 3

    grid_loc_x = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    grid_loc_y = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    text_loc = {'c13', 'c9', 'c5', 'c1', 'c14', 'c10', 'c6', 'c2', 'c15', 'c11', 'c7', 'c3', 'c16', 'c12', 'c8', 'c4'}
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(grid_loc_x, grid_loc_y, 'o', markersize=msz, markerfacecolor='black')
    # set(gcf,'Visible', 'off') # Keep it from popping up
    # plt.hold()
    plt.plot(goal_cell_x, goal_cell_y, 'o', markersize=msz, markerfacecolor='blue')
    plt.text(goal_cell_x + 0.1, goal_cell_y + 0.2, "Goal")
    # plt.plot(home_cell_x, home_cell_y, 'o', markersize=msz, markerfacecolor='blue')
    # plt.text(home_cell_x + 0.1, home_cell_y + 0.2, "Home")
    # plt.plot(refuel_cell_x, refuel_cell_y, 'o', markersize=msz, markerfacecolor='blue')
    # plt.text(refuel_cell_x + 0.1, refuel_cell_y + 0.2, "Refuel")
    ax.set_xlim(0,5)
    ax.set_ylim(0,5)
    plt.xticks(np.arange(5))
    plt.yticks(np.arange(5))
    
    return fig, ax

# Synthesize test suite:
# Fixed initial condition:
e0 = [2,15]
Ntests = 3
TS = test_suite(Ntests, e0, env_win, sys_win)
test_file_names = ["grid_env2_sys15_test1.avi", "grid_env2_sys15_test2.avi", "grid_env2_sys15_test3.avi"]
for ii in range(0,Ntests):
    test = TS[ii]
    fn = test_file_names[ii]
    env_traj = [v[0] for v in test]
    sys_traj = [v[1] for v in test]
    anim = plot_grid_world(env_traj, sys_traj)
    anim.save(fn)