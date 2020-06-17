# ======================= Overview ======================= #
# % ------------ Class GameGraph ------------ % #
# This class makes an abstraction of the gridworld class
# A node in the graph is: n1 = ([l_p1, l_p2, ...], pa), where l_pi is the location of player i in the grid and pa is the agent that is active in the current node
# + state: Returns the state of the game as a function of individual locations of players in the game graph
# + state2node: Returns the locations of the environment and system based on the state input into the graph
# + edges: A function that returns the edges of the game graph. it sets edges from transitions of the agents on the grid. If the transitions of an agent are independent of other agents on the grid, 
#  then the bridge variable is empty. If bridge is non-empty, and if the agent transitions to one of the bridge locations, it must stay there until the system is closer to all (or any) of the other bridges than the environment is
# + shortest_distance(bridge, start): Returns a list of shortest distances from the start point to every element in bridge.
# + setup_dict(): Sets up a dictionary to convert environment locations on the grid to a composed state in the range (1,Ne). Similarly, it composes all system locations on the grid into a composed state in the range (0,Ns).
# + Also returns a list pEdges of length Ne (Ns) where each element of the list points to other possible transitions. So pEdges[3] returns the possible transitions of the collective environment from state 3.

import numpy as np
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

import gridworld_class 
import Player_class

class GameGraph:
    def __init__(self, Nrows, Ncols, static_obstacles, cell2node, node2cell, players):
        self.G = nx.DiGraph()
        self.M = Nrows
        self.N = Ncols
        self.static_obs = static_obstacles
        self.static_grid = nx.Graph() # Graph based on just the grid and not the 2-player game graph with any of the players involved
        self.env_pEdges = None
        self.sys_pEdges = None
        self.env_dict = None
        self.sys_dict = None
        self.Ns = None
        self.Ne = None
        self.V = None
        self.Vweight = None # Dictionary mapping vertices to their weights
        self.E = None
        self.U = None
        self.C2N = cell2node.copy()
        self.N2C = node2cell.copy()
        self.vertices(players)     
        self.edges() # This sets the edges based on the data collected from vertices.
        self.G.add_edges_from(self.E)
        self.G.add_nodes_from(self.V)
        self.unsafe_states()
        self.set_vertex_weight()
        self.Val_sys = self.init_val_function('s') # Value function corresponding to system objective
        self.Val_env = self.init_val_function('e') # Value function corresponding to environment objective
    def set_vertex_weight(self, *args):
        if args:
            self.Vweight = None
        else:
            Vweight_list = [[v, 0] for v in self.V]
            self.Vweight = dict(Vweight_list)
    def init_val_function(self, ptype):
        Val_list = []
        if ptype == 's':
            base = 'inf'
        elif ptype == 'e':
            base = '0'
        else:
            print("Error: input argument pytpe must be 's' or 'e'")
        for vi in self.V:
            Val_list.append([vi, float(base)])
        Val = dict(Val_list)
        return Val
    def get_value_function(self, ptype):
        if ptype == 's':
            return self.Val_sys
        elif ptype == 'e':
            return self.Val_env
        else:
            print("Err: input argument must be 's' or 'e'")
    def setup_static_grid(self):
        for row in range(1, self.M+1):
            for col in range(1, self.N+1):
                node = self.C2N[(row, col)]
                node_nbrs = []
                if node not in self.static_obs:
                    if (1<row<self.M):
                        node_nbrs.extend([node+self.N, node-self.N])
                    if (1<col<self.N):
                        node_nbrs.extend([node-1, node+1])
                    if(row==1):
                        node_nbrs.extend([node+self.N])
                    if(row==self.M):
                        node_nbrs.extend([node-self.N])
                    if(col==1):
                        node_nbrs.extend([node+1])
                    if(col==self.N):
                        node_nbrs.extend([node-1])
                    node_succ = [n for n in node_nbrs if n not in self.static_obs] # Only successors that are not static obstacles are included
                else:
                    node_succ = node_nbrs
                for succ in node_succ:
                    self.static_grid.add_edge(node, succ)
    def state(self, Ns, Ne, ns, ne):
        return Ne*(ns - 1) + ne
    def get_vertex(self, v):
        cellv = []
        if(v[0:2]=="v1" or v[0:2]=="v2"):
            state = int(v[3:])
            env_st, sys_st = self.state2node(self.Ns, self.Ne, state)
            envdict = self.env_dict[1]
            sysdict = self.sys_dict[1]
            env_node = envdict[env_st]
            sys_node = sysdict[sys_st]
            env_cell = self.N2C[env_node]
            sys_cell = self.N2C[sys_node]
            cellv = [env_cell, sys_cell]
        else:
            print("Error: enter vertex starting with 'v1' or 'v2'")
        return cellv
    def state2node(self, Ns, Ne, st):
        if st%Ne == 0:
            env_state = Ne
            sys_state = st/Ne 
        else:
            env_state = st%Ne
            sys_state = st//Ne + 1
        return env_state, sys_state

    def shortest_distance(self, bridge, start):
        dist_to_bridge = []
        for b in bridge:
            path_b = nx.dijkstra_path(self.static_grid, start, b)
            dist_to_bridge.append(len(path_b))
        return dist_to_bridge

    # p2n: dictionary with key 0 to Nmax and converting to a position on the grid
    # n2p: dictionary with key that is position on the grid converting to a number from 0 to Nmax
    # p: player whose dictionary we are interested in
    def setup_dict(self, players, ptype, Nmax):
        p2n_list = []
        n2p_list = []
        for pi in players:
            player = players[pi]
            if (player.get_type() == ptype):
                p = player
        t = p.get_transitions()
        transitions_p = [x for x in t if x!=[]] # If p.get_transitions() = [[], [], [2], [3]], then this line returns transitions_p = [[2],[3]]
        # Assertion check to make sure that each element of transitions_p is a list containing elements numbered from 1 to M*N
        for l in transitions_p:
            li = [(1<=ii<=self.M*self.N) for ii in l]
            assert(all(li))
        transitions_p_index = [ii+1 for ii in range(len(t)) if t[ii]!=[]] # If p.get_transitions() = [[], [], [4], [5]], then this line returns transitions_p_index = [3,4], where 3 and 4 are static_grid locations that the agent is active in.
        assert(len(transitions_p_index) == Nmax)
        for j in range(Nmax):
            n2p_list.append((transitions_p_index[j], j+1))
            p2n_list.append((j+1, transitions_p_index[j]))
        
        # Sanity Check:
        assert(len(p2n_list) == Nmax)
        assert(len(n2p_list) == Nmax)
        p2n = dict(p2n_list)
        n2p = dict(n2p_list)
        pEdges = self.setup_player_edges(Nmax, transitions_p, transitions_p_index, n2p)
        return p2n, n2p, pEdges
    
    # This returns transitions of the player in the form of a list that is of length Nmax: [[1], [2,3], ..., [Nmax]]
    def setup_player_edges(self, Nmax, transitions_p, transitions_p_index, n2p):
        pEdges = [[] for j in range(Nmax)]
        for jidx in range(len(transitions_p_index)):
            tp = transitions_p[jidx]
            tp2n = [n2p[ii] for ii in tp]
            pEdges[jidx].extend(tp2n)
        return pEdges

    def edges(self):
        self.setup_static_grid()
        env_edges = self.env_pEdges
        sys_edges = self.sys_pEdges
        Edges = []
        for ns in range(1, self.Ns+1):
            for ne in range(1, self.Ne+1):
                start_state = self.state(self.Ns, self.Ne, ns, ne)
                env_transitions = env_edges[ne-1]
                sys_transitions = sys_edges[ns-1]
                end_states_env = [self.state(self.Ns, self.Ne, ns, ne_end) for ne_end in env_transitions]
                end_states_sys = [self.state(self.Ns, self.Ne, ns_end, ne) for ns_end in sys_transitions]

                vstart_sys = "v2_"+str(start_state)
                vstart_env = "v1_"+str(start_state)
                
                # Make assertions to check if vertices are in V:
                assert(vstart_env in self.V)
                assert(vstart_sys in self.V)

                # Environment to system transitions:
                for end_state in end_states_env:
                    vend_sys = "v2_"+str(end_state)
                    Edges.append((vstart_env, vend_sys))
                
                # System to environment transitions:
                for end_state in end_states_sys:
                    vend_env = "v1_"+str(end_state)
                    Edges.append((vstart_sys, vend_env))
        self.E = Edges.copy() # Setting the edges variable

    def vertices(self, players):
        V= []
        Ns = 1
        Ne = 1
        # No. of states each enviornment / system agent holds
        n_env_states = [] 
        n_sys_states = []
        for pi in players:
            p = players[pi]
            n_states_p = p.get_nstates() # Number of states this player can be in
            if(p.get_type() == 's'):
                Ns = Ns*n_states_p
                n_sys_states.append(n_states_p)
            else:
                Ne = Ne*n_states_p
                n_env_states.append(n_states_p)
        env_p2n_dict, env_n2p_dict, env_pEdges = self.setup_dict(players, 'e', Ne) # Environment states on grid to a number in [1, Ne]
        sys_p2n_dict, sys_n2p_dict, sys_pEdges = self.setup_dict(players, 's', Ns) # System states on grid to a number in [1, Ns]
        for ns in range(1, Ns+1):
            for ne in range(1, Ne+1):
                s = self.state(Ns, Ne, ns, ne)
                V.extend(["v1_"+str(s)]) # Environment action vertices
                V.extend(["v2_"+str(s)]) # System action vertices
        
        # Setting gridworld game graph variables:
        self.env_pEdges = env_pEdges
        self.sys_pEdges = sys_pEdges
        self.env_dict = [env_n2p_dict, env_p2n_dict]
        self.sys_dict = [sys_n2p_dict, sys_p2n_dict]
        self.Ns = Ns
        self.Ne = Ne
        self.V = V

    def get_edges_vertices(self):
        return self.E, self.V
# This function computes the set of states that are unsafe, i.e when system collides with static obstacles or with the moving environment
    def unsafe_states(self):
        unsafe_states = []
        env_n2p_dict = self.env_dict[0] 
        env_p2n_dict = self.env_dict[1]
        sys_n2p_dict = self.sys_dict[0] 
        sys_p2n_dict = self.sys_dict[1]
        # Unsafe states from system being in the same state as a static obstacle:
        for nstat in self.static_obs:
            for ne in range(1, self.Ne+1):
                ns = sys_n2p_dict[nstat]
                s = self.state(self.Ns, self.Ne, ns, ne)
                unsafe_states.extend(["v1_"+str(s)]) # Environment action vertices
                unsafe_states.extend(["v2_"+str(s)]) # System action vertices
        
        # Unsafe states from system and environment being in the same state:
        for ne in range(1, self.Ne+1):
            env_node = env_p2n_dict[ne]
            assert(1<=env_node<=self.M*self.N)
            ns = sys_n2p_dict[env_node]
            s = self.state(self.Ns, self.Ne, ns, ne)
            unsafe_states.extend(["v1_"+str(s)]) # Environment action vertices
            unsafe_states.extend(["v2_"+str(s)]) # System action vertices
        # Unsafe
        self.U = unsafe_states.copy()
    def get_unsafe_states(self):
        return self.U
# This function is the predecessor operator that computes the set of states from which for all
#   + pre: Predecessor computation on a game graph
#   + W0: Winning set that the transition must end up in
#   + ptype: 's' or 'e' for system or environment, i.e whether the pre set should be comprised of system or environment states
#   + quantifier: 'exists' or 'forall'
    def pre(self, W0, ptype, quantifier):
        pre_W0 = []
        for n0 in W0:
            pred_n0 = list(self.G.predecessors(n0))
            if(quantifier == 'exists'):
                pre_W0.extend(pred_n0)
            if(quantifier == 'forall'):
                for pred_n0_ii in pred_n0:
                    pred_n0_ii_succ = list(self.G.successors(pred_n0_ii))
                    pred_n0_ii_succ_W0 = [p in W0 for p in pred_n0_ii_succ]
                    if all(pred_n0_ii_succ_W0):
                        if pred_n0_ii not in pre_W0:
                            pre_W0.append(pred_n0_ii)
        pre_W0 = list(dict.fromkeys(pre_W0)) # Removes duplicates
        return pre_W0

# This function needs to be called only after the initial value function has been set for each of the respective agents
# Sets the value function for the system and the environment depending on their quantifiers. For the system, the quantifier does not change the value function. The environment might have different ways of setting the value function depending on the quantifier.
# TODO: Change how the environemnt value function is set.
# TODO: If the winning agent is the environment, the value function for the system states do not change no matter if the quantifier is forall or exists
    def determine_value(self, V, W0, quant, ptype, N, win_agent):
        Val = V.copy()
        if win_agent == 's':
            if(ptype  == 's'):
                for v in W0:
                    Val[v] = N
            if(ptype == 'e'):
                if (quant == 'forall'):
                    for v in W0:
                        Val[v] = N-1
                if (quant == 'exists'):
                    for v in W0:
                        Val[v] = N-1
        elif win_agent == 'e':
            if ptype == 's':
                if (quant == 'forall'):
                    for v in W0:
                        successors = list(self.G.successors(v))
                        Val_successors = [Val[s] for s in successors]
                        min_successor = successors[Val_successors.index(min(Val_successors))]
                        Val[v] = Val[min_successor]
                if (quant == 'exists'):
                    for v in W0:
                        successors = list(self.G.successors(v))
                        Val_successors = [Val[s] for s in successors]
                        min_successor = successors[Val_successors.index(min(Val_successors))]
                        Val[v] = Val[min_successor]
                    
            if ptype == 'e':
                for v in W0:
                    successors = list(self.G.successors(v))
                    if successors != []:
                        Val_successors = [Val[s] for s in successors]
                        max_successor = successors[Val_successors.index(max(Val_successors))]
                        Val[v] = self.Vweight[v] + Val[max_successor]
        return Val

# Function to return set of winning states as forms of vertices in V by taking in a list of lists: [[env_cell], [sys_cell]] of environment and system locations where env and sys are cell locations of the environment and system on the gridworld respectively
    def set_win_states(self, goal_list):
        goal_env = []
        goal_sys = []
        envdict = self.env_dict[0].copy()
        sysdict = self.sys_dict[0].copy()
        for g in goal_list:
            genv = g[0]
            gsys = g[1]
            assert(len(gsys)==2)
            assert(len(genv)==2)
            env_node = self.C2N[(genv[0], genv[1])]
            sys_node = self.C2N[(gsys[0], gsys[1])]
            ne = envdict[env_node]
            ns = sysdict[sys_node]
            st = self.state(self.Ns, self.Ne, ns, ne)
            env_st = "v1_"+str(st)
            sys_st = "v2_"+str(st)
            goal_env.append(env_st)
            goal_sys.append(sys_st)
        Wgoal = [goal_env, goal_sys]
        return Wgoal
# The following function is to synthesize the reachability winning set:
# + win_agent: the player ('s' or 'e') for which the winning set is being determined
# + goal: The set of states which is the goal for the winning agent. Goal is a list: [[goal_env], [goal_sys]] where goal_sys is the goal with system action states and goal_env is goal with environment action states
# + quant1: 'exists' or 'forall'; this is the quantifier for the other agent that is not the win_agent (env)
# + quant2: 'exists' or 'forall' this is the quantifier for the win_agent (sys)
# Outputs:
# + W: This is a winning set with two arguments: [W_env, W_sys]
# + Val: Value function for each element of the winning set. Converted to a dictionary from a list
    def win_reach(self, win_agent, goal, quant1, quant2):
        W = [[goal[0], goal[1]]]
        lW = len(W)
        W0 = W[lW-1]
        W0_sys = W0[1].copy()
        W0_env = W0[0].copy()
        if (win_agent == 's'):
            other_agent = 'e'
            quant_e = quant1
            quant_s = quant2
            Val = self.Val_sys.copy()
            for v in W0_sys:
                Val[v] = 0
            for v in W0_env:
                Val[v] = 0
        else:
            other_agent = 's'
            quant_s = quant1
            quant_e = quant2
            Val = self.Val_env.copy()
            # assert(W0_sys == []) # We only want environment states in the original winning set of the system
            for v in W0_env:
                Val[v] = self.Vweight[v]
        # fixpoint_env checks if W0_env which is the winning set with environment states in it is a fixpoint
        # fixpoint_sys checks if W0_sys which is the winning set with system states in it is a fixpoint
        fixpoint_env = False
        fixpoint_sys = False
        N = 0

        while (not fixpoint_env or not fixpoint_sys):
            N += 1
            lW = len(W)
            W0 = W[lW - 1]
            # Predecessor set to system states:
            W0_sys = W0[1].copy()
            W0_env = W0[0].copy()
            pre_sys = self.pre(W0_sys, 'e', quant_e)
            pre_env = self.pre(W0_env, 's', quant_s)
            Wnew_sys = W0_sys.copy()
            Wnew_env = W0_env.copy()
            if pre_env:
                Wnew_sys.extend(pre_env) # pre_env contains system states and is a predecessor to a environment winning set
                Wnew_sys = list(dict.fromkeys(Wnew_sys)) # Removes duplicates
            if pre_sys:
                Wnew_env.extend(pre_sys) # pre_sys contains environment states and is a predecessor to a system winning set
                Wnew_env = list(dict.fromkeys(Wnew_env)) # Removes duplicates
            Val = self.determine_value(Val, pre_sys, 'e', quant_e, N, win_agent)
            Val = self.determine_value(Val, pre_env, 's', quant_s, N, win_agent)
            Wcur = [Wnew_env, Wnew_sys]
            if Wnew_env == W0_env:
                fixpoint_env = True
            if Wnew_sys == W0_sys:
                fixpoint_sys = True
            W.append(Wcur)
        if(win_agent == 's'):
            self.Val_sys = Val.copy()
        if(win_agent == 'e'):
            self.Val_env = Val.copy()
        return W
    def win_cells(self, W):
        lW = len(W)
        Wenv = [[] for ii in range(lW)]
        Wsys = [[] for ii in range(lW)]
        Wenv_cells = [[] for ii in range(lW)]
        Wsys_cells = [[] for ii in range(lW)]
        for ii in range(lW):
            Wenv[ii] = W[ii][0]
            Wsys[ii] = W[ii][1]
            Wenv_cells[ii] = [self.get_vertex(v) for v in Wenv[ii]]
            Wsys_cells[ii] = [self.get_vertex(v) for v in Wsys[ii]]
        return Wenv_cells, Wsys_cells