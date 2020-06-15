# % ================ Formal Methods for T&E and V&V of autonomous systems =============== #
# % ================ Apurva Badithela // 6/2/2020 =============== #
# % ================ File defining grid world environment ============= #
# This file contains code for 2 players on the grid...
# % Gridworld class:
# %     + init: Takes in size of the grid as list [M,N] where M is the number of rows and N is the number of columns
import numpy as np
from random import randrange
import importlib
import itertools
import pickle
import matplotlib
import os
import math
import networkx as nx
import pdb
# from grid_construct import grid

# % ------- Class Gridworld ---------- % #
# Constructing a gridworld with 2 players: system and environment agents and other objects such as static obstacles, static cones.
# + Attributes: 
#   - Nrows: No. of rows in grid
#   - Ncols: No. of columns in grid
#   - Nsys: No. of system agents
#   - Nenv: No. of environment agents
#   - cones: List of all cones on the grid
#   - static_obstacles: List of static obstacles in the grid
#
# + add_player:
#   - Arguments: player_name is the name of the agent, player_type is the 's' or 'e' for system or environment, player_transitions_type is the type of transitions an agent is restricted to take. If player_transitions_type = ['all'], then the agent can occupy any part of the grid
# If player_transitions_type == ['specific', transitions], where transitions is a list of length M*N of the form transitions[ii] = {jj| where jj is the node location on the grid that the agent can transition to}.
# + add_static_obstacles:
#   - Arguments: player_name is the name of the agent, player_type is the 's' or 'e' for system or environment
# + set_states: Cretaes a dictionary of coordinates to numbers representing state ids. Numbering starts from the top left corner of the grid. So cell [1,1] has node_id 1, cell [1,2] has node_id 2, so on ...
# + construct_graph():
# + get_game_graph():
# + plot_win_set():
class GridWorld:
    def __init__(self, size):
        self.Nrows = size[0]
        self.Ncols = size[1]
        self.Nsys = 0 
        self.Nenv = 0
        self.cones = []
        self.static_obstacles = []
        self.player_names = []
        self.players = dict()
        self.node2cell = None
        self.cell2node = None
        self.set_states()
        self.G = []
        self.gridT = None
    
    def get_player(self, player_name):
        return self.players[player_name]
    def add_player(self, player_name, player_transitions_type, player_type):
        self.player_names.append(player_name)
        self.players[player_name] = Player(player_name, self, player_transitions_type, player_type)

        if(player_type == 's'):
            self.Nsys += 1
        else:
            self.Nenv += 1
    
    def grid_transitions(self):
        cell2node = self.cell2node.copy()
        T = [[] for ii in range(self.Nrows * self.Ncols)]
        for ii in range(1, self.Nrows+1):
            for jj in range(1, self.Ncols+1):
                cell = [ii, jj]
                cell_trans = [[ii, jj]]
                if cell2node[(cell[0], cell[1])] not in self.static_obstacles:
                    if (ii == 1):
                        cell_trans.append([ii+1, jj])
                    if (1 < ii and ii < self.Nrows):
                        cell_trans.append([ii+1, jj])
                        cell_trans.append([ii-1, jj])
                    if (ii == self.Nrows):
                        cell_trans.append([ii-1, jj])
                    if (jj == 1):
                        cell_trans.append([ii, jj+1])
                    if (1 < jj and jj < self.Ncols):
                        cell_trans.append([ii, jj+1])
                        cell_trans.append([ii, jj-1])
                    if (jj == self.Ncols):
                        cell_trans.append([ii, jj-1])
                    transitions = [cell2node[(c[0], c[1])] for c in cell_trans if cell2node[(c[0], c[1])] not in self.static_obstacles]
                    # We need to index by 1.
                    T[cell2node[(cell[0], cell[1])] - 1] = transitions.copy()
                else:
                    T[cell2node[(cell[0], cell[1])] - 1] = [cell2node[(cell[0], cell[1])]]
        self.gridT = T.copy()
    
    def add_static_obstacles(self, s):
        snodes = [self.cell2node[(si[0], si[1])] for si in s]
        self.static_obstacles.extend(snodes)

    def add_cones(self, c):
        self.cones.extend(c)
    
    def set_states(self):
        node2cell_list = []
        cell2node_list = []
        for ii in range(1, self.Nrows+1):
            for jj in range(1, self.Ncols+1):
                cell = self.Nrows*(ii-1) + jj
                node2cell_list.append((cell, (ii, jj)))
                cell2node_list.append(((ii, jj), cell))
        node2cell = dict(node2cell_list)
        cell2node = dict(cell2node_list)
        self.node2cell = node2cell.copy()
        self.cell2node = cell2node.copy()
    
    def construct_graph(self):
        self.G = GameGraph(self.Nrows, self.Ncols, self.static_obstacles, self.cell2node, self.node2cell, self.players)
    def get_game_graph(self):
        return self.G
    def base_plot(self):
        pass
    def plot_win_set(self, W):
        pass
# % ------------------------------ Class Player ----------------------------% #
# This class defines a player, its transitions of the gridworld and its policies
# + succ: Function that returns the set of successors to a player's node
# + pre: Function that returns set of predecessors to a player's node. Input node is in the form of a node id
# + get_transitions: Returns players physical transition on the grid
class Player:
    def __init__(self, name, grid, player_transitions_type, ptype):
        self.name = name
        self.type = ptype
        self.grid = grid
        self.transitions = None
        self.set_transitions(player_transitions_type)

    def set_transitions(self, player_transitions_type):
        gridT = self.grid.gridT
        C2N = self.grid.cell2node
        if (player_transitions_type[0]=='all'):
            self.transitions = gridT
        if(player_transitions_type[0]=='specific'):
            col = player_transitions_type[1]
            if(len(player_transitions_type) == 2):
                bump_idx = []
            else:
                bump_idx = player_transitions_type[2]
            
            T = [[] for ii in range(self.grid.Nrows * self.grid.Ncols)]
            for row in range(1, self.grid.Nrows+1):
                if(row==1):
                    cell = [row, col]
                    cell_trans = [[row+1, col]]
                elif(row == self.grid.Nrows):
                    cell = [row, col]
                    cell_trans = [[row-1, col]]
                else:
                    cell = [row, col]
                    cell_trans = [[row+1, col], [row-1, col]]
                
                T[C2N[(cell[0], cell[1])] - 1] = [C2N[(c[0], c[1])] for c in cell_trans]
                
                if(bump_idx):
                    # Clearing the transition in the cell of the bumped row and adding transitions to the one adjacent to it
                    if(row == bump_idx):
                        cell = [row, col]
                        T[C2N[(cell[0], cell[1])] - 1] = []
                        
                        cell = [row, col+1]
                        cell_trans = [[row+1, col+1], [row-1, col+1]]
                        T[C2N[(cell[0], cell[1])] - 1] = [C2N[(c[0], c[1])] for c in cell_trans]
                    if (row == (bump_idx - 1)):
                        cell1 = [row, col]
                        cell1_trans = [[row-1, col], [row, col+1]]
                        cell2 = [row, col+1]
                        cell2_trans = [[row, col], [row+1, col+1]]
                        T[C2N[(cell1[0], cell1[1])] - 1] = [C2N[(c[0], c[1])] for c in cell1_trans]
                        T[C2N[(cell2[0], cell2[1])] - 1] = [C2N[(c[0], c[1])] for c in cell2_trans]
                    if(row == bump_idx + 1):
                        cell1 = [row, col]
                        cell1_trans = [[row+1, col], [row, col+1]]
                        cell2 = [row, col+1]
                        cell2_trans = [[row, col], [row-1, col+1]]
                        T[C2N[(cell1[0], cell1[1])] - 1] = [C2N[(c[0], c[1])] for c in cell1_trans]
                        T[C2N[(cell2[0], cell2[1])] - 1] = [C2N[(c[0], c[1])] for c in cell2_trans]     
            self.transitions = T.copy()
        
    def get_transitions(self):
        return self.transitions

    def get_type(self):
        return self.type

    def get_nstates(self):
        N = 0
        for t in self.transitions:
            if t:
                N = N+1
        return N
    def succ(self, node):
        S = []
        return S

    def pre(self, node):
        P = []
        return P
    

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
                pre_W0 = pred_n0.copy()
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
    def determine_value(self, V, W0, quant, pytpe, N, win_agent):
        Val = V.copy()
        if win_agent == 's':
            if(pytpe  == 's'):
                for v in W0:
                    Val[v] = N
            if(pytpe == 'e'):
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
            assert(W0_sys == []) # We only want environment states in the original winning set of the system
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
                Wnew_sys.extend(pre_env) # pre_sys contains system states and is a predecessor to a environment winning set
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
# % ======= Main function to test the classes ========== % #
def main():
    ## Setting up gridworld and static obstacles:
    M = 10
    N = 10
    GW = GridWorld([M,N])
    
    # Setting STATIC obstacles:
    obs_col = 7
    static_obs = [[ii, obs_col] for ii in range(1,N)]
    GW.add_static_obstacles(static_obs)

    # Construct gridworld states and transitions:
    GW.grid_transitions()

    # Adding players and transitions:
    tester_col = obs_col+1
    tester_bump_row = math.floor(N/2)
    agent_transitions = ['all']
    tester_transitions = ['specific', tester_col, tester_bump_row]
    GW.add_player('agent', agent_transitions, 's')
    GW.add_player('tester', tester_transitions, 'e')

    # Checking player transitions have been correctly added:
    tester = GW.get_player('tester')
    tester_transitions = tester.get_transitions()

    # Making game graph and getting vertices and edges:
    GW.construct_graph()
    GAME = GW.get_game_graph()
    E, V = GAME.get_edges_vertices()
    
    # Printing out edges and vertices:
    # print(E)

    # Finding the system winning set and value function:
    bump_row=5
    sys_goal_cell = [M, N]
    env_goal_cell = [[ii, tester_col] for ii in range(1,M+1) if ii!=bump_row]
    extra_cells = [[bump_row-1, tester_col+1], [bump_row, tester_col+1], [bump_row+1, tester_col+1]]
    env_goal_cell.extend(extra_cells)
    goal_cells = [[env_cell, sys_goal_cell] for env_cell in env_goal_cell]
    goal_vertices = GAME.set_win_states(goal_cells)
    assert(all([elem in V for elem in goal_vertices[0]]))
    assert(all([elem in V for elem in goal_vertices[1]]))

    # Robust Pre for system winning set computation:
    quant_env = 'forall'
    quant_sys = 'exists'
    win_agent = 's'
    W_sys = GAME.win_reach(win_agent, goal_vertices, quant_env, quant_sys)
    Val_sys = GAME.get_value_function(win_agent)
    pdb.set_trace()
    # Finding the environment winning set and value function:
    # !!!! The environment and the system places need to be switched
    # IMPORTANT_NOTE: When we find the environment winning set, we need to have environment and system cells switched
    sys_goal_cell = [[1, obs_col], [M, obs_col]]
    env_goal_cell = [[ii, tester_col] for ii in range(1,M+1) if ii!=bump_row]
    env_goal_cell.extend(extra_cells)
    goal_cells = [[env_cell, sys_cell] for env_cell in env_goal_cell for sys_cell in sys_goal_cell]
    goal_vertices = GAME.set_win_states(goal_cells)

    # Robust Pre for system winning set computation:
    quant_env = 'exists'
    quant_sys = 'exists'
    win_agent = 'e'
    W_env = GAME.win_reach(win_agent, goal_vertices, quant_env, quant_sys)
    Val_env = GAME.get_value_function(win_agent)

    #   Printing the value functions:
    pdb.set_trace()
    print(Val_sys)
if __name__ == '__main__':
    main()