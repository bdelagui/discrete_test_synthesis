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
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os
import math
import networkx as nx
import pdb

import gridworld_class as gc
import Player_class as pc
import Game_graph_class as gg
# -----------------------------------------------------------------------------------------------------------------------------------------#
# % ======= Main function to test the classes ========== % #
def main():
    ex = 2
    if ex == 1:
        M = 10
        N = 10
        # Construct gridworld states and transitions:
        GW = gc.GridWorld([M,N])
        # Adding players and transitions:
        GW.grid_transitions()
        obs_col  = M-3
        tester_col = obs_col+1
        tester_bump_row = math.floor(N/2)
        agent_transitions = ['all']
        tester_transitions = ['specific', tester_col, tester_bump_row]
        sys_goal_cell = [M, N]
        env_goal_cell = [[ii, tester_col] for ii in range(1,M+1) if ii!=tester_bump_row]
        extra_cells = [[tester_bump_row-1, tester_col+1], [tester_bump_row, tester_col+1], [tester_bump_row+1, tester_col+1]]
        env_goal_cell.extend(extra_cells)
    
    if ex == 2:
        M = 10
        N = 10
        # Construct gridworld states and transitions:
        obs_col  = M-3
        GW = gc.GridWorld([M,N])
        static_obs = [[ii, obs_col] for ii in range(1,N-1)]
        GW.add_static_obstacles(static_obs)
        # Adding players and transitions:
        GW.grid_transitions()
        tester_col = obs_col+1
        agent_transitions = ['all']
        tester_transitions = ['specific', tester_col]
        sys_goal_cell = [M, N]
        env_goal_cell = [[ii, tester_col] for ii in range(1,M+1)]
    
    if ex == 3:
        M = 4
        N = 4
        # Construct gridworld states and transitions:
        GW = gc.GridWorld([M,N])
        # Adding players and transitions:
        GW.grid_transitions()
        Tsys_list = [(1,[1,2,5]), (2,[1,2,3,6]), (3,[2,3,4,7]), (4,[3,4,8]), (5,[1,5,6,9]), (6,[2,5,6,7,10]), (7,[3,6,7,8,11]), (8,[4,7,8,12]), (9,[5,9,10,13]), (10,[6,9,10,11,14]), (11,[7,10,11,12,15]), (12,[8,11,12,16]), (13,[9,13,14]), (14,[10,13,14,15]), (15,[11,14,15,16]), (16,[12,15,16])] # System transitions
        Tsys = dict(Tsys_list)
        agent_transitions=['very_specific', Tsys]
        
        Tenv_list = [(1, [1]), (2,[3,6]), (3,[2,7]), (4, [4]), (5, [5]), (6,[2,10]), (7,[3,11]), (8, [8]), (9, [9]), (10, [6,14]), (11, [7,15]), (12, [12]), (13, [13]), (15, [11,14]), (14, [10, 15]), (16, [16])]
        
        Tenv = dict(Tenv_list)
        tester_transitions = ['very_specific', Tenv]
        sys_goal_cell = [M,N]
        env_goal_cell = [[1,2], [1,3], [2,2], [2,3], [3,2], [3,3], [4,2], [4,3]]
    
    # Simple runner blocker:
    if ex == 4:
        M = 5
        N = 1
        Ns = 5
        Ne = 5
        # Construct gridworld states and transitions:
        GW = gc.GridWorld([M,N])
        # Adding players and transitions:
        GW.grid_transitions()
        Tenv_list = [(1,[1]), (2,[3]), (3, [2,4]), (4, [3]), (5, [5])]
        Tenv = dict(Tenv_list)
        Tsys_list = [(1, [1,2,3,4]), (2, [1,2,3,5]), (3, [1,2,3,4,5]), (4, [1,3,4,5]), (5, [2,3,4,5])]
        Tsys = dict(Tsys_list)
        agent_transitions=['very_specific', Tsys]
        tester_transitions = ['very_specific', Tenv]
        sys_goal_cell = [M,N]
        env_goal_cell = [[2,1], [3,1], [4,1]]
    #--------------------------------- Comment Below to get plot -------------------------------
    # M = 10
    # N = 10
    # GW = GridWorld([M,N])
    # ex = 2

    # # Setting STATIC obstacles:
    # obs_col = 7
    # static_obs = [[ii, obs_col] for ii in range(1,N-1)]
    # GW.add_static_obstacles(static_obs)

    # # Construct gridworld states and transitions:
    # GW.grid_transitions()

    # # Testing base plot
    # fig = 1
    # ax, im = GW.base_plot(fig)
    # plt.show()

    # # Adding players and transitions:
    # tester_col = obs_col+1
    # if ex == 1:
    #     tester_bump_row = math.floor(N/2)
    #     agent_transitions = ['all']
    #     tester_transitions = ['specific', tester_col, tester_bump_row]
    #     sys_goal_cell = [M, N]
    #     env_goal_cell = [[ii, tester_col] for ii in range(1,M+1) if ii!=tester_bump_row]
    #     extra_cells = [[tester_bump_row-1, tester_col+1], [tester_bump_row, tester_col+1], [tester_bump_row+1, tester_col+1]]
    #     env_goal_cell.extend(extra_cells)
    # if ex == 2:
    #     agent_transitions = ['all']
    #     tester_transitions = ['specific', tester_col]
    #     sys_goal_cell = [M, N]
    #     env_goal_cell = [[ii, tester_col] for ii in range(1,M+1)]

    # -------------------------------------- Keep things below this line the same ------------------------------------------------------

    GW.add_player('agent', agent_transitions, 's')
    GW.add_player('tester', tester_transitions, 'e')

    # Checking player transitions have been correctly added:
    tester = GW.get_player('tester')
    tester_transitions2 = tester.get_transitions()

    # Making game graph and getting vertices and edges:
    GW.construct_graph()
    GAME = GW.get_game_graph()
    E, V = GAME.get_edges_vertices()

    # Add trace here to make sure that the edges and vertices are set correctly
    # pdb.set_trace()

    # Finding the system winning set and value function:
    goal_cells = [[env_cell, sys_goal_cell] for env_cell in env_goal_cell]
    goal_vertices = GAME.set_win_states(goal_cells)
    assert(all([elem in V for elem in goal_vertices[0]]))
    assert(all([elem in V for elem in goal_vertices[1]]))

    # Robust Pre for system winning set computation:
    quant_env = 'forall'
    quant_sys = 'exists'
    win_agent = 's'
    Wsys2 = GAME.win_reach(win_agent, goal_vertices, quant_env, quant_sys)
    assert(Wsys2[-1]==Wsys2[-2]) # Assertion to make sure we have the right fixpoint or else we need to increase the threshold in Game_graph_class.py
    # Wsys, Wsys_env, Wsys_sys = GAME.synt_winning_set2(win_agent, goal_vertices, quant_env, quant_sys)
    Val_sys = GAME.get_value_function(win_agent)
    Wenv_phisys, Wsys_phisys = GAME.win_cells(Wsys2)

    # # Testing base plot
    fig = 1
    # ax, im = GW.base_plot(fig)
    GW.plot_win_set(W, env_locations, fignum)
    plt.show()

    # Add trace here for the winning set synthesis
    pdb.set_trace()

    # Finding the environment winning set and value function:
    # !!!! The environment and the system places need to be switched
    # IMPORTANT_NOTE: When we find the environment winning set, we need to have environment and system cells switched
    if ex == 1:
        bump_row=5
        sys_goal_cell = [[1, obs_col], [M, obs_col]]
        env_goal_cell = [[ii, tester_col] for ii in range(1,M+1) if ii!=bump_row]
        extra_cells = [[bump_row-1, tester_col+1], [bump_row, tester_col+1], [bump_row+1, tester_col+1]]
        env_goal_cell.extend(extra_cells)
    if ex == 2:
        sys_goal_cell = [[1, obs_col], [M, obs_col]]
        env_goal_cell = [[ii, tester_col] for ii in range(1,M+1)]

    goal_cells = [[env_cell, sys_cell] for env_cell in env_goal_cell for sys_cell in sys_goal_cell]
    goal_vertices = GAME.set_win_states(goal_cells)

    # Robust Pre for system winning set computation:
    quant_env = 'exists'
    quant_sys = 'exists'
    win_agent = 'e'
    W_env = GAME.win_reach(win_agent, goal_vertices, quant_env, quant_sys)
    Val_env = GAME.get_value_function(win_agent)
    Wenv_phienv, Wsys_phienv = GAME.win_cells(W_env)
    #   Printing the value functions:
    print(Val_sys)
if __name__ == '__main__':
    main()