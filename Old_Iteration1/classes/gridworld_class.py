# ====================== Overview ====================== #
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

# Import functions
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

import Player_class as pc
import Game_graph_class as gg

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
        self.players[player_name] = pc.Player(player_name, self, player_transitions_type, player_type)

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
        self.G = gg.GameGraph(self.Nrows, self.Ncols, self.static_obstacles, self.cell2node, self.node2cell, self.players)
    
    def get_game_graph(self):
        return self.G

    def base_plot(self, fignum):
        lw = 2

        fig = plt.figure(fignum)
        ax = plt.gca()

        # Setting up grid and static obstacles:
        # Grid matrix has extra row and column to accomodate the 1-indexing of the gridworld
        grid_matrix = np.zeros((self.Nrows+1, self.Ncols+1))
        # Positioning static obstacles:
        for s in self.static_obstacles:
            scell = self.node2cell[s]
            grid_matrix[scell[0]+1][scell[1]] = 2

        cmap = ListedColormap(['w', 'k', 'r'])
        im = ax.imshow(grid_matrix, cmap=cmap)
        # Setting up gridlines
        ax.set_xlim(0.5,self.Ncols+0.5)
        ax.set_ylim(self.Nrows+0.5, 0.5)

        # Labels for major ticks
        ax.set_xticklabels(np.arange(1, self.Ncols+1, 1), minor='True')
        ax.set_yticklabels(np.flipud(np.arange(1, self.Nrows+1, 1)), minor='True')

        # Gridlines based on minor ticks
        ygrid_lines = np.flipud(np.arange(1, self.Nrows+1, 1)) - 0.5
        xgrid_lines = np.arange(1, self.Ncols+1, 1) - 0.5
        
        # Major ticks
        ax.set_xticks(xgrid_lines)
        ax.set_yticks(ygrid_lines)

        # Minor ticks
        ax.set_yticks(ygrid_lines+0.5, minor='True')
        ax.set_xticks(xgrid_lines+0.5, minor='True')
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=lw)
        plt.setp(ax.get_xmajorticklabels(), visible=False)
        plt.setp(ax.get_ymajorticklabels(), visible=False)

        return ax, im

    # W: winning set, env_locations are cells on the grid that the environment can occupy
    def plot_win_set(self, W, env_locations, fignum):
        ax, im = self.base_plot(fignum)

        # Plotting environment regions on grid
        env_color = cm.get_cmap('Blues', 128)
        sys_color = cm.get_cmap('Greens', 128)
