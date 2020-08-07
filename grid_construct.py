# Apurva Badithela 5/23/2020
# This script generates gridworlds with static and moving obstacles
# This is just to test something for github
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation

def grid(M, N, static_obs, mov_obs_col):
    # Number of states and environment
    Ns = M*N
    Ne = M*N 
    env_states = [] # States controlled by the environment
    mov_obs_states = []
    T1 = [] # Transition for the environment
    T2 = [] # Transition for the system
    for row in range(1, M+1):
        for col in range(1, N+1):
            cell = (row - 1)*N + col
            if([row,col] in static_obs):
                env_states.append(cell)
            
            T1_transition = [cell]
            if col == mov_obs_col:
                env_states.append(cell)
                mov_obs_states.append(cell)

                if(row == 1):
                    T1_transition = [cell, cell+N]
                elif(row == M):
                    T1_transition = [cell, cell-N]
                else:
                    T1_transition = [cell, cell+N, cell-N]
            
            T2_transition = [cell]
            if(cell not in static_obs):
                if(row == 1):
                    t_cell = cell+N
                    if([row+1, col] not in static_obs):
                        T2_transition.append(t_cell)
                elif(row == M):
                    t_cell = cell-N
                    if([row-1, col] not in static_obs):
                        T2_transition.append(t_cell)
                else:
                    t1_cell = cell+N
                    t2_cell = cell-N
                    if([row+1, col] not in static_obs):
                        T2_transition.append(t1_cell)
                    if([row-1, col] not in static_obs):
                        T2_transition.append(t2_cell)
                
                if(col == 1):
                    if([row, col+1] not in static_obs):
                        T2_transition.append(cell+1)
                elif(col == N):
                    if([row, col-1] not in static_obs):
                        T2_transition.append(cell-1)
                else:
                    if([row, col+1] not in static_obs):
                        T2_transition.append(cell+1)
                    if([row, col-1] not in static_obs):
                        T2_transition.append(cell-1)
            T1.append(T1_transition)
            T2.append(T2_transition)
    
    return [Ns, Ne, env_states, mov_obs_states, T1, T2]

# Obstacle remains at the endpoints and does not patrol. This is due to avoiding the complexity associated with the grid patrolling behavior
def grid2(M, N, static_obs, mov_obs_col):
    # Number of states and environment
    Ns = M*N
    Ne = M*N 
    env_states = [] # States controlled by the environment
    mov_obs_states = []
    T1 = [] # Transition for the environment
    T2 = [] # Transition for the system
    for row in range(1, M+1):
        for col in range(1, N+1):
            cell = (row - 1)*N + col
            if([row,col] in static_obs):
                env_states.append(cell)
            
            T1_transition = [cell]
            if col == mov_obs_col:
                env_states.append(cell)
                mov_obs_states.append(cell)

                if(row == 1):
                    T1_transition = [cell]
                elif(row == M):
                    T1_transition = [cell]
                else:
                    T1_transition = [cell, cell+N, cell-N]
            
            T2_transition = [cell]
            if(cell not in static_obs):
                if(row == 1):
                    t_cell = cell+N
                    if([row+1, col] not in static_obs):
                        T2_transition.append(t_cell)
                elif(row == M):
                    t_cell = cell-N
                    if([row-1, col] not in static_obs):
                        T2_transition.append(t_cell)
                else:
                    t1_cell = cell+N
                    t2_cell = cell-N
                    if([row+1, col] not in static_obs):
                        T2_transition.append(t1_cell)
                    if([row-1, col] not in static_obs):
                        T2_transition.append(t2_cell)
                
                if(col == 1):
                    if([row, col+1] not in static_obs):
                        T2_transition.append(cell+1)
                elif(col == N):
                    if([row, col-1] not in static_obs):
                        T2_transition.append(cell-1)
                else:
                    if([row, col+1] not in static_obs):
                        T2_transition.append(cell+1)
                    if([row, col-1] not in static_obs):
                        T2_transition.append(cell-1)
            T1.append(T1_transition)
            T2.append(T2_transition)
    
    return [Ns, Ne, env_states, mov_obs_states, T1, T2]
# The gridworld is of size M-by-N. 
# static_obs is the row-column location of static obstacles in the gridworld
# move_obs_col is the column of in which moving obstacles move
# bridge: list of states on the gridworld in which the obstacle must wait in for a certain amount of time before deciding to move in order for the system to have a feasible path to the goal
# If cell is in bridge, the environment must check another extra variable before deciding to move out of the bridge variable. This extra variable is checking if there is a path for the system to continue to satisfy its specifications
# Let bridge = [cell1, cell2], then the environment being in cell1 and cell2 has two states corresponding to it: The normal cell1 and cell2 from which it is free to transition normally, and cell1_prime and cell2_prime in which it is compelled to stay in the bridge location until the system
# has a clear trajectory to continue towards its specification.

def grid_bridge(M, N, static_obs, mov_obs_col, bridge):
    # Number of states and environment
    Ns = M*N
    Ne = M*N + len(bridge)
    env_states = [] # States controlled by the environment
    mov_obs_states = []
    T1 = [] # Transition for the environment
    T2 = [] # Transition for the system
    T1_prime = []
    for row in range(1, M+1):
        for col in range(1, N+1):
            cell = (row - 1)*N + col
            if([row,col] in static_obs):
                env_states.append(cell)
            
            T1_transition = [cell]
            if col == mov_obs_col:
                env_states.append(cell)
                mov_obs_states.append(cell)

                # Setup extra variables in which if the obstacle has just arrived at a bridge position, it must stay there until a guard indicates that it is free to return to the "normal" version of the cell
                if(cell in bridge):
                    cell_idx = bridge.index(cell)
                    cell_prime = M*N + cell_idx
                    T1_prime.append([cell_prime])

                if(row == 1):
                    T1_transition = [cell, cell+N]
                elif(row == M):
                    T1_transition = [cell, cell-N]
                else:
                    T1_transition = [cell, cell+N, cell-N]
            
            T2_transition = [cell]
            if(cell not in static_obs):
                if(row == 1):
                    t_cell = cell+N
                    if([row+1, col] not in static_obs):
                        T2_transition.append(t_cell)
                elif(row == M):
                    t_cell = cell-N
                    if([row-1, col] not in static_obs):
                        T2_transition.append(t_cell)
                else:
                    t1_cell = cell+N
                    t2_cell = cell-N
                    if([row+1, col] not in static_obs):
                        T2_transition.append(t1_cell)
                    if([row-1, col] not in static_obs):
                        T2_transition.append(t2_cell)
                
                if(col == 1):
                    if([row, col+1] not in static_obs):
                        T2_transition.append(cell+1)
                elif(col == N):
                    if([row, col-1] not in static_obs):
                        T2_transition.append(cell-1)
                else:
                    if([row, col+1] not in static_obs):
                        T2_transition.append(cell+1)
                    if([row, col-1] not in static_obs):
                        T2_transition.append(cell-1)
            T1.append(T1_transition)
            T2.append(T2_transition)
    
    T1.extend(T1_prime)
    return [Ns, Ne, env_states, mov_obs_states, T1, T2]

# Plotting function for a gridworld trajectory that is of size M-by-N
def plot_grid_world(env_traj, sys_traj, static_obs, M, N):
    # Plotting tools
    alw = 0.75    # AxesLineWidth
    fsz = 12      # Fontsize
    lw = 2        # LineWidth
    msz = 6       # MarkerSize

    # Plotting system trajectory
    Cr = env_traj.copy()    
    Xr = sys_traj.copy()
    
    if static_obs:
        fig, ax = base_plot(M, N, static_obs)
    else: 
        fig, ax = base_plot(M, N, [])
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


def base_plot(M, N, static_obs):
    msz = 6       # MarkerSize
    goal_cell_x = N
    goal_cell_y = 1

    grid_loc_x = []
    grid_loc_y = []
    for m in range(1,M+1):
        for n in range(1,N+1):
            grid_loc_x.append(m)
            grid_loc_y.append(n)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(grid_loc_x, grid_loc_y, 'o', markersize=msz, markerfacecolor='black')
    plt.plot(goal_cell_x, goal_cell_y, 'o', markersize=msz, markerfacecolor='blue')
    plt.text(goal_cell_x + 0.1, goal_cell_y + 0.2, "Goal")

    if static_obs:
        static_obs_x= []
        static_obs_y = []
        for obs in static_obs:
            static_obs_x.append(obs[1])
            static_obs_y.append(obs[0])
        plt.plot(static_obs_x, static_obs_y, 's', markersize=msz, markerfacecolor='red')
    ax.set_xlim(0,N+1)
    ax.set_ylim(0,M+1)
    plt.xticks(np.arange(1,N+1))
    plt.yticks(np.arange(1, M+1))
    
    return fig, ax

