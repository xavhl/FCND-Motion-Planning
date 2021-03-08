from enum import Enum
from queue import PriorityQueue
from queue import LifoQueue # from collections import deque
from queue import Queue
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from bresenham import bresenham
import networkx as nx
import matplotlib.pyplot as plt
import random


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)



def create_grid_and_edges(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    along with Voronoi graph edges given obstacle data and the
    drone's altitude.
    """
    # minimum and maximum north coordinate
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))
    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))
    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min)))
    east_size = int(np.ceil((east_max - east_min)))
    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    # Center offset for grid
    north_min_center = np.min(data[:, 0])
    east_min_center = np.min(data[:, 1])
    
    # Define a list to hold Voronoi points
    points = []
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(north - d_north - safety_distance - north_min_center),
                int(north + d_north + safety_distance - north_min_center),
                int(east - d_east - safety_distance - east_min_center),
                int(east + d_east + safety_distance - east_min_center),
                ]
            grid[obstacle[0]:obstacle[1], obstacle[2]:obstacle[3]] = 1
    
            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])

    # create a voronoi graph based on location of obstacle centres
    graph = Voronoi(points)
    # voronoi_plot_2d(graph); plt.show()
    # check each edge from graph.ridge_vertices for collision
    edges = []
    for edge in graph.ridge_vertices:
        point1 = graph.vertices[edge[0]]
        point2 = graph.vertices[edge[1]]
        
        cells = list(bresenham(int(point1[0]), int(point1[1]), int(point2[0]), int(point2[1])))
        infeasible = False
    
        for cell in cells:
            if np.amin(cell) < 0 or cell[0] >= grid.shape[0] or cell[1] >= grid.shape[1]:
                infeasible = True
                break
            if grid[cell[0], cell[1]] == 1:
                infeasible = True
                break
        if infeasible == False:
            point1 = (point1[0], point1[1])
            point2 = (point2[0], point2[1])
            edges.append((point1,point2))
            
    return grid, int(north_min), int(east_min), edges


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    return valid_actions


def a_star(grid, h, start, goal):

    path = [] # [node]
    path_cost = 0
    queue = PriorityQueue() # [(queue_cost, next_node)]
    queue.put((0, start))
    visited = set(start) # [node]

    branch = {} # {next_node: (branch_cost, current_node, action)}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost


def dfs(grid, _, start, goal):
    
    path = []
    path_cost = 0
    stack = LifoQueue()
    stack.put(start)
    visited = set(start)
    
    branch = {}
    found = False
    
    while not stack.empty():
        # item = stack.get()
        current_node = stack.get() # item
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]
            
        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                
                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    stack.put(next_node)
    
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost


def iterative_astar(grid, h, start, goal):
    def a_star_for_iterative_astar(threshold):
        # this function explores child nodes with heuristic <= threshold
        #               outputs (bool isFound, list path, int path_cost, list threshold_list)
        #               if isFound == True,  (True, path[::-1], path_cost, None)
        #               if isFound == False, (False, None, None, threshold_list)
        
        path = []
        path_cost = 0
        queue = PriorityQueue()
        queue.put((0, start))
        visited = set(start)

        branch = {}
        found = False
    
        threshold_list = [] # records queue_cost of child nodes from previous iteration
        
        while not queue.empty():
            item = queue.get()
            current_node = item[1]
            if current_node == start:
                current_cost = 0.0
            else:
                current_cost = branch[current_node][0]
                
            if current_node == goal:
                print('Found a path.')
                found = True
                break
            else:
                for action in valid_actions(grid, current_node):
                    # get the tuple representation
                    da = action.delta
                    next_node = (current_node[0] + da[0], current_node[1] + da[1])
                    branch_cost = current_cost + action.cost
                    queue_cost = branch_cost + h(next_node, goal)
                    
                    if queue_cost <= threshold:
                        if next_node not in visited:
                            visited.add(next_node)
                            branch[next_node] = (branch_cost, current_node, action)
                            queue.put((queue_cost, next_node))
                    else: # if queue_cost > threshold
                        threshold_list.append(queue_cost)
        
        if found:
            # retrace steps
            n = goal
            path_cost = branch[n][0]
            path.append(goal)
            while branch[n][1] != start:
                path.append(branch[n][1])
                n = branch[n][1]
            path.append(branch[n][1])
            
            return (True, path[::-1], path_cost, None)
        else:
            return (False, None, None, threshold_list)
            #situation A: all nodes visited, empty threshold_list; no more following actions; terminate
            #situation B: there exists following actions, non empty list; continue next round of iterative a star
    
    
    path = []
    path_cost = 0
    threshold_prev = h(start, goal)

    while True:
    
        (found, path, path_cost, threshold_list) = a_star_for_iterative_astar(threshold_prev)
        
        if found:
            break
        else:
            if threshold_list: # situation A: if has elements
                threshold_prev = min(threshold_list)
            else: # situation B: all thresholds <= threshold_prev # all possibilities exhausted
                break
    
    if found:
        pass
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost
    
def ucs(grid, _, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                #get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        #retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost


def heuristic(position, goal_position): # Euclidean distance; L2 norm
    return np.linalg.norm(np.array(position) - np.array(goal_position))

def heuristic2(position, goal_position): # Manhattan distance; L1 norm
    return np.linalg.norm(np.array(position) - np.array(goal_position), ord=1)


def bfs(graph, _, start, goal):

    path = []
    path_cost = 0
    queue = Queue()
    queue.put(start)
    visited = set(start)
    
    branch = {}
    found = False
    
    while not queue.empty():
        # item = queue.get()
        current_node = queue.get() # item
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]
            
        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph.neighbors(current_node):# neighbors(G, n)
                branch_cost = current_cost + 1 # action_cost = 1

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node)
                    queue.put(next_node)
    
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost
    

def a_star_traverse(grid, h, start, goal, traverse_points):
    total_path = [start]
    prev = start # previous traversed point
    
    while 1 < len(traverse_points):
        euclidean_dist = [heuristic(prev, p) for p in traverse_points] # calculate distance for each point in list
        argmin = euclidean_dist.index(min(euclidean_dist)) #argmin
        
        nearest_point = traverse_points[argmin]
        path, _ = a_star(grid, heuristic, prev, nearest_point) # print('Found path:', path)
        total_path.extend(path[1:]) # skip first point to avoid duplicate
        traverse_points.remove(nearest_point)
        prev = nearest_point

    nearest_point = traverse_points[0] # last point in list
    path, _ = a_star(grid, heuristic, prev, nearest_point)
    total_path.extend(path[1:])
    traverse_points.remove(nearest_point)
    prev = nearest_point

    path, _ = a_star(grid, heuristic, prev, goal) # goal point
    total_path.extend(path[1:])

    # print('Found total path:', total_path)
    return total_path, _
    