# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import heapq
# this is part of standard python lib?

# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod)(maze)

# f = g + min(h)
def AStarHeuristics(g,hs):
    return g + min(hs)

# f = min(h)
def GreedyHeuristics(g,hs):
    return min(hs)

def NoHeuristics(g,hs):
    return 0

def manhattan(c1, c2):
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])

# Priority Queue for Greedy and AStar, fval used
class PriorityQueue(object): 
    def __init__(self):
        self.heap = []
    
    def __len__(self):
        return len(self.heap)
  
    # for inserting an element in the queue with specified priority value fval
    def insert(self, fval, node): 
        heapq.heappush(self.heap, (fval, node))
  
    # for popping an element based on Priority 
    def pop(self):
        return heapq.heappop(self.heap)[1]

    def clear(self):
        self.heap.clear()

# FIFO QUEUE for BFS, fval ignored
class FIFO(object):
    def __init__(self):
        self.queue = []

    def __len__(self):
        return len(self.queue)
    
    def pop(self):
        return self.queue.pop(0)
    
    def insert(self, fval, node):
        self.queue.append(node)

    def clear(self):
        self.queue.clear()

# LIFO QUEUE for BFS,  fval ignored
class LIFO(object):
    def __init__(self):
        self.stack = []

    def __len__(self):
        return len(self.stack)
    
    def pop(self):
        return self.stack.pop()
    
    def insert(self, fval, node):
        self.stack.append(node)
    
    def clear(self):
        self.stack.clear()

# since all search mechanics are the same, changing the frontier datastructure type is sufficient
def comboSearch(maze, frontier, heuristic):

    # initialization
    start = maze.getStart()
    objectives = set(maze.getObjectives())

    # explored mappings for finding parent and cost
    exploredParent = {start: start}
    exploredCost = {start: 0}
    statesExplored = 0

    # add start
    frontier.insert(0, start)

    tpath = [start]

    while len(frontier) > 0:
        # pull 1 coordinate from frontier
        current = frontier.pop()
        statesExplored += 1
        
        # check if hit obj
        if current in objectives:

            objectives.remove(current)

            # tpath.append(current)

            temp = current
            # reverse from temp to start to get path
            while temp != start:
                tpath.insert(1,temp)
                temp = exploredParent[temp]
            
            if len(objectives) == 0:
                return tpath, statesExplored

            # reset and start from current
            start = current
            exploredParent.clear()
            exploredParent[start] = start
            exploredCost.clear()
            exploredCost[start] = 0
            frontier.clear()
            frontier.insert(0, start)


        # loop through neighbors
        for n in maze.getNeighbors(current[0], current[1]):

            # calculate heuristics and apply heuristic function
            g = exploredCost[current] + 1
            h = [manhattan(n, target) for target in objectives]
            f = heuristic(g, h)

            # replacement with shorter backpath
            if n not in exploredCost.keys() or g < exploredCost[n]:
                exploredParent[n], exploredCost[n] = current, g
                frontier.insert(f, n)

    
    return [], 0
    
def bfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored

    return comboSearch(maze, FIFO(), NoHeuristics)

def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored

    return comboSearch(maze, LIFO(), NoHeuristics)

def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    
    return comboSearch(maze, PriorityQueue(), GreedyHeuristics)


def astar(maze):

    # TODO: Write your code here
    # return path, num_states_explored

    return comboSearch(maze, PriorityQueue(), AStarHeuristics)