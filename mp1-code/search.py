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
from time import time

# TODO: Implement filtering of overridden states. (States with shorter path to achieve)

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

# HEURISTIC FOR MULTI DOT
# h = est2nearest + cost(MST of unvisited)

# Calculate estimated MST of list of unvisited nodes
# Will always be less than true MST because using manhattan distance between vertices
def MSTcost(nodes):

    if len(nodes) < 1:
        return 0

    totalCost = 0
    untouched = set(nodes)
    
    current = nodes[0]
    untouched.remove(current)

    # prims algorithm
    for i in range(len(nodes) - 1):
        # choose neighbor based on minimum manhattan distance
        minDist, current = min([(manhattan(current,neighbor), neighbor) for neighbor in nodes if neighbor in untouched])
        untouched.remove(current)
        totalCost += minDist
    
    return totalCost
        
class State(object):
    def __init__(self, coord, parent, gVal, hVal, fVal, MST, unvisitedObjs):
        self.coord = coord
        self.parent = parent
        self.gVal = gVal
        self.hVal = hVal
        self.fVal = fVal
        self.MST = MST
        self.unvisitedObjs = unvisitedObjs
        self.hashval = hash(((self.coord), str(unvisitedObjs)))

    def __str__(self):
        if self.parent != None:
            return "coord: " + str(self.coord) + "\tparent: " + str(self.parent.coord) + "\tunvisitedObjs: " + str(self.unvisitedObjs)
        return "coord: " + str(self.coord) + "\tparent: None" + "\tunvisitedObjs: " + str(self.unvisitedObjs)
    
    def __hash__(self):
        return self.hashval

    def __lt__(self, other):
        return self.fVal < other.fVal

    def newSimpleChild(self, coord, gVal, hVal, fVal):
        return State(coord, self, gVal, hVal, fVal, self.MST, self.unvisitedObjs)

    def update(self, MST, unvisitedObjs):
        self.MST = MST
        self.unvisitedObjs = unvisitedObjs
        self.hashval = hash(((self.coord), str(unvisitedObjs)))
    
    def getPath(self):
        p = []
        curr = self
        while curr.parent != None:
            p.insert(0, curr.coord)
            curr = curr.parent
        p.insert(0, curr.coord)
        return p


# since all search mechanics are the same, changing the frontier datastructure type is sufficient
def comboSearch(maze, frontier, heuristic):

    start_time = time()

    # initialization
    start = maze.getStart()
    temp = []

    # store exploredStates
    exploredStates = 0
    explored = {start:0}

    # setup start state
    frontier.insert(State(start, None, 0, 0, 0, MSTcost(maze.getObjectives()), set(maze.getObjectives())))

    while len(frontier) > 0:
        currentState = frontier.pop()
        # print (currentState)
        exploredStates += 1
        
        # if currentstate hit an objective
        if currentState.coord in currentState.unvisitedObjs:

            temp.append(currentState.coord)

            # done
            if len(currentState.unvisitedObjs) <= 1:
                print (currentState.getPath())
                print ("Time Elapsed:", time() - start_time)
                return currentState.getPath(), exploredStates
            
            # update current state to reflect retriving obj
            newObjs = set(currentState.unvisitedObjs)
            newObjs.remove(currentState.coord)
            newMST = MSTcost(list(newObjs))

            currentState.update(newMST, newObjs)

            # update explored set
            explored[hash(currentState)] = currentState.gVal


        # based on current state, move into neighbors
        for n in maze.getNeighbors(currentState.coord[0], currentState.coord[1]):
            g = currentState.gVal + 1
            h = min([manhattan(n, target) for target in currentState.unvisitedObjs])
            f = heuristic(g,h) + currentState.MST

            # create new child with same objs
            child = currentState.newSimpleChild(n, g, h, f)

            if hash(child) not in explored.keys() or g < explored[hash(child)]:
                frontier.insert(child)
                explored[hash(child)] = g
    
    return [], 0

# Return:
# path, numStatesExplored

def bfs(maze):
    return comboSearch(maze, FIFO(), NoHeuristics)

def dfs(maze):
    return comboSearch(maze, LIFO(), NoHeuristics)

def greedy(maze):
    return comboSearch(maze, PriorityQueue(), GreedyHeuristics)

def astar(maze):
    return comboSearch(maze, PriorityQueue(), AStarHeuristics)



# f = g + min(h)
def AStarHeuristics(g,h):
    return g + h

# f = min(h)
def GreedyHeuristics(g,h):
    return h

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

    def __str__(self):
        return str(self.heap)
  
    # for inserting an element in the queue with specified priority value fval
    def insert(self, node): 
        heapq.heappush(self.heap, node)
  
    # for popping an element based on Priority 
    def pop(self):
        return heapq.heappop(self.heap)

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
    
    def insert(self, node):
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
    
    def insert(self, node):
        self.stack.append(node)
    
    def clear(self):
        self.stack.clear()