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

def shortestBetweenObjectives(maze, s, objs):

    frontier = FIFO()
    frontier.insert(State(s, None, 0, 0, 0, set([])))

    explored = set()
    remainingObjs = set(objs)

    objsmap = {}

    while len(frontier) > 0:
        currentState = frontier.pop()
        
        # if currentstate hit an objective
        if currentState.coord in remainingObjs:

            objsmap[currentState.coord] = currentState.g
            remainingObjs.remove(currentState.coord)

            if len(remainingObjs) == 0:
                return objsmap


        # based on current state, move into neighbors
        for n in maze.getNeighbors(currentState.coord[0], currentState.coord[1]):

            # create new child with same objs
            child = currentState.newSimpleChild(n, currentState.g + 1, 0)

            if hash(child) not in explored:
                frontier.insert(child)
                explored.add(hash(child))
    
    return [], 0


def memoMST(maze):

    objs = maze.getObjectives()

    truedist = {}

    for i, n in enumerate(objs):
        truedist[n] = shortestBetweenObjectives(maze, n, objs)

    # print (truedist)

    memo = {}

    def MST(nodes):

        totalCost = 0

        visited = set()
        unvisited = set(nodes)
        
        visited.add(unvisited.pop())

        # prims algorithm
        while len(unvisited) > 0:
            # choose neighbor based on minimum manhattan distance
            minDist, minU = min([ min([ (truedist[u][v], u) for u in unvisited ]) for v in visited ])
            visited.add(minU)
            unvisited.remove(minU)
            totalCost += minDist
        
        return totalCost
    
    return MST
        
class State(object):
    def __init__(self, coord, parent, g, f, MST, unvisitedObjs, unvisitedHash=None):
        self.coord = coord
        self.parent = parent
        self.g = g
        self.f = f
        self.MST = MST
        self.unvisitedObjs = unvisitedObjs
        self.unvisitedHash = hash(str(self.unvisitedObjs)) if unvisitedHash==None else unvisitedHash
        self.stateHash = hash((self.coord, self.unvisitedHash))

    def __str__(self):
        if self.parent != None:
            return "coord: " + str(self.coord) + "\tparent: " + str(self.parent.coord) + "\tunvisitedObjs: " + str(self.unvisitedObjs)
        return "coord: " + str(self.coord) + "\tparent: None" + "\tunvisitedObjs: " + str(self.unvisitedObjs)
    
    def __hash__(self):
        return self.stateHash

    def __lt__(self, other):
        return self.f < other.f

    def newSimpleChild(self, coord, g, f):
        return State(coord, self, g, f, self.MST, self.unvisitedObjs, self.unvisitedHash)

    def update(self, MST, unvisitedObjs):
        self.MST = MST
        self.unvisitedObjs = unvisitedObjs
        self.unvisitedHash = hash(str(unvisitedObjs))
        self.stateHash = hash((self.coord, self.unvisitedHash))
        
    
    def getPathFromRoot(self):
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

    # Used memoized MSTcost calculation
    MSTcost = memoMST(maze)

    # setup start state
    s = State(start, None, 0, 0, MSTcost(maze.getObjectives()), set(maze.getObjectives()))
    frontier.insert(s)

    # store exploredStates
    exploredStates = 0
    explored = {hash(s):0}
    ignore = set()

    while len(frontier) > 0:
        currentState = frontier.pop()

        # skip if already found shorter path to state
        if hash(currentState) in explored and explored[hash(currentState)] < currentState.g:
            continue

        exploredStates += 1
        
        # if currentstate hit an objective
        if currentState.coord in currentState.unvisitedObjs:

            temp.append(currentState.coord)

            # done
            if len(currentState.unvisitedObjs) <= 1:
                print ("Time Elapsed:", time() - start_time)
                return currentState.getPathFromRoot(), exploredStates
            
            # update current state to reflect retriving obj
            newObjs = set(currentState.unvisitedObjs)
            newObjs.remove(currentState.coord)
            newMST = MSTcost(list(newObjs))

            currentState.update(newMST, newObjs)

            # update explored set
            explored[hash(currentState)] = currentState.g


        # based on current state, move into neighbors
        for n in maze.getNeighbors(currentState.coord[0], currentState.coord[1]):
            g = currentState.g + 1
            h = min([manhattan(n, target) for target in currentState.unvisitedObjs]) + currentState.MST
            f = heuristic(g,h)

            # create new child with same objs
            child = currentState.newSimpleChild(n, g, f)

            if hash(child) not in explored or g < explored[hash(child)]:
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