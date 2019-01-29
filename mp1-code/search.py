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

def manhattan(c1, c2):
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])

class GreedyQueue(object): 
    def __init__(self, target):
        self.target = target
        self.heap = []
    
    def __len__(self):
        return len(self.heap)
  
    # for inserting an element in the queue 
    def insert(self, node): 
        heapq.heappush(self.heap, (manhattan(node,self.target), node))
  
    # for popping an element based on Priority 
    def pop(self):
        return heapq.heappop(self.heap)[1]
        
class FIFO(object):
    def __init__(self):
        self.queue = []

    def __len__(self):
        return len(self.queue)
    
    def pop(self):
        return self.queue.pop(0)
    
    def insert(self, node):
        self.queue.append(node)

class LIFO(object):
    def __init__(self):
        self.queue = []

    def __len__(self):
        return len(self.queue)
    
    def pop(self):
        return self.queue.pop()
    
    def insert(self, node):
        self.queue.append(node)

# since all search mechanics are the same, changing the frontier datastructure type is sufficient
def comboSearch(maze, frontier):

    # initialization
    start = maze.getStart()
    end = maze.getObjectives()[0]

    backpath = {start: start}
    frontier.insert(start)
    statesExplored = 0

    while len(frontier) > 0:
        # pull 1 coordinate from frontier
        statesExplored += 1
        current = frontier.pop()
        
        # check if end
        if current == end:
            path = []
            
            # reverse backpath
            while current != start:
                path.append(current)
                current = backpath[current]
            path.append(start)
            path.reverse()

            return path, statesExplored


        # loop through neighbors
        for n in maze.getNeighbors(current[0], current[1]):
            # ignore those already explored, since they already have the shortest backpath
            if n in backpath.keys():
                continue
            backpath[n] = current
            frontier.insert(n)
    
    return [], 0
    

def bfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored

    return comboSearch(maze, FIFO())

def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored

    return comboSearch(maze, LIFO())

def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    
    return comboSearch(maze, GreedyQueue(maze.getObjectives()[0]))


def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored

    # initialization
    frontier = []
    start = maze.getStart()
    end = maze.getObjectives()[0]

    backpath = {start: (start,0)}
    
    heapq.heappush(frontier, (0,start))
    statesExplored = 0

    while len(frontier) > 0:
        (cCost, cNode) = heapq.heappop(frontier)
        statesExplored += 1

        if cNode == end:
            path = []
            
            # reverse backpath
            while cNode != start:
                path.append(cNode)
                cNode = backpath[cNode][0]
            path.append(start)
            path.reverse()

            return path, statesExplored
        
        cNodeParent, cNodeCost = backpath[cNode]

        for n in maze.getNeighbors(cNode[0], cNode[1]):
            if n in backpath.keys():
                continue
            
            p = cNodeCost + 1
            h = manhattan(n, end)
            f = p + h
            backpath[n] = (cNode, p)
            heapq.heappush(frontier, (f, n))

    return [], 0