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


def bfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored

    # initialization
    start = maze.getStart()
    end = maze.getObjectives()[0]

    # print ("start:", start)
    # print ("obj:", end)

    backpath = {start: start}
    frontier = [start]
    statesExplored = 0

    # BFS loop
    while (len(frontier) > 0):
        # pull 1 coordinate from frontier
        statesExplored += 1
        current = frontier.pop(0)
        # print (current)
        
        # check if end
        if current == end:
            path = []
            
            # reverse backpath
            while current != start:
                path.insert(0, current)
                current = backpath[current]
            path.insert(0, start)

            # print (path)
            return path, statesExplored


        # loop through neighbors
        neighbors = maze.getNeighbors(current[0], current[1])
        for n in neighbors:
            # ignore those already explored, since they already have the shortest backpath
            if n in backpath.keys():
                continue
            backpath[n] = current
            frontier.append(n)


    return [], 0

statesExplored = 0

def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored

    # initialization
    start = maze.getStart()
    end = maze.getObjectives()[0]

    explored = set()
    path = []
    statesExplored = [0]

    # use recursion to do dfs
    def dfsHelper(current):

        statesExplored[0] = statesExplored[0] + 1
        explored.add(current)

        if current == end:
            path.append(current)
            return True

        for n in maze.getNeighbors(current[0], current[1]):
            if n not in explored:
                if dfsHelper(n):
                    path.append(current)
                    return True
        
        return False

    dfsHelper(start)
    path.reverse()

    return path, statesExplored[0]


def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    return [], 0


def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    return [], 0