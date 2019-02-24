import numpy as np
import time

# modified ALGORITHM X code from https://www.cs.mcgill.ca/~aassaf9/python/algorithm_x.html

class ALGOX(object):

    def __init__(self,X,Y):
        self.X = ALGOX._preprocess(X, Y)
        self.Y = Y
    
    def solve(self):
        return ALGOX._solve(self.X,self.Y)

    def _solve(X, Y, solution=[]):

        if not X:
            sol = list(solution)
            return True, sol
        else:
            # chose column with minimum 1s.
            # "Fill out a position with minimum choice"
            c = min(X, key=lambda c: len(X[c]))
            # "chose iteratively the choice to fill out the position"
            for r in list(X[c]):
                solution.append(r)
                cols = ALGOX._select(X, Y, r)
                ret, sol = ALGOX._solve(X, Y, solution)
                if ret:
                    return True, sol
                ALGOX._deselect(X, Y, r, cols)
                solution.pop()

        return False, []
    
    def _select(X, Y, r):
            cols = []
            for j in Y[r]:
                for i in X[j]:
                    for k in Y[i]:
                        if k != j:
                            X[k].remove(i)
                cols.append(X.pop(j))
            return cols

    def _deselect(X, Y, r, cols):
        for j in reversed(Y[r]):
            X[j] = cols.pop()
            for i in X[j]:
                for k in Y[i]:
                    if k != j:
                        X[k].add(i)

    def _preprocess(X, Y):
        Xnew = {j: set() for j in X}
        for i in Y:
            for j in Y[i]:
                Xnew[j].add(i)
        return Xnew

# check if pent can be added via element wise multiplication of pent and board
def can_add_pent(board, pent, coord):

    if coord[0] < 0 or coord[1] < 0:
        return False
    if coord[0] + pent.shape[0] > board.shape[0] or coord[1] + pent.shape[1] > board.shape[1]:
        return False

    temp = np.multiply(board[coord[0]:coord[0]+pent.shape[0], coord[1]:coord[1]+pent.shape[1]], pent)
    return not np.any(temp)

# add pent to board, if not possible, leave board unchanged
def add_pentomino(board, pent, coord):
    # check for overlap
    if not can_add_pent(board,pent,coord):
        return False
    board[coord[0]:coord[0]+pent.shape[0], coord[1]:coord[1]+pent.shape[1]] += pent
    return True

# remove pent from board
def del_pentomino(board, pent, coord):
    board[coord[0]:coord[0]+pent.shape[0], coord[1]:coord[1]+pent.shape[1]] -= pent

def generateAllPents(pents):

    def nphash(arr):
        return hash(str(arr))

    all_pents = [[] for i in range(len(pents))]
    # for every pent
    for i, pent in enumerate(pents):
        rot_pent = pent
        no_repeat = set()
        # rotate 4 times
        for rot in range(4):
            flip_pent = rot_pent
            # flip 2 times
            for flip in range(2):
                # check for identical pent, if no repeat, add to list
                if nphash(flip_pent) not in no_repeat:
                    all_pents[i].append(flip_pent)
                    no_repeat.add(nphash(flip_pent))

                flip_pent = np.fliplr(flip_pent)
            rot_pent = np.rot90(rot_pent)
    return all_pents

def boardCoord2IDX(board, coord):
    return coord[0] * board.shape[1] + coord[1]

def generateMapping(board, all_pents):

    Y = {}

    for pent_idx, orientations in enumerate(all_pents):
        for pent_orientation_idx,pent in enumerate(orientations):
            for coord, val in np.ndenumerate(board):
                # try adding pent to board
                if add_pentomino(board, pent, coord):
                    # key = (pent_idx, pent_orientation_idx, coord) : value = [-pent_idx, covered coordinates...]
                    Y[(pent_idx, pent_orientation_idx, coord)] = [(pent_idx * -1) - 1] + [ boardCoord2IDX(board, c) for c in np.argwhere(board > 0)]
                    # remove pent from board
                    del_pentomino(board,pent,coord)

    return Y

def solve(board, pents):

    # reformat board so empty space = 0, blocked off = -1
    board = board.astype(int) - 1

    # generate all pents
    all_pents = generateAllPents(pents)

    # Y is subsets of numbers we want to chose to cover X
    Y = generateMapping(board, all_pents)

    # X is set of numbers we want to cover
    # [all pents used once ... , all coordinates used ... ]
    X = list(range(-len(pents),0)) + [ boardCoord2IDX(board,coord) for coord in np.argwhere(board==0)]

    _, sol = ALGOX(X,Y).solve()

    # select out correct pents for final answer
    final = [(all_pents[chosen[0]][chosen[1]], chosen[2]) for chosen in sol]

    # print final answer
    [add_pentomino(board, pent, coord) for (pent, coord) in final]
    print (board)   
    
    return final



    
