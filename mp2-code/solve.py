import numpy as np
import time
import algox

def nphash(arr):
    return hash(str(arr))

def can_add_pent(board, pent, coord):
    if coord[0] < 0 or coord[1] < 0:
        return False
    if coord[0] + pent.shape[0] > board.shape[0] or coord[1] + pent.shape[1] > board.shape[1]:
        return False

    for row in range(pent.shape[0]):
        for col in range(pent.shape[1]):
            if pent[row][col] != 0:
                if board[coord[0]+row][coord[1]+col] != 0:
                    return False
    return True

def add_pentomino(board, pent, coord):
    """
    Adds a pentomino pent to the board. The pentomino will be placed such that
    coord[0] is the lowest row index of the pent and coord[1] is the lowest 
    column index. 
    
    check_pent will also check if the pentomino is part of the valid pentominos.
    """
    # check for overlap
    if not can_add_pent(board,pent,coord):
        return False
    board[coord[0]:coord[0]+pent.shape[0], coord[1]:coord[1]+pent.shape[1]] += pent
    return True

def generateAllPents(pents):
    all_pents = [[] for i in range(len(pents))]
    for i, pent in enumerate(pents):
        rot_pent = pent
        no_repeat = set()
        for rot in range(4):
            flip_pent = rot_pent
            for flip in range(2):
                # check for identical pent
                # print (flip_pent, nphash(flip_pent))
                if nphash(flip_pent) not in no_repeat:
                    all_pents[i].append(flip_pent)
                    no_repeat.add(nphash(flip_pent))
                
                flip_pent = np.fliplr(flip_pent)
            
            rot_pent = np.rot90(rot_pent)

    return all_pents

def boardCoord2IDX(board, coord):
    return coord[0] * board.shape[1] + coord[1]

def generateMapping(board, all_pents):

    choiceMapping = {}
    Y = {}

    choice = 0
    for pent_idx, pentlist in enumerate(all_pents):
        for pent_orientation,pent in enumerate(pentlist):
            for coord, val in np.ndenumerate(board):
                board_cpy = board.copy()
                if add_pentomino(board_cpy, pent, coord):
                    Y[choice] = [(pent_idx * -1) - 1] + [ boardCoord2IDX(board, c) for c in np.argwhere(board_cpy > 0)]
                    choiceMapping[choice] = (pent_idx, pent_orientation, (coord))
                    choice += 1

    return Y, choiceMapping
                    



def solve(board, pents):
    all_pents = generateAllPents(pents)

    board = board.astype(int)
    board = (board * -1) + 1

    Y, choiceMapping = generateMapping(board, all_pents)

    # print (Y)

    X = list(range(-len(pents),0)) + [ boardCoord2IDX(board,coord) for coord in np.argwhere(board==0)]

    X = algox.preprocess(X,Y)

    sol = algox.solve(X,Y)[1]

    chosen_pents = [choiceMapping[choice] for choice in sol]

    final = [(all_pents[chosen[0]][chosen[1]], chosen[2])for chosen in chosen_pents]

    print (final)
    
    return final



    
