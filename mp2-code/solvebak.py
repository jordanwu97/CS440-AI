# -*- coding: utf-8 -*-
import numpy as np
import time
from os import system

def nphash(arr):
    return hash(str(arr))

def can_add_pent(board, pent, coord):
    if coord[0] < 0 or coord[1] < 0:
        return False
    if coord[0] + pent.shape[0] >= board.shape[0] or coord[1] + pent.shape[1] >= board.shape[1]:
        return False

    # boardslice = board[coord[0]:coord[0]+pent.shape[0], coord[1]:coord[1]+pent.shape[1]]
    # boardslice = np.multiply(boardslice, pent)
    # return not len(np.argwhere(boardslice!=0))  0
    
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

def del_pentomino(board, pent, coord):
    for row in range(pent.shape[0]):
        for col in range(pent.shape[1]):
            if pent[row][col] != 0:
                board[coord[0]+row][coord[1]+col] = 0

# def forward_check(board, pent):


## Variables: the pents
## Assignment: coordinate

## LRV: pent with least remaining coordinate that can be assigned
## MCV: 
## FowardCheck: coordinate can be assigned with variable

def forward_check(board, all_pents):

    zeroes = set(np.argwhere(board==0))

    for pentlist in all_pents:
        for coord in np.argwhere(board==0):
            can_add = False
            for pent in pentlist:
                if can_add_pent(board, pent, coord):
                    can_add = True
                    break
            if can_add:
                break
        if not can_add:
            print (coord)
            print (board)
            return False
    return True


def rec_solve(board, board_size, all_pents, solution):

    print (board)
    system("clear")

    zeroes = np.argwhere(board==0)
    if len(zeroes) == 0:
        print (board)
        return True

    top_left = zeroes[0]

    for pent_idx in range(len(all_pents)):
        pentlist = list(all_pents[pent_idx])
        # remove current pent
        del all_pents[pent_idx]
        for pent in pentlist:
            # try fitting pent to all possible positions
            coord = top_left
            for h in range(-pent.shape[0],1):
                for w in range(-pent.shape[1],1):
                    offset = np.array([h,w])
                    newcoord = coord + offset
                    if add_pentomino(board, pent, newcoord):
                        if rec_solve(board, board_size, all_pents, solution):
                            solution.append((pent,newcoord))
                            return True
                        del_pentomino(board, pent, newcoord)
        # re-add
        all_pents.insert(pent_idx, pentlist)
    
    return False
    

def solve(board, pents):
    """
    This is the function you will implement. It will take in a numpy array of the board
    as well as a list of n tiles in the form of numpy arrays. The solution returned
    is of the form [(p1, (row1, col1))...(pn,  (rown, coln))]
    where pi is a tile (may be rotated or flipped), and (rowi, coli) is 
    the coordinate of the upper left corner of pi in the board (lowest row and column index 
    that the tile covers).
    
    -Use np.flip and np.rot90 to manipulate pentominos.
    
    -You can assume there will always be a solution.
    """

    board = (board * -1) + 1

    board = board.astype(int)
    
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

    sol = []

    rec_solve(board, board.shape[0] * board.shape[1], all_pents, sol)

    return sol
