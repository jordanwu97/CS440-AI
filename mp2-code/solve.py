# -*- coding: utf-8 -*-
import numpy as np
import time

def nphash(arr):
    return hash(str(arr))

def add_pentomino(board, pent, coord, check_pent=False, valid_pents=None):
    """
    Adds a pentomino pent to the board. The pentomino will be placed such that
    coord[0] is the lowest row index of the pent and coord[1] is the lowest 
    column index. 
    
    check_pent will also check if the pentomino is part of the valid pentominos.
    """
    if check_pent and not is_pentomino(pent, valid_pents):
        return False
    # check for overlap
    try:
        for row in range(pent.shape[0]):
            for col in range(pent.shape[1]):
                if pent[row][col] != 0:
                    if board[coord[0]+row][coord[1]+col] != 0:
                        return False
    except IndexError:
        return False


    for row in range(pent.shape[0]):
        for col in range(pent.shape[1]):
            if pent[row][col] != 0:
                board[coord[0]+row][coord[1]+col] = pent[row][col]
    return True

def del_pentomino(board, pent, coord):
    for row in range(pent.shape[0]):
        for col in range(pent.shape[1]):
            if pent[row][col] != 0:
                board[coord[0]+row][coord[1]+col] = 0

def rec_solve(board, board_size, all_pents, depth):

    time.sleep(0.5)
    print (board)

    zeroes = np.argwhere(board==0)
    
    if len(zeroes) == 0:
        return True

    top_left = zeroes[0]

    tried = 0

    for pent_idx in range(len(all_pents)):
        pentlist = list(all_pents[pent_idx])
        # remove current pent
        del all_pents[pent_idx]
        for pent in pentlist:
            if add_pentomino(board, pent, top_left):
                if rec_solve(board, board_size, all_pents, depth+1):
                    return True
                del_pentomino(board, pent, top_left)
            
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
    
    all_pents = [[] for i in range(12)]

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

    rec_solve(board, board.shape[0] * board.shape[1], all_pents, 1)

    return []
