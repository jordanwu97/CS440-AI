from time import sleep
from math import inf
from random import randint

class ultimateTicTacToe:
    def __init__(self):
        """
        Initialization of the game.
        """
        self.board=[['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]
        self.empty = '_'
        self.maxPlayer='X'
        self.minPlayer='O'
        self.maxDepth=3
        #The start indexes of each local board
        self.globalIdx=[(0,0),(0,3),(0,6),(3,0),(3,3),(3,6),(6,0),(6,3),(6,6)]

        #Start local board index for reflex agent playing
        self.startBoardIdx=4
        #self.startBoardIdx=randint(0,8)

        #utility value for reflex offensive and reflex defensive agents
        self.winnerMaxUtility=10000
        self.twoInARowMaxUtility=500
        self.preventThreeInARowMaxUtility=100
        self.cornerMaxUtility=30

        self.winnerMinUtility=-10000
        self.twoInARowMinUtility=-100
        self.preventThreeInARowMinUtility=-500
        self.cornerMinUtility=-30
		
        self.count = 0

        self.utilityMemo={}

    def printGameBoard(self):
        """
        This function prints the current game board.
        """
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[:3]]))
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[3:6]]))
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[6:9]]))
        print()

    def _utility(self, arr, isMax):
        """
        This function checks the status of an array arr
        Return:
            1: max wins
            2: max has unblocked 2 in a row
            3: max prevents 2 in a row
            neg values of min
        """

        stringify = "".join(arr)+str(isMax)

        if stringify in (self.utilityMemo).keys():
            return (self.utilityMemo)[stringify]

        def twoAndOne(two, one):
            max2Unblocked = []
            for i in range(3):
                x = [two] * 2
                x.insert(i, one)
                max2Unblocked.append(x)
            return max2Unblocked

        val = 0

        if isMax:
            if arr == [self.maxPlayer] * 3:
                val = self.winnerMaxUtility
            elif arr in twoAndOne(self.maxPlayer, self.empty):
                val = self.twoInARowMaxUtility
            elif arr in twoAndOne(self.minPlayer, self.maxPlayer):
                val = self.preventThreeInARowMaxUtility

        else:
            if arr == [self.minPlayer] * 3:
                val = self.winnerMinUtility
            elif arr in twoAndOne(self.minPlayer, self.empty):
                val = self.twoInARowMinUtility
            elif arr in twoAndOne(self.maxPlayer, self.minPlayer):
                val = self.preventThreeInARowMinUtility
        
        (self.utilityMemo)[stringify] = val

        return val

            
    def _copyLocalBoard(self,boardIdx):
        """
        This function returns a copy of local board of boardIdx for evaluation
        """
        r,c = self.globalIdx[boardIdx]
        local = [row[c:c+3] for row in uttt.board[r:r+3]]
        return local


    def evaluatePredifined(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predifined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """

        winnerUtil = self.winnerMaxUtility if isMax else self.winnerMinUtility
        totalUtil = 0

        # Rule 1 and 2
        for localBoard in (self._copyLocalBoard(i) for i in range(len(self.globalIdx))):
            
            # Rows
            for i in range(3):
                u = self._utility(localBoard[i], isMax)
                if u == winnerUtil:
                    return winnerUtil
                totalUtil += u
            
            # Columns
            for i in range(3):
                u = self._utility([localBoard[r][i] for r in range(3)], isMax)
                if u == winnerUtil:
                    return winnerUtil
                totalUtil += u

            # Diagonal \
            u = self._utility([ localBoard[i][i] for i in range(3)], isMax)
            if u == winnerUtil:
                return winnerUtil
            totalUtil += u
            
            # Diagonal /
            u = self._utility([ localBoard[i][2-i] for i in range(3)], isMax)
            if u == winnerUtil:
                return winnerUtil
            totalUtil += u
            

        # Third rule
        if totalUtil == 0:
            symbol, cornerUtil = (self.maxPlayer, self.cornerMaxUtility) if isMax else (self.minPlayer, self.cornerMinUtility)
            for localBoard in (self._copyLocalBoard(i) for i in range(len(self.globalIdx))):
                for (r,c) in [(0,0),(0,2),(2,0),(2,2)]:
                    totalUtil += cornerUtil if localBoard[r][c] == symbol else 0

        return totalUtil


    def evaluateDesigned(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for your own agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        winnerUtil = self.winnerMaxUtility if isMax else self.winnerMinUtility
        totalUtil = 0

        # Rule 1 and 2
        for localBoard in (self._copyLocalBoard(i) for i in range(len(self.globalIdx))):
            
            # Rows
            for i in range(3):
                u = self._utility(localBoard[i], isMax)
                if u == winnerUtil:
                    return winnerUtil
                totalUtil += u
            
            # Columns
            for i in range(3):
                u = self._utility([localBoard[r][i] for r in range(3)], isMax)
                if u == winnerUtil:
                    return winnerUtil
                totalUtil += u

            # Diagonal \
            u = self._utility([ localBoard[i][i] for i in range(3)], isMax)
            if u == winnerUtil:
                return winnerUtil
            totalUtil += u
            
            # Diagonal /
            u = self._utility([ localBoard[i][2-i] for i in range(3)], isMax)
            if u == winnerUtil:
                return winnerUtil
            totalUtil += u

            if isMax not in localBoard:
                if isMax:
                    totalUtil -= 100
                else:
                    totalUtil += 100
        # Third rule
        if totalUtil == 0:
            symbol, cornerUtil = (self.maxPlayer, self.cornerMaxUtility) if isMax else (self.minPlayer, self.cornerMinUtility)
            for localBoard in (self._copyLocalBoard(i) for i in range(len(self.globalIdx))):
                for (r,c) in [(0,0),(0,2),(2,0),(2,2)]:
                    totalUtil += cornerUtil if localBoard[r][c] == symbol else 0
        if totalUtil == 0:
            symbol, cornerUtil = (self.maxPlayer, self.cornerMaxUtility) if isMax else (self.minPlayer, self.cornerMinUtility)
            for localBoard in (self._copyLocalBoard(i) for i in range(len(self.globalIdx))):
                totalUtil -= cornerUtil if localBoard[1][1] == symbol else 0

        return totalUtil

    def checkMovesLeft(self):
        """
        This function checks whether any legal move remains on the board.
        output:
        movesLeft(bool): boolean variable indicates whether any legal move remains
                        on the board.
        """
        #YOUR CODE HERE
        for row in self.board:
            for v in row:
                if v == self.empty:
                    return True

        return False

    def checkWinner(self):
        #Return termimnal node status for maximizer player 1-win,0-tie,-1-lose
        """
        This function checks whether there is a winner on the board.
        output:
        winner(int): Return 0 if there is no winner.
                     Return 1 if maxPlayer is the winner.
                     Return -1 if miniPlayer is the winner.
        """
        
        # check for both players
        for isMax in [True, False]:

            v = self.evaluatePredifined(isMax)
            if isMax and v == self.winnerMaxUtility:
                return 1
            elif not isMax and v == self.winnerMinUtility:
                return -1

        return 0
    
    def makeMove(self, currBoardIdx, localRow, localCol, isMax, eraseMove=False):
        
        globalRow, globalCol = uttt.globalIdx[currBoardIdx]
        if eraseMove:
            self.board[globalRow + localRow][globalCol + localCol] = self.empty
            return True
        elif self.board[globalRow + localRow][globalCol + localCol] != self.empty:
            return False
        self.board[globalRow + localRow][globalCol + localCol] = self.maxPlayer if isMax else self.minPlayer
        return True
            

    def alphabeta(self,depth,currBoardIdx, isMax, actualPlayer, usePredefined, alpha=-10000, beta=10000):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        bestValue=0.0
        if depth == 0:
            self.count = self.count + 1
            if usePredefined:
                return self.evaluatePredifined(isMax), None
            else:
                return self.evaluateDesigned(isMax), None
        
        value = -100000000 if isMax else 100000000
        coord = (0,0)
        for r in range(3):
            for c in range(3):
                if self.makeMove(currBoardIdx, r, c, isMax):
                    if isMax:
                        value, coord = max([(value,coord), (self.alphabeta(depth - 1, r * 3 + c,not isMax,usePredefined,alpha,beta)[0], (r,c))], key = lambda pair: pair[0])
                        if value > beta:
                            self.count = self.count + 1
                            self.makeMove(currBoardIdx, r, c, isMax, eraseMove=True)
                            return value, coord
                        else:
                            alpha = value
                    else:
                        value, coord = min([(value,coord), (self.alphabeta(depth - 1, r * 3 + c,not isMax,usePredefined,alpha,beta)[0], (r,c))], key = lambda pair: pair[0])
                        if value < alpha:
                            self.count = self.count + 1
                            self.makeMove(currBoardIdx, r, c, isMax, eraseMove=True)
                            return value, coord
                        else:
                            beta = value
                    self.makeMove(currBoardIdx, r, c, isMax, eraseMove=True)
        self.count = self.count + 1
        return value, coord

    def minimax(self, depth, currBoardIdx, isMax, actualPlayer, usePredefined):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        
        # increment statesExplored
        self.statesExplored = self.statesExplored + 1
        
        # if at level 3 or no moves left to play (someone already won)
        if depth == 0 or self.checkWinner() != 0:
            if usePredefined:
                return self.evaluatePredifined(actualPlayer), None
            else:
                return self.evaluateDesigned(actualPlayer), None
        value = -100000000 if isMax else 100000000
        coord = (0,0)

        # play all valid moves on local board
        for r in range(3):
            for c in range(3):
                if self.makeMove(currBoardIdx, r, c, isMax):
                    if isMax:
                        value, coord = max([(value,coord), (self.minimax(depth - 1, r * 3 + c, not isMax, actualPlayer, usePredefined)[0], (r,c))], key = lambda pair: pair[0])
                    else:
                        value, coord = min([(value,coord), (self.minimax(depth - 1, r * 3 + c, not isMax, actualPlayer, usePredefined)[0], (r,c))], key = lambda pair: pair[0])
                    self.makeMove(currBoardIdx, r, c, isMax, eraseMove=True)

        return value, coord

    def playGamePredifinedAgent(self,maxFirst,isMinimax, maxMethod=None, minMethod=None):
        if maxMethod == None:
            maxMethod = self.minimax
        if minMethod == None:
            minMethod = self.minimax
            
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        isMinimax(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm.
                        True is minimax and False is alpha-beta.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        bestValue=[]
        expandedNodes = []
        gameBoards=[]
        winner=0
        boardIdx = self.startBoardIdx
        curPlayerIsMax = maxFirst
        while self.checkMovesLeft():
            # minVal, (r,c) = self.minimax(2, boardIdx, curPlayerIsMax,True) if isMinimax else self.alphabeta(2, boardIdx,-10000,10000, curPlayerIsMax,True) 
            if curPlayerIsMax:
                minVal, (r,c) = maxMethod(3, boardIdx, curPlayerIsMax, curPlayerIsMax, True)
            else:
                minVal, (r,c) = minMethod(3, boardIdx, curPlayerIsMax, curPlayerIsMax, True)

            print("minimax") if isMinimax else print("alphabeta")
            print("minVal", minVal)
            bestMove.append((r,c))
            bestValue.append(minVal)
            expandedNodes.append(self.statesExplored)
            self.statesExplored = 0
            self.makeMove(boardIdx, r, c, curPlayerIsMax)
            boardIdx = r * 3 + c
            curPlayerIsMax = not curPlayerIsMax
            self.printGameBoard()
            winner = self.checkWinner()
            print (winner)
            if winner != 0:
                break

        return gameBoards, bestMove, expandedNodes, bestValue, winner

    def playGameYourAgent(self,maxFirst,isMinimax):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]

        gameBoards=[]
        winner=0
        boardIdx = self.startBoardIdx
        curPlayerIsMax = maxFirst
        while self.checkMovesLeft():
            if curPlayerIsMax:
                minVal, (r,c) = self.minimax(2, boardIdx, curPlayerIsMax,False) if isMinimax else self.alphabeta(2, boardIdx,-1000000,1000000, curPlayerIsMax,False) 
                self.count = 0
                self.makeMove(boardIdx, r, c, curPlayerIsMax)
            else:    
                minVal, (r,c) = self.minimax(2, boardIdx, curPlayerIsMax,True) if isMinimax else self.alphabeta(2, boardIdx,-1000000,1000000, curPlayerIsMax,True) 
                self.count = 0
                self.makeMove(boardIdx, r, c, curPlayerIsMax)
            bestMove.append((r,c))
            boardIdx = r * 3 + c
            curPlayerIsMax = not curPlayerIsMax
            self.printGameBoard()
            winner = self.checkWinner()
            print (winner)
            if winner != 0:
                break

        return gameBoards, bestMove, winner


    def playGameHuman(self,maxFirst,isMinimax):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]

        gameBoards=[]
        winner=0
        boardIdx = self.startBoardIdx
        curPlayerIsMax = maxFirst
        while self.checkMovesLeft():
            if curPlayerIsMax:
                print("currBoardIdx:",boardIdx)
                self.printGameBoard()
                ans = input("your turn:(x,y)") #your input should be in the form  of "(r,c)"
                (r,c) = (int(ans[1]),int(ans[3]))
                
                if not self.makeMove(boardIdx, r,c, curPlayerIsMax):
                    print("invalid movement")
                    continue
            else:    
                minVal, (r,c) = self.minimax(3, boardIdx, curPlayerIsMax,False) if isMinimax else self.alphabeta(3, boardIdx,-1000000,1000000, curPlayerIsMax,False) 
                self.count = 0
                self.makeMove(boardIdx, r, c, curPlayerIsMax)
            bestMove.append((r,c))
            boardIdx = r * 3 + c
            curPlayerIsMax = not curPlayerIsMax
            self.printGameBoard()
            winner = self.checkWinner()
            print (winner)
            if winner != 0:
                break

        return gameBoards, bestMove, winner

import numpy as np

if __name__=="__main__":
    uttt=ultimateTicTacToe()
    #print (uttt.minimax(2, 4, True))

    #print (max((100, (1,2)), (2, (0,0)) ))
    #uttt.playGamePredifinedAgent(True, True)

    gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(True,False)
    # gameBoards,bestMove, winner=uttt.playGameYourAgent(False,True)
    #gameBoards,bestMove, winner=uttt.playGameHuman(True,True)
    
    uttt.printGameBoard()
    print("bestMove:",bestMove)
    #print("bestValue:",bestValue)
    #print("expandedNodes:",expandedNodes)
    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")