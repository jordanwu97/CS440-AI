from time import sleep
from math import inf
import random

customboard1 =     [['_','_','X','_','_','_','O','_','_'],
                    ['_','X','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','O','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]

customstart1 = 1

customboard2 =     [['_','_','X','_','_','_','O','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','X','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','O','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]

customstart2 = 0

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

        self.winnerDesignUtility=10000
        self.twoInARowDesignUtility=300
        self.preventThreeInARowDesignUtility=300
        self.cornerDesignUtility=30
		
        # set evaluation functions for min and max players
        self.maxEvalFunc = lambda: self.evaluatePredifined(True)
        self.minEvalFunc = lambda: self.evaluatePredifined(False)

        # some housekeeping for states explored + memoziation
        self.statesExplored = 0
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
        local = [row[c:c+3] for row in self.board[r:r+3]]
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

    def _designedUtility(self, arr, isMax):
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

        if isMax:
            playerSymbol = self.maxPlayer
            opponentSymbol = self.minPlayer
        else:
            playerSymbol = self.minPlayer
            opponentSymbol = self.maxPlayer

        val = 0

        # evaluate assuming were max player
        if arr == [opponentSymbol] * 3:
            val = -self.winnerDesignUtility
        if arr == [playerSymbol] * 3:
            val = self.winnerDesignUtility
        elif arr in twoAndOne(playerSymbol, self.empty):
            val = self.twoInARowDesignUtility
        elif arr in twoAndOne(opponentSymbol, playerSymbol):
            val = self.preventThreeInARowDesignUtility
        # flip value if were min player
        val = val if isMax else -val
        
        (self.utilityMemo)[stringify] = val

        return val

    def evaluateDesigned(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for your own agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """

        """
        Priority of our agent
        1) If we won, return the util for winning (10000 for isMax = True)
        2) If we lost, return the -util for losing (-10000 for isMax = True)
        3) Balanced scoring at 300 for each 2 in a row, and each prevent3inarow
        4) 
        """

        winnerUtil = self.winnerDesignUtility if isMax else -self.winnerDesignUtility
        loserUtil = -winnerUtil
        
        totalUtil = 0

        # Rule 1 and 2
        for localBoard in (self._copyLocalBoard(i) for i in range(len(self.globalIdx))):
            
            # Rows
            for i in range(3):
                u = self._designedUtility(localBoard[i], isMax)
                if u == winnerUtil:
                    return winnerUtil
                if u == loserUtil:
                    return loserUtil
                totalUtil += u
            
            # Columns
            for i in range(3):
                u = self._designedUtility([localBoard[r][i] for r in range(3)], isMax)
                if u == winnerUtil:
                    return winnerUtil
                if u == loserUtil:
                    return loserUtil
                totalUtil += u

            # Diagonal \
            u = self._designedUtility([ localBoard[i][i] for i in range(3)], isMax)
            if u == winnerUtil:
                return winnerUtil
            if u == loserUtil:
                return loserUtil
            totalUtil += u
            
            # Diagonal /
            u = self._designedUtility([ localBoard[i][2-i] for i in range(3)], isMax)
            if u == winnerUtil:
                return winnerUtil
            if u == loserUtil:
                return loserUtil
            totalUtil += u
            

        # Third rule
        if totalUtil == 0:
            symbol, cornerUtil = (self.maxPlayer, self.cornerMaxUtility) if isMax else (self.minPlayer, self.cornerMinUtility)
            for localBoard in (self._copyLocalBoard(i) for i in range(len(self.globalIdx))):
                for (r,c) in [(0,0),(0,2),(2,0),(2,2)]:
                    totalUtil += cornerUtil if localBoard[r][c] == symbol else 0

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
    
    def _makeMove(self, currBoardIdx, localRow, localCol, isMax, eraseMove=False):
        
        globalRow, globalCol = self.globalIdx[currBoardIdx]
        if eraseMove:
            self.board[globalRow + localRow][globalCol + localCol] = self.empty
            return True
        elif self.board[globalRow + localRow][globalCol + localCol] != self.empty:
            return False
        self.board[globalRow + localRow][globalCol + localCol] = self.maxPlayer if isMax else self.minPlayer
        return True
            
    def alphabeta(self,depth,currBoardIdx,alpha,beta,isMax, returnCord=False):
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
        # increment statesExplored
        self.statesExplored += 1
        
        # if at maxdepth
        # or no moves left to play (someone already won), keep going down tree
        if depth == self.maxDepth or self.checkWinner() != 0:
            # some jank here. 
            # need to invert the evaluation functions if depth is odd
            # because isMax will be false when it should be true
            if depth % 2 == 0:
                return self.maxEvalFunc() if isMax else self.minEvalFunc()
            else:
                return self.minEvalFunc() if isMax else self.maxEvalFunc()
        
        bestValue = -inf if isMax else inf
        coord = (0,0)

        # alphabeta search
        for r in range(3):
            for c in range(3):
                if self._makeMove(currBoardIdx, r, c, isMax):
                    if isMax:
                        bestValue, coord = max([(bestValue,coord), (self.alphabeta(depth + 1, r * 3 + c, alpha, beta, not isMax), (r,c))], key = lambda pair: pair[0])
                        if bestValue > beta:
                            self._makeMove(currBoardIdx, r, c, isMax, eraseMove=True)
                            return (bestValue, coord) if returnCord else bestValue
                        else:
                            alpha = bestValue
                    else:
                        bestValue, coord = min([(bestValue,coord), (self.alphabeta(depth + 1, r * 3 + c, alpha, beta, not isMax), (r,c))], key = lambda pair: pair[0])
                        if bestValue < alpha:
                            self._makeMove(currBoardIdx, r, c, isMax, eraseMove=True)
                            return (bestValue, coord) if returnCord else bestValue
                        else:
                            beta = bestValue
                    self._makeMove(currBoardIdx, r, c, isMax, eraseMove=True)

        return (bestValue, coord) if returnCord else bestValue

    def minimax(self, depth, currBoardIdx, isMax, returnCord=False):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        # increment statesExplored
        self.statesExplored += 1
        
        # if at maxdepth
        # or no moves left to play (someone already won), keep going down tree
        if depth == self.maxDepth or self.checkWinner() != 0:
            # some jank here. 
            # need to invert the evaluation functions if depth is odd
            # because isMax will be false when it should be true
            if depth % 2 == 0:
                return self.maxEvalFunc() if isMax else self.minEvalFunc()
            else:
                return self.minEvalFunc() if isMax else self.maxEvalFunc()
        
        bestValue = -inf if isMax else inf
        coord = (0,0)

        # play all valid moves on local board
        for r in range(3):
            for c in range(3):
                if self._makeMove(currBoardIdx, r, c, isMax):
                    if isMax:
                        
                        bestValue, coord = max([(bestValue,coord), (self.minimax(depth + 1, r * 3 + c, not isMax), (r,c))], key = lambda pair: pair[0])
                    else:
                        bestValue, coord = min([(bestValue,coord), (self.minimax(depth + 1, r * 3 + c, not isMax), (r,c))], key = lambda pair: pair[0])
                    
                    self._makeMove(currBoardIdx, r, c, isMax, eraseMove=True)

        return (bestValue, coord) if returnCord else bestValue

    def _playGameAgent(self, maxFirst, a1Search, a2Search):
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        a1Search(function): Search function for maxPlayer
        a2Search(function): Search function for minPlayer
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        # Init
        bestMove=[]
        bestValue=[]
        expandedNodes = []
        gameBoards=[]
        winner=0
        boardIdx = self.startBoardIdx

        # check if anyone won or no more moves
        while self.checkMovesLeft() and self.checkWinner() == 0:
            
            # search based on who is playing
            if maxFirst:
                val, (r,c) = a1Search(boardIdx)
            else:
                val, (r,c) = a2Search(boardIdx)
            
            # make move best on search
            self._makeMove(boardIdx, r, c, maxFirst)
            
            # update results list
            bestMove.append((r,c))
            bestValue.append(val)
            expandedNodes.append(self.statesExplored)
            self.statesExplored = 0 # reset statesExplored
            gameBoards.append(list(self.board))
    
            # update for next round
            boardIdx = r * 3 + c
            maxFirst = not maxFirst

            # print (val)
            # self.printGameBoard()

        return gameBoards, bestMove, expandedNodes, bestValue, self.checkWinner()

    def _printGameInfo(self, maxFirst):
        # print ("maxPlayer: ", self.maxPlayer)
        # print ("minPlayer: ", self.minPlayer)

        if maxFirst:
            print ("maxFirst")
        else:
            print ("minFirst")

    def playGamePredifinedAgent(self,maxFirst,isMinimaxOffensive,isMinimaxDefensive):
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for offensive agent.
                        True is minimax and False is alpha-beta.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for defensive agent.
                        True is minimax and False is alpha-beta.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        # set Search
        if isMinimaxOffensive:
            maxPlayerSearch = lambda boardIdx: self.minimax(0, boardIdx, True, returnCord=True)
        else:
            maxPlayerSearch = lambda boardIdx: self.alphabeta(0, boardIdx, -inf, inf, True, returnCord=True)
        
        if isMinimaxDefensive:
            minPlayerSearch = lambda boardIdx: self.minimax(0, boardIdx, False, returnCord=True)
        else:
            minPlayerSearch = lambda boardIdx: self.alphabeta(0, boardIdx, -inf, inf, False, returnCord=True)
        
        # set evaluation
        self.maxEvalFunc = lambda: self.evaluatePredifined(True)
        self.minEvalFunc = lambda: self.evaluatePredifined(False)

        # replace symbols based on who goes first
        if not maxFirst:
            self.minPlayer = 'X'
            self.maxPlayer = 'O'
        
        self._printGameInfo(maxFirst)

        return self._playGameAgent(maxFirst, maxPlayerSearch, minPlayerSearch)

    def playGameYourAgent(self):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        maxFirst = random.choice([True, False])
        self.startBoardIdx=random.randint(0,8)

        # maxPlayer keeps same strategy
        maxPlayerSearch = lambda boardIdx: self.alphabeta(0, boardIdx, -1000000, 1000000, True, returnCord=True)
        # we replace minPlayer with our player
        minPlayerSearch = lambda boardIdx: self.alphabeta(0, boardIdx, -1000000, 1000000, False, returnCord=True)
        
        # set evaluation
        self.maxEvalFunc = lambda: self.evaluatePredifined(True)
        self.minEvalFunc = lambda: self.evaluateDesigned(False)

        # replace symbols based on who goes first
        if not maxFirst:
            self.minPlayer = 'X'
            self.maxPlayer = 'O' 
        
        self._printGameInfo(maxFirst)

        gameBoards,bestMove, expandedNodes, bestValue, winner = self._playGameAgent(maxFirst, maxPlayerSearch, minPlayerSearch)
        return gameBoards, bestMove, winner

    def playGameHuman(self):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        def human(*arg, **kwargs):
            self.printGameBoard()
            a = None
            while a == None or len(a) != 2:
                a = [int(b) for b in input("your turn: (x,y) ").split(",")]
            return 0, (a[0],a[1])

        maxPlayerSearch = lambda boardIdx: human()
        minPlayerSearch = lambda boardIdx: self.alphabeta(0, boardIdx, -1000000, 1000000, False, returnCord=True)

        # set evaluation
        self.maxEvalFunc = lambda: self.evaluatePredifined(True)
        self.minEvalFunc = lambda: self.evaluateDesigned(False)

        gameBoards,bestMove, expandedNodes, bestValue, winner = self._playGameAgent(True, maxPlayerSearch, minPlayerSearch)

        return gameBoards, bestMove, winner

def playPredefinedAgents():
    uttt=ultimateTicTacToe()
    gameBoards,bestMove, expandedNodes, bestValue, winner = uttt.playGamePredifinedAgent(False, False, False)

    uttt.printGameBoard()
    print ("bestMove:", bestMove)
    print ("expandedNodes:", expandedNodes)
    print ("bestValues:", bestValue)
    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")

# def playMyAgentAllRounds():
#     myAgentScore = 0
#     maxAgentScore = 0
#     for maxFirst in (True, False):
#         for i in range(0,9):
#                 uttt=ultimateTicTacToe()
#                 uttt.startBoardIdx = i
#                 print ("startBoardIdx:", uttt.startBoardIdx)
#                 gameBoards,bestMove, winner=uttt.playGameYourAgent(maxFirst)
#                 # print (bestMove)
#                 print ("winner", "minAgent" if winner == -1 else "maxAgent")
#                 print ()
#                 if winner == 1:
#                     maxAgentScore += 1
#                 elif winner == -1:
#                     myAgentScore += 1
#                 else:
#                     print("Tie. No winner:(")
#     print ("myAgentScore:", myAgentScore)
#     print ("maxAgentScore:", maxAgentScore)

def playMyAgent1Round():
    
    uttt=ultimateTicTacToe()
    gameBoards,bestMove, winner=uttt.playGameYourAgent()
    uttt.printGameBoard()
    print ("startBoardIdx:", uttt.startBoardIdx)
    print ("winner", "minAgent" if winner == -1 else "maxAgent")



def playHuman():
    uttt=ultimateTicTacToe()
    gameBoards,bestMove, winner = uttt.playGameHuman()

if __name__=="__main__":
    # playPredefinedAgents()
    playMyAgent1Round()
    # playHuman()