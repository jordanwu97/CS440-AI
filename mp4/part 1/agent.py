import numpy as np
import utils
import random
import math


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        ## TEMP
        self.step = 1

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''

        def to12(x):
            return (x + 1) // 40

        # convert all down to 12x12
        curSnakeHead = to12(state[0]), to12(state[1])
        curSnakeBody = [(to12(x), to12(y)) for x,y in state[2]]
        curFood = to12(state[3]), to12(state[4])

        def retrieveQandN(snakeHead, snakeBody, food):

            snakeHeadX, snakeHeadY = snakeHead
            foodX, foodY = food

            def checkAdjoining(snake, v1, v2):
                if snake == v1:
                    return 1
                elif snake == v2:
                    return 2
                return 0

            # check for adjoining wall
            adjWallX, adjWallY = checkAdjoining(snakeHeadX,1,13),checkAdjoining(snakeHeadY,1,13)

            # check where food is
            def checkDir(a):
                if a > 0:
                    return 2
                elif a < 0:
                    return 1
                return 0
            foodDirX, foodDirY = checkDir(foodX - snakeHeadX), checkDir(foodY - snakeHeadY)
            
            # check if adjoining is snake body
            adjoiningBody = [int((snakeHeadX + offX,snakeHeadY + offY) in snakeBody) for offX, offY in ((0,-1),(0,1),(-1,0),(1,0))]

            # retrieve current Q and N
            tQ = self.Q[adjWallX, adjWallY, foodDirX, foodDirY, adjoiningBody[0], adjoiningBody[1], adjoiningBody[2], adjoiningBody[3]]
            tN = self.N[adjWallX, adjWallY, foodDirX, foodDirY, adjoiningBody[0], adjoiningBody[1], adjoiningBody[2], adjoiningBody[3]]

            return tQ, tN

        Q, N = retrieveQandN(curSnakeHead, set(curSnakeBody), curFood)

        # Get best action
        def explorationFunc(u,n):
            return 1 if n < self.Ne else u
        # Tiebreak action by right > left > down > up, so we need to reverse Q and N then do argmax,
        # then invert the resulting arg
        a = (np.argmax([explorationFunc(Q[a], N[a]) for a in (3,2,1,0)]) - 3) * -1

        # Update N visiting a
        N[a] += 1

        # Get next state s'
        movementList = [(0,-1),(0,1),(-1,0),(1,0)]
        nextSnakeHead = curSnakeHead[0] + movementList[a][0], curSnakeHead[1] + movementList[a][1]
        nextSnakeBody = list(curSnakeBody+[nextSnakeHead])
        del(nextSnakeBody[0])

        # Get Momentary Reward
        def reward(snakeHead, snakeBody, food):
            # hit itself
            if snakeHead in snakeBody:
                return -1
            # hit wall
            if snakeHead[0] == 0 or snakeHead[0] == 13 or snakeHead[1] == 0 or snakeHead[1] == 13:
                return -1
            # gets food
            if snakeHead == food:
                return 1
            return -0.1
        R = reward(nextSnakeHead, curSnakeBody, curFood)
        
        # Get Util of next state
        Qprime, _ = retrieveQandN(nextSnakeHead, set(nextSnakeBody), curFood)
        Uprime = max(Qprime)

        # Update
        Q[a] = Q[a] + (self.C/(self.C * N[a])) * (R + self.gamma * Uprime - Q[a])
        self.step += 1

        return a
