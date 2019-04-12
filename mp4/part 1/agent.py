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

        # Takes longform state and returns discretized state
        def discretizeState(state):

            def to12(x):
                return (x + 1) // 40

            # convert all down to 12x12
            snakeHead = to12(state[0]), to12(state[1])
            snakeBody = [(to12(x), to12(y)) for x,y in state[2]]
            food = to12(state[3]), to12(state[4])

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

            # return state
            return adjWallX, adjWallY, foodDirX, foodDirY, adjoiningBody[0], adjoiningBody[1], adjoiningBody[2], adjoiningBody[3]

        # Takes a discretized state tuple and retrieves Q and N as slice
        def retrieveQandN(discretizedState):

            adjWallX, adjWallY, foodDirX, foodDirY, adjoiningBody0, adjoiningBody1, adjoiningBody2, adjoiningBody3 = discretizedState

            # retrieve current Q and N for given state
            Qs = self.Q[adjWallX, adjWallY, foodDirX, foodDirY, adjoiningBody0, adjoiningBody1, adjoiningBody2, adjoiningBody3]
            Ns = self.N[adjWallX, adjWallY, foodDirX, foodDirY, adjoiningBody0, adjoiningBody1, adjoiningBody2, adjoiningBody3]

            return Qs, Ns

        s_prime = discretizeState(state)
        Q_s_prime, N_s_prime = retrieveQandN(s_prime)

        ### STEP 1, Update Q table using prev state and current state, prev action
        ### SKIP IF NO PREVIOUS STATE
        if hasattr(self, "s") and self.s != None:
            # Update Q table using prev state and current state, prev action
            Q_s, N_s = retrieveQandN(self.s)
            # Calculate Reward using current point and previous point, and dead variable
            def Reward(points_prime, points, dead):
                # died
                if dead:
                    return -1
                # got food since points increased
                if points_prime > points:
                    return 1
                return -0.1

            # Update Q table
            Q_s[self.a] = Q_s[self.a] + (self.C/(self.C * N_s[self.a])) * (Reward(points,self.points, dead) + self.gamma * max(Q_s_prime) - Q_s[self.a])


        ### Step 2, choose best action for current state
        def explorationFunc(u,n):
            return 1 if n < self.Ne else u
        # Tiebreak action by right > left > down > up, so we need to reverse Q and N then do argmax,
        # then invert the resulting arg
        a_prime = (np.argmax([explorationFunc(Q_s_prime[a], N_s_prime[a]) for a in (3,2,1,0)]) - 3) * -1


        ### STEP 3, Update N table, update class variables
        N_s_prime[a_prime] += 1
        if dead:
            self.reset()
        else:
            self.s = s_prime
            self.a = a_prime
            self.points = points

        return a_prime
