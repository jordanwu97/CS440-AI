import snake_main
import time
from multiprocessing import Pool

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def run(v):
    start = time.time()
    args = Namespace(C=v, Ne=40, food_x=80, food_y=80, gamma=0.3, human=False, model_name='q_agent.npy', show_eps=0, snake_head_x=200, snake_head_y=200, test_eps=1000, train_eps=25000, window=10000)
    game = snake_main.Application(args)
    score = game.execute()
    end = time.time()
    return (v,score,end-start)

scorelist = [run(v) for v in range(40,200,40)]

print (scorelist)


'''
Gamma Scores:
[(0.1, 25.427, 79.8231291770935), 
(0.2, 25.16, 77.87106132507324), 
(0.3, 25.923, 81.52754235267639), 
(0.4, 23.43, 72.92591047286987), 
(0.5, 24.389, 85.8355062007904), 
(0.6, 23.437, 75.8369128704071), 
(0.7, 24.376, 74.57681655883789), 
(0.8, 22.92, 80.2971510887146), 
(0.9, 22.311, 77.00139999389648), 
(1.0, 14.206, 59.134013652801514)]

'''