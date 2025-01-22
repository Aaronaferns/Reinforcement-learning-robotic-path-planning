import numpy as np
import random
from environments.pygame.renderer import PygameRenderer

class TabularEnv:
    def __init__(self,grid_size,render = False, start_pos=None,target_pos=None):
        if start_pos==None:
            self.start_pos = (random.randint(0,grid_size-1),random.randint(0,grid_size-1))
        else: self.start_pos = start_pos
        if target_pos==None:
            self.target_pos = (random.randint(0,grid_size-1),random.randint(0,grid_size-1))
        else: self.target_pos = target_pos
        self.grid_size=grid_size
        self.render = render
        if render: 
            self.renderer = PygameRenderer(grid_size)
        # actions = [0,1,2,3]


        
        def step(self,action):
            raise NotImplementedError

        
        def reset(self):
            raise NotImplementedError


        
        def close(self):
            pass
        



