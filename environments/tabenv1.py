from environment_wrapper import TabularEnv
import random

class tabenv1(TabularEnv):
    def __init__(self,grid_size,max_steps=200,start_pos=None,target_pos=None,render=True):
        super(tabenv1,self).__init__(grid_size,render,start_pos,target_pos)
        self.nA = 4
        self.nS = 2
        self.step_count = 0
        self.max_steps=max_steps
        self.rand = start_pos==None
    
    def turnOffRender(self):
        self.render = False
    def turnONRender(self):
        self.render = True
        
        
    
    def step(self,state,action):
            # print(action)
            self.step_count+=1
            
            state_=state
            i,j=state
            if action == 0: 
                if i!=0:state_ = (i-1,j)
            elif action == 1:
                if j!=self.grid_size-1: state_ = (i,j+1)
            elif action == 2: 
                if i!=self.grid_size-1: state_ = (i+1,j)
            elif action == 3: 
                if j!=0: state_ = (i,j-1)
            truncated,terminated =self.isTruncated(state_),self.isTerminated(state_)
            reward = self.reward(state_)
            if self.render:
                self.renderer.render(state_,self.target_pos)
                if truncated or terminated: self.renderer.close()
            return state_, reward, truncated,terminated

    def reward(self,state):
        if state==self.target_pos: return 1
        return 0
    def isTruncated(self,state):
        if self.step_count==self.max_steps : return True
        return False
    def isTerminated(self,state):
        if state==self.target_pos:
            return True
        return False

        
    def reset(self):
        self.step_count=0
        if self.rand:
            self.start_pos = (random.randint(0,self.grid_size-1),random.randint(0,self.grid_size-1))
        if self.render:
            self.renderer.initialize_window()
            self.renderer.render(self.start_pos,self.target_pos)
            if self.isTerminated(self.start_pos): self.renderer.close()
        return self.start_pos

        


