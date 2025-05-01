from environment_wrapper import TabularEnv
import random
import numpy as np
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
    
    def vision_access(self, center, heading_dir, angle_deg, vision):
        x0, y0 = center
        grid_h, grid_w = self.env_grid.shape
        
        # Calculate view window bounds
        x_min = max(x0 - vision, 0)
        x_max = min(x0 + vision + 1, grid_h)
        y_min = max(y0 - vision, 0)
        y_max = min(y0 + vision + 1, grid_w)
        
        view_area = self.env_grid[x_min:x_max, y_min:y_max]
        
        # Create meshgrid relative to center
        xv, yv = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')
        dx = xv - x0
        dy = yv - y0
        
        # Convert to polar coordinates
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        theta = np.mod(theta, 2 * np.pi)

        # Determine heading angle in radians
        heading = np.deg2rad(heading_dir * 36)  # 10 possible directions (0 to 324)
        half_angle = np.deg2rad(angle_deg / 2)
        angle_start = np.mod(heading - half_angle, 2 * np.pi)
        angle_end = np.mod(heading + half_angle, 2 * np.pi)

        # Create sector mask
        if angle_start < angle_end:
            mask = (r <= vision) & (theta >= angle_start) & (theta <= angle_end)
        else:
            mask = (r <= vision) & ((theta >= angle_start) | (theta <= angle_end))

        return mask.astype(int)

    
    def step(self,state,action):
            # # print(action)
            # self.step_count+=1
            
            # state_=state
            # i,j=state
            # if action == 0: 
            #     if i!=0:state_ = (i-1,j)
            # elif action == 1:
            #     if j!=self.grid_size-1: state_ = (i,j+1)
            # elif action == 2: 
            #     if i!=self.grid_size-1: state_ = (i+1,j)
            # elif action == 3: 
            #     if j!=0: state_ = (i,j-1)
            # truncated,terminated =self.isTruncated(state_),self.isTerminated(state_)
            # reward = self.reward(state_)
            # if self.render:
            #     self.renderer.render(state_,self.target_pos)
            #     if truncated or terminated: self.renderer.close()
            # return state_, reward, truncated,terminated
    
        
            self.step_count += 1
            
            # Reset state and position based on action
            state_ = state
            i, j = state
            rr =0
            if action == 0:
                if i != 0: state_ = (i - 1, j)
            elif action == 1:
                if j != self.grid_size - 1: state_ = (i, j + 1)
            elif action == 2:
                if i != self.grid_size - 1: state_ = (i + 1, j)
            elif action == 3:
                if j != 0: state_ = (i, j - 1)
            if self.env_grid[state_]==1:
                state_=state
                rr-=0.01
            truncated, terminated = self.isTruncated(state_), self.isTerminated(state_)
            reward = self.reward(state_)+rr
            
            if self.render and hasattr(self, 'renderer') and self.renderer is not None:
                if not self.renderer.initialized:  # Check if renderer has been initialized
                    self.renderer.initialize_window()
                self.renderer.render(state_, self.target_pos)
                if truncated or terminated:
                    self.renderer.close()
            
            return state_, reward, truncated, terminated


    def reward(self,state):
        if state==self.target_pos: return 1
        return 0
    def isTruncated(self,state):
        if  self.step_count==self.max_steps  : return True
        return False
    def isTerminated(self,state):
        if state==self.target_pos:
            return True
        return False

        
    def reset(self):
        # self.step_count=0
        # if self.rand:
        #     self.start_pos = (random.randint(0,self.grid_size-1),random.randint(0,self.grid_size-1))
        # if self.render:
        #     self.renderer.initialize_window()
        #     self.renderer.render(self.start_pos,self.target_pos)
        #     if self.isTerminated(self.start_pos): self.renderer.close()
        # return self.start_pos

        self.step_count = 0
        if self.rand:
            while(True):
                self.start_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
                if self.env_grid[self.start_pos]==0:break
        if self.render:
            if not self.renderer.initialized:  # Check if the window is already initialized
                self.renderer.initialize_window()  # Initialize if necessary
            self.renderer.render(self.start_pos, self.target_pos)
            if self.isTerminated(self.start_pos):
                self.renderer.close()
        return self.start_pos

if __name__ == "__main__":
    env = tabenv1(100,200,(5,5))
    print(env.vision_access((40,40),1,30,10))

        
