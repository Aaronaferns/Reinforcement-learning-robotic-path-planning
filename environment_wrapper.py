import numpy as np
import random
from environments.pygame.renderer import PygameRenderer

class TabularEnv:
    def __init__(self,
                grid_size,
                render = False,   
                start_pos=None,
                target_pos=None,
                random_radii = True,
                max_radius=1,
                num_obstacles = 4
                ):
        if start_pos==None:
            self.start_pos = (random.randint(0,grid_size-1),random.randint(0,grid_size-1))
        else: self.start_pos = start_pos
        if target_pos==None:
            self.target_pos = (random.randint(0,grid_size-1),random.randint(0,grid_size-1))
        else: self.target_pos = target_pos
        self.grid_size=grid_size
        self.render = render
        self.random_radii = random_radii
        self.max_radius = max_radius
        self.num_obstacles = num_obstacles
        self.env_grid = np.zeros((self.grid_size,self.grid_size))
        self.obstacles =[]
        self.create_obstacles()



        if render: 
            self.renderer = PygameRenderer(grid_size, self.env_grid)
        # actions = [0,1,2,3]




    def create_obstacles(self):
        fail_count = 0  # To avoid infinite loops
        max_attempts = self.num_obstacles * 10
        
        while len(self.obstacles) < self.num_obstacles and fail_count < max_attempts:
            radius = random.randint(1, self.max_radius)
            x = random.randint(radius, self.grid_size - radius - 1)
            y = random.randint(radius, self.grid_size - radius - 1)
            center = (x, y)
            if self.valid_center(center, radius):
                self.obstacles.append((center, radius))
            else:
                fail_count += 1

        if fail_count >= max_attempts:
            print(f"Could only place {len(self.obstacles)} out of {self.num_obstacles} obstacles.")
        self.fit_circles()
        
    
    # def valid_center(self,center,radius):  
    #     x, y = center
    #     if not (radius <= x < self.grid_size - radius and radius <= y < self.grid_size - radius):
    #         return False  
    #     for (o_center, o_radius) in self.obstacles:
    #         ox, oy = o_center
    #         if ((x - ox) ** 2 + (y - oy) ** 2) < (radius + o_radius + 1) ** 2:
    #             return False
    #     return True
    
    def valid_center(self, center, radius):
        x, y = center

        # Check if the obstacle is fully inside the grid
        if not (radius <= x < self.grid_size - radius and radius <= y < self.grid_size - radius):
            return False  

        # Check if the obstacle overlaps with the target position
        target_x, target_y = self.target_pos
        if ((x - target_x) ** 2 + (y - target_y) ** 2) <= radius ** 2:
            return False

        # Check for overlaps with other obstacles (with a gap)
        for (o_center, o_radius) in self.obstacles:
            ox, oy = o_center
            distance_squared = (x - ox) ** 2 + (y - oy) ** 2
            min_distance = radius + o_radius + 1  # 1 unit gap between obstacles
            if distance_squared < min_distance ** 2:  # Ensuring they don't overlap
                return False

        return True



    def fit_circles(self):
        for (center, radius) in self.obstacles:
            x, y = center
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if (i - y) ** 2 + (j - x) ** 2 <= radius ** 2:
                        self.env_grid[i][j] = 1



       
        
    def step(self,action):
        raise NotImplementedError

    
    def reset(self):
        raise NotImplementedError


    
    def close(self):
        pass
    



