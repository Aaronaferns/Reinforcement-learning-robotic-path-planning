import pygame
import sys
import time 

class PygameRenderer:
    def __init__(self, grid_size,env_grid, cell_size=50):
        """
        Initialize the Pygame rendering environment.

        :param grid_size: The size of the grid (N x N).
        :param cell_size: The size of each cell in pixels.
        """
        self.grid_size = grid_size
        self.env_grid = env_grid
        self.cell_size = cell_size
        self.window_size = self.grid_size * self.cell_size
        self.screen = None
        self.agent_color = (0, 0, 255)  # Blue
        self.target_color = (255, 0, 0)  # Red
        self.background_color = (255, 255, 255)  # White
        self.grid_color = (200, 200, 200)  # Light Gray
        self.obstacle_color = (225,225,0)
        self.initialized = False

    def initialize_window(self):
        self.initialized=True
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Tabular Environment")

    def draw_grid(self):
        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, self.grid_color, (0, y), (self.window_size, y))
    
    def draw_obstacles(self):
        """
        Draw obstacles on the grid based on the env_grid.
        Obstacles are represented as filled circles.
        """
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.env_grid[i, j] == 1:  # Obstacle present
                    # Calculate the center of the grid cell in pixel coordinates
                    center_pixel = (j * self.cell_size + self.cell_size // 2, 
                                    i * self.cell_size + self.cell_size // 2)
                    radius_pixel = self.cell_size // 3  # Fixed radius relative to cell size
                    pygame.draw.circle(self.screen, self.obstacle_color, center_pixel, radius_pixel)

    def render(self, agent_pos, target_pos):
        """
        Render the grid with the agent and target.

        :param agent_pos: Tuple (i, j) for the agent's current position.
        :param target_pos: Tuple (i, j) for the target's position.

        
        """

        if not self.initialized:
            return
        self.screen.fill(self.background_color)
        self.draw_grid()

        self.draw_obstacles()
        # Draw the agent
        agent_x, agent_y = agent_pos
        agent_rect = pygame.Rect(agent_y * self.cell_size, agent_x * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.agent_color, agent_rect)

        # Draw the target
        target_x, target_y = target_pos
        target_rect = pygame.Rect(target_y * self.cell_size, target_x * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.target_color, target_rect)


        pygame.display.flip()
        time.sleep(0.01)

    def close(self):
        self.initialized=False
        pygame.quit()