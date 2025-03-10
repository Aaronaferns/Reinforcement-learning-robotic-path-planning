import numpy as np
import torch
import matplotlib.pyplot as plt
from display_grid import display_grid
from display_grid import display_agents
from display_grid import display_targets

class gridworld_env:
    def __init__(self):
        self.grid_dim = 8
        self.agent_position = self.init_agent_position()
        self.target_position = self.init_target_position(self.agent_position)

        # up down left right
        self.n_acts = 4
        self.action_dict = self.init_action_dict()

        # agent position, target position
        self.n_obs = 4

        # action mask
        self.action_mask = [0 for _ in range(self.n_acts)]
        self.grid_mask = np.zeros((self.grid_dim, self.grid_dim))

    def init_agent_position(self):
        x = np.random.randint(0, self.grid_dim)
        y = np.random.randint(0, self.grid_dim)
        offset = 1 / self.grid_dim

        x = x * offset + offset / 2
        y = y * offset + offset / 2
        agent_position = np.array((x, y))
        return agent_position

    def init_target_position(self, agent_position):
        bad_x, bad_y = self.convert_position_to_grid_cell(agent_position)
        good_position = False
        while not good_position:
            x = np.random.randint(0, self.grid_dim)
            y = np.random.randint(0, self.grid_dim)

            if x == bad_x and y == bad_y:
                good_position = False
            else:
                good_position = True

        offset = 1 / self.grid_dim

        x = x * offset + offset / 2
        y = y * offset + offset / 2
        target_position = np.array((x, y))

        return target_position

    def init_action_dict(self):
        action_value = 1 / self.grid_dim
        return{
            0: np.array([action_value, 0]),
            1: np.array([-action_value, 0]),
            2: np.array([0, action_value]),
            3: np.array([0, -action_value])
        }

    def convert_position_to_grid_cell(self, position):
        x, y = position
        col = int((x * self.grid_dim))
        row = int((y * self.grid_dim))
        return col, row

    def update_grid_mask(self):
        x, y = self.convert_position_to_grid_cell(self.agent_position)
        self.grid_mask[x, y] = 1

    def update_action_mask(self):
        self.action_mask = [0]* self.n_acts
        self.update_grid_mask()

        for action in range(self.n_acts):
            proposed_position = self.agent_position + self.action_dict[action]
            x, y = proposed_position
            if 0 <= x <= 1 and 0 <= y <= 1:
                next_row, next_col = self.convert_position_to_grid_cell(proposed_position)
                # grid locations that have already been visited
                if self.grid_mask[next_row, next_col] == 1:
                    self.action_mask[action] = 1
            # out of bounds
            else:
                self.action_mask[action] = 1

    def step(self, action):
        # self.render()
        done = False
        # self.update_action_mask()

        if action == 0: # right
            self.agent_position[0] += 1 / self.grid_dim
        elif action == 1: # left
            self.agent_position[0] -= 1 / self.grid_dim
        elif action == 2: # up
            self.agent_position[1] += 1 / self.grid_dim
        elif action == 3: # down
            self.agent_position[1] -= 1 / self.grid_dim

        self.update_action_mask()
        if sum(self.action_mask) == 4:
            self.grid_mask = np.zeros((self.grid_dim, self.grid_dim))
            self.update_action_mask()

        mask = self.action_mask
        obs = [self.agent_position[0], self.agent_position[1], self.target_position[0], self.target_position[1]]
        rew = 1

        agent_cell = self.convert_position_to_grid_cell(self.agent_position)
        target_cell = self.convert_position_to_grid_cell(self.target_position)
        if agent_cell == target_cell:
            done = True
            # self.render()

        return obs, mask, rew, done

    def render(self):
        fig, ax = plt.subplots()
        display_grid(ax, self.grid_dim)
        display_agents(ax, self.agent_position, self.grid_dim, self.grid_mask)
        display_targets(ax, self.target_position)
        plt.show()

        print(f"agent position: {self.agent_position}")
        print(f"target position: {self.target_position}")

        input("Press any key to continue...")

    def reset(self):
        self.agent_position = self.init_agent_position()
        self.target_position = self.init_target_position(self.agent_position)
        self.grid_mask = np.zeros((self.grid_dim, self.grid_dim))

        obs = [self.agent_position[0], self.agent_position[1], self.target_position[0], self.target_position[1]]
        self.update_action_mask()
        mask = self.action_mask

        return obs, mask
