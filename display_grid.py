import matplotlib.pyplot as plt
import numpy as np
# Create a figure and axis


# Set the limits of the plot
def display_grid(ax, grid_dim):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Draw the grid lines
    for i in range(4):
        ax.axhline(i / grid_dim, color='black', linewidth=0.5)
        ax.axvline(i / grid_dim, color='black', linewidth=0.5)

    # Add labels to the grid cells
    # for i in range(3):
    #     for j in range(3):
    #         ax.text(i + 0.5, j + 0.5, f'({i},{j})', ha='center', va='center')

    # Set the ticks and labels
    ax.set_xticks(np.linspace(0, 1, 7))
    ax.set_yticks(np.linspace(0, 1, 7))
    # ax.set_xticklabels(np.round(np.linspace(0, 1, 4), 2))
    # ax.set_yticklabels(np.round(np.linspace(0, 1, 4), 2))

def display_agents(ax, agent_position, grid_dim, grid_mask):
    # Draw the agent
    x, y = agent_position
    ax.text(x, y + 0.1, f'({x:.2f},{y:.2f})', ha='center', va='center')
    # ax.text(x, y + 0.05, next_action, ha='center', va='center')
    ax.plot(agent_position[0], agent_position[1], 'bo')

    for i in range(grid_dim):
        for j in range(grid_dim):
            if grid_mask[i, j] == 1:
                ax.add_patch(plt.Rectangle((i / grid_dim, j / grid_dim), 1 / grid_dim, 1 / grid_dim, fill=True, color='gray'))

def display_targets(ax, target_position):
    # Draw the target
    ax.plot(target_position[0], target_position[1], 'r^')
