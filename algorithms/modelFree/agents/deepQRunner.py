from environments.tabenv1 import tabenv1
import random
import pygame
import sys
from algorithms.modelFree.agents.deepQAgent import DeepQAgent
import numpy as np
import torch as th

env = tabenv1(5,target_pos=(4,4),max_steps=300,render=False)
# env = tabenv1(20,target_pos=(19,19),max_steps=1000)
state = env.reset()
agent = DeepQAgent(env,lr=0.01)
print(f"Target Position = {env.target_pos}, Start Position = {state}")
agent.polulate_rb()#initializing the replay buffer

episodes = 500
reward = []
counter = 1
steps = [0]*episodes
for e in range(1,episodes+1):
    print(f"Episode: {e}")
    reward_cum = 0
    done=False
    agent.reset_epsilon()
    
    print()
    while(not done):
        if env.render==True:
            for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.renderer.close()
                            sys.exit()
        act = agent.act(state)
        state_, reward, isTruncated, isTerminated = env.step(state,act)
        if isTerminated:agent.replay_buffer.add(agent.normalize_state(state),act,reward,agent.normalize_state(state_),1)
        else: agent.replay_buffer.add(agent.normalize_state(state),act,reward,agent.normalize_state(state_),0)
        if(counter%100==0): 
            agent.update_target()
            counter=0
        agent.train_model()
        counter+=1
        steps[e-1]+=1
        # print(f"s, a, s_, r = ({state},{act},{state_},{reward})")
        if isTruncated or isTerminated: 
            done=True
            if isTerminated: print("Reached Goal")
            else: print("Reached Max steps")
            state = env.reset()
            break
        state=state_


with th.no_grad():
    agent.qmodel.eval()
    viz = np.zeros((env.grid_size,env.grid_size))
    for i in range(env.grid_size):
        for j in range(env.grid_size):
                state = (i,j)
                state = np.array(state)
                state = agent.normalize_state(state)
                state = th.tensor(state, dtype=th.float32).unsqueeze(0)
                val = agent.qmodel(state)
                print(val)
import matplotlib.pyplot as plt

plt.plot(steps, marker='o', label='steps Values')

# Add labels and a title
plt.xlabel('episodes')
plt.ylabel('steps')
plt.title('steps vs episode')
plt.legend()

# Display the plot
plt.show()
# Plot with gridlines to resemble a board
plt.figure(figsize=(8, 6))
plt.imshow(viz, cmap='viridis', interpolation='nearest')

# Add gridlines
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.xticks(np.arange(0, env.grid_size, 1))  # Set x-axis tick positions and labels
plt.yticks(np.arange(0, env.grid_size, 1))  # Set y-axis tick positions and labels
plt.gca().set_xticks(np.arange(-0.5, env.grid_size, 1), minor=True)
plt.gca().set_yticks(np.arange(-0.5, env.grid_size, 1), minor=True)
plt.gca().grid(which='minor', color='black', linestyle='-', linewidth=0.5)

# Add colorbar and labels
plt.colorbar(label="Q-value")
plt.title("Visualization of Q-values")
plt.xlabel("State Dimension 1")
plt.ylabel("State Dimension 2")
plt.show()