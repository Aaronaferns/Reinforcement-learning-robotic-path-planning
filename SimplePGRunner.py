from environments.tabenv1 import tabenv1
import random
import pygame
import sys
from algorithms.modelFree.agents.simple_pg import PGAgent
import numpy as np
import torch as th
import matplotlib.pyplot as plt

env = tabenv1(10,target_pos=(9,9),max_steps=500,render=True)
# env = tabenv1(20,target_pos=(19,19),max_steps=1000)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print("Using device:", device)

agent = PGAgent(env,lr=0.01,device=device)
state = env.reset()


print(f"Target Position = {env.target_pos}, Start Position = {state}")


BATCH_SIZE=500





episodes = 500


for e in range(1,episodes+1):
    print(f"Episode: {e}")
    done=False

    ep_rewards=[]

    batch_count=0
    
    batch_state=[]
    batch_actions=[]
    batch_rewards=[]
    batch_lens=[]
    steps=0
    while(True):
        steps+=1
        batch_count+=1
        if env.render and hasattr(env.renderer, 'initialized') and env.renderer.initialized:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.renderer.close()
                    sys.exit()
                   
        act = agent.act(agent.normalize_state(state))
        print("state",state)
        state_, reward, isTruncated, isTerminated = env.step(state,act)
        reward += steps*(-0.001)
        print(state_)
        
        #append state, action and reward to batch buffer
        batch_actions.append(act)
        batch_state.append(agent.normalize_state(state))
        ep_rewards.append(reward)
        
        
        state=state_
        #when the episode terminates, reset ep_rewards, state
        if isTruncated or isTerminated or batch_count==BATCH_SIZE :
            ep_reward = sum(ep_rewards)
            ep_len = len(ep_rewards)
            ep_rewards=[]
            batch_lens.append(ep_len)
            batch_rewards.append(ep_reward)
            state = env.reset()
            steps=0

            if batch_count==BATCH_SIZE:
                batch_count==0
                break
    
    weights=[]
    for r,l in zip(batch_rewards,batch_lens):
        weights.extend([r]*l)
    print(f"weights,{len(weights)}")
    loss=agent.train_one_epoch(batch_state,batch_actions,weights)
    print(f"Epoch {e}: Loss {loss}")
