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
agent = PGAgent(env,lr=0.01)
state = env.reset()
agent.normalize_state(state)

print(f"Target Position = {env.target_pos}, Start Position = {state}")


BATCH_SIZE=500





episodes = 500
reward = []

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
        batch_count+=1
        if env.render and hasattr(env.renderer, 'initialized') and env.renderer.initialized:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.renderer.close()
                    sys.exit()
                   
        act = agent.act(state)
        state_, reward, isTruncated, isTerminated = env.step(state,act)
        agent.normalize_state(state_)
        
        batch_actions.append(act)
        batch_state.append(state)
        ep_rewards.append(reward)

        if isTruncated or isTerminated :
            ep_reward = sum(ep_rewards)
            ep_len = len(ep_rewards)

            batch_lens.append(ep_len)
            state = env.reset()
            agent.normalize_state(state)
            if batch_count==BATCH_SIZE:
                break
        state=state_
    
    weights=[]
    for r,l in zip(batch_rewards,batch_lens):
        weights+=[r]*l
    agent.train_one_epoch(batch_state,batch_actions,weights)
