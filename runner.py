from environments.tabenv1 import tabenv1
import random
import pygame
import sys
env = tabenv1(10,target_pos=(9,9),max_steps=300)
# env = tabenv1(20,target_pos=(19,19),max_steps=1000)
state = env.reset()
print(f"Target Position = {env.target_pos}, Start Position = {state}")
while(True):
    for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.renderer.close()
                    sys.exit()
    act = random.randint(0,3) #random action
    state_, reward, isTruncated, isTerminated = env.step(state,act)
    
    print(f"s, a, s_, r = ({state},{act},{state_},{reward})")
    if isTruncated or isTerminated: 
        if isTerminated: print("Reached Goal")
        else: print("Reached Max steps")
        break
    state=state_