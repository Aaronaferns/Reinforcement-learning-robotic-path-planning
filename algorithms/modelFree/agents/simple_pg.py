
from algorithms.modelFree.neuralNets.policy import Policy
import torch as th
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import random
from torch.distributions.categorical import Categorical

class PGAgent:
    def __init__(self,env,lr,device):
        self.DEVICE=device
        self.env = env
        self.policy = Policy(env.nS,env.nA).to(self.DEVICE)
        self.learning_rate = lr
        self.optimizer = Adam(self.policy.parameters(),lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.train_loss = []
        self.epoch_loss=[]
        self.debug = False
  


    def normalize_state(self, state):
        i, j = state[0],state[1]
        grid_size = self.env.grid_size
        i_normalized = i / (grid_size - 1)
        j_normalized = j / (grid_size - 1)
        ret = np.array([i_normalized, j_normalized], dtype=np.float32)
        if self.debug: print("check if correct state is returned by normalize: ",state,ret)
        return ret


    def act(self,state_norm):
        state_norm = th.tensor(state_norm, dtype=th.float32)  
        return self.get_policy_(state_norm.to(self.DEVICE)).sample().item()
    
    def get_policy_(self,state_norm):
        logits = self.policy(state_norm)
        return Categorical(logits=logits)
    
    def compute_loss_(self,state_norm,act,weights):
        logp=self.get_policy_(state_norm).log_prob(act)
        return -(logp*weights).mean()
    
    def train_one_epoch(self,state_norm,act,weights):
        self.optimizer.zero_grad()
        state_norm,act,weights=th.tensor(state_norm,dtype=th.float32),th.tensor(act,dtype=th.float32),th.tensor(weights,dtype=th.float32)
        loss=self.compute_loss_(state_norm.to(self.DEVICE),act.to(self.DEVICE),weights.to(self.DEVICE))
        loss.backward()
        self.optimizer.step()
        self.epoch_loss.append(loss.item())
        return loss.item()

    







            

