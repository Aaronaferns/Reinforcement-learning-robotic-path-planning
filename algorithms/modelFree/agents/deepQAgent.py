#initialize replay memory D to capacity N
#initialize action_value function Q with random weights
#   loop through episodes:
#          initialize state 
#          loop till end:
#           use epsilon greedy action selection
#           execute action, get s_ and r
#           store s,a,r,s_ in replay memory
#           sample a minibatch from replay memory
#           set yj ={ rj for terminal s_     or rj = GAMMA*maxa'Q(s_,a':O)}
#           perform gradient step on the diff of states
from algorithms.modelFree.uitls import ReplayBuffer
from algorithms.modelFree.neuralNets.qmodel import QNet
import torch as th
import torch.nn as nn
import numpy as np
from torch.optim import Adam


import random
class DeepQAgent:
    def __init__(self,env,lr):
        self.replay_buffer = ReplayBuffer(500,env.nA,env.nS)
        self.env = env
        self.GAMMA = 0.99
        self.EPSILON = 0.5
        self.qmodel = QNet(env.nS,env.nA)
        self.qmodel._initialize_weights() #randomly initilaize weights
        self.learning_rate = lr
        self.optimizer = Adam(self.qmodel.parameters(),lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.train_loss = []
        self.target_model = QNet(env.nS,env.nA) 
        self.target_model.load_state_dict(self.qmodel.state_dict())
        self.episodic_loss=[]
        self.batch_size=64
        self.debug = False

    def decay_epsilon(self):
        self.EPSILON = max(self.EPSILON * 0.995, 0.01)  # Epsilon decays till it reaches 0.01
    def update_target(self):
        self.target_model.load_state_dict(self.qmodel.state_dict())

    def normalize_state(self, state):
        i, j = state[0],state[1]
        grid_size = self.env.grid_size
        i_normalized = i / (grid_size - 1)
        j_normalized = j / (grid_size - 1)
        ret = np.array([i_normalized, j_normalized], dtype=np.float32)
        if self.debug: print("check if correct state is returned by normalize: ",state,ret)
        return ret

    def polulate_rb(self):
        render = True
        if self.env.render == True: 
            self.env.turnOffRender()
        else: render = False
        state = self.env.reset()
        
        
        while(not self.replay_buffer.buffer_full):
            act = random.randint(0,self.env.nA-1) #random action
            state_, reward, isTruncated, isTerminated = self.env.step(state,act)

            if isTerminated:
                self.replay_buffer.add(self.normalize_state(state),act,reward,self.normalize_state(state_),1)
                if self.debug: print("Terminated, added this to buffer: ",self.normalize_state(state),act,reward,self.normalize_state(state_),1)
            else: self.replay_buffer.add(self.normalize_state(state),act,reward,self.normalize_state(state_),0)
            state=state_
            if isTruncated or isTerminated: 
                state=self.env.reset()
        if render == True: self.env.turnONRender()

    
    
    def act(self,state):  
        with th.no_grad():
            self.qmodel.eval()
            state = self.normalize_state(state)
            state = th.tensor(state, dtype=th.float32).unsqueeze(0) #This adds a batch dimension as it is needed by the model
            q_values = self.qmodel(state).squeeze() # back to a single vector

            #Epsilon Greedy actions
            if random.random() < self.EPSILON:
                action = random.randint(0, self.env.nA - 1) # Random 
            else:
                action = th.argmax(q_values).item()         # Greedy action: argmax selects the action with the highest value; item() gives the action as an integert
            
            return action
    
    def train_model(self):
        batch_size = self.batch_size
        s,a,s_,r,t = self.replay_buffer.batch(batch_size)
        # y_=np.zeros(batch_size)
        
        with th.no_grad():
            self.target_model.eval()
            next_q_vals = self.target_model(th.tensor(s_, dtype=th.float32))
            max_next_q_vals = th.max(next_q_vals, dim=1).values    ##values gives actual values not indices
            terminal_states = th.tensor(t,dtype=th.bool).squeeze()
            if self.debug: 
                print("max q_ shape and terminal mask shape: ",max_next_q_vals.shape,"UU", terminal_states.shape)
                print(terminal_states)
            rewards = th.tensor(r,dtype=th.float32)
            y_=rewards.clone().squeeze()
            y_[~terminal_states]+=self.GAMMA * max_next_q_vals[~terminal_states]
            if self.debug: 
                print("Y_ shape: ",y_.shape)
    
        # index = 0
        # for r1,t in zip(r,t):
        #     if t==1:
        #         y_[index]=r1
        #     else: 
        #         y_[index]=r1+self.GAMMA*th.max(probs[index]).item()
        #     index+=1
        

        a,s,y_ = th.tensor(a,dtype=th.int64),th.tensor(s, dtype=th.float32),th.tensor(y_, dtype=th.float32)
        
        self.qmodel.train()
        q_values = self.qmodel(s)

        q_a = q_values.gather(1, a.long())
        if self.debug: print("shape of the Q_vals returned by the policy newtork is:  ",q_a.shape)
        loss = self.loss_fn(q_a.squeeze(),y_)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_loss.append(loss.item())

    def add_episodic_loss(self):
        self.episodic_loss.append( sum(self.train_loss)/len(self.train_loss))
        self.train_loss=[]







            
