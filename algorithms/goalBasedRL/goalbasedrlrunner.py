
import numpy as np
import random
import torch as th
import torch.optim as optim

from environments.tabenv1 import tabenv1
from svgg import *
env = tabenv1(10,2000)
#goal conditioned policy gradient algorithm
#success predictor network D for pskills(g)
from skillspredictor import SkillsPredictorModel
#validity predictor
# from sklearn.svm import OneClassSVM
# ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
from anomalygoal import AnomalyDetector, log_p_valid

#Buffer
from memory import Buffer
replay_buffer = Buffer(env.nS,env.nA,50000) #has buffer, R, O

q=[]



#constants
#**********************************************#

num_epochs = 10000

#**********************************************# 

#DATA COLLECTION
num_traj = 10000

#**********************************************#

#Anomaly Detector
lr_anomaly = 1e-3
batch_r = 1000

#**********************************************#

#skills Detector
lr_skills = 1e-3
batch_s = 1000

#**********************************************#

#SVGG
alpha=2.0
beta_p = 2.0
lr_svgg=1e-2
temperature =0.1
num_svgd_steps=100

#*********************************************#
#MODELS
oneClassAnomalyDetector = AnomalyDetector(env.nS)
optimizer_anomaly = optim.Adam(oneClassAnomalyDetector.parameters(),lr=lr_anomaly)

skillsModel = SkillsPredictorModel(env.nS)
optimizer_skills = optim.Adam(skillsModel.parameters(),lr=lr_skills)


#*********************************************#




def train(num_epochs,num_traj):
    for n in range(num_epochs):
        #Data collection
        return

def sample_data(num_traj):  
    for r in range(num_traj):
        #sample a goal g from q
        g = random.choice(q)
        obs = env.reset()
        #perform rollouts
        done=False
        trajectory_memory=[]
        while not done:
            act=get_action(obs) #from actor 
            obs_,r,truncated,terminated=env.step(obs,act)
            # replay_buffer.add((obs,act,r,obs_,g))
            trajectory_memory.append(g,obs,act,r,obs_)
            if truncated or terminated:
                done = True
                if terminated:
                    replay_buffer.add_R(obs_)
                    replay_buffer.add_O(g,1)
                else: replay_buffer.add_O(g,0)      
        replay_buffer.add(trajectory_memory)
def update_skillsmodel(iterations,batch_size,model,optimizer,loss_fn):
    model.train()
    for t in range(iterations):
        g,success = replay_buffer.get_batch_o(batch_size)    
        g,success = th.tensor(g,type=th.float32),th.tensor(success,type=th.float32)
        y=model(g)
        loss=loss_fn(y,success)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t % 100 == 0:
            print(f"Iter {t}: Loss = {loss.item():.4f}")

def update_validmodel(model,optimizer,batch_r):
    batch = replay_buffer.get_batch_r()
    for start_idx in range(0,replay_buffer.size_r,batch_r):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Zero out the gradients from the previous step
        if start_idx+batch_r <= replay_buffer.size_r:mini_batch_goals = th.tensor(batch[start_idx,start_idx+batch_r],th.float32)
        else: mini_batch_goals = th.tensor(batch[start_idx,-1],th.float32)
        # Compute the log probability of valid goals
        log_p_val = log_p_valid(mini_batch_goals, model, temperature)

        # We want to minimize the negative log probability (i.e., minimize reconstruction error)
        loss = -log_p_val.mean()  # Negative because we are maximizing the log probability
        
        # Backpropagation and optimization step
        loss.backward()  # Compute gradients
        optimizer.step()  # Update the model parameters
        
        print(f" Loss: {loss.item():.4f}")

def svgd(goals,num_svgd_steps,model,anomaly_model):
    for step in range(num_svgd_steps):
        goals = svgd_step(goals, model, anomaly_model, alpha, beta_p, lr_svgg, temperature)
        
        # Optionally print the progress of the optimization
        if step % 10 == 0:
            logp = log_pgoals(th.tensor(goals,th.float32), model, anomaly_model, alpha, beta_p, temperature)
            print(f"Step {step}: mean log p = {logp.mean().item():.4f}") 
    return goals   
                  
def train_one_epoch(
    num_traj,
    iterations,
    batch_size,
    model,
    optimizer,
    loss_fn
    ):
    
    sample_data(num_traj)
    update_skillsmodel(iterations,batch_size,model,optimizer,loss_fn)
    update_validmodel(oneClassAnomalyDetector,optimizer_anomaly,batch_r)
    goals = svgd(np.array(q),num_svgd_steps,skillsModel,oneClassAnomalyDetector)
    q=goals
    
    
                
        
         
    