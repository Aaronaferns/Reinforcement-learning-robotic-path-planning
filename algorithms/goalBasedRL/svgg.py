# File for stein variational goal generation
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch as th
import torch.nn.functional as F
from torch.distributions import Beta
from anomalygoal import log_p_valid
import numpy as np
import matplotlib.pyplot as plt

alpha=2.0
beta_p = 2.0
lr_svgg=1e-1
temperature =0.1
num_svgd_steps=10000

def rbf_kernel(x, h=None):
    """
    RBF kernel and its gradient.
    :param x: Tensor of shape (n_particles, dim)
    :param h: Bandwidth parameter. If None, use median heuristic.
    :return: Kernel matrix K and gradient dK
    """
    pairwise_dists = th.cdist(x, x, p=2).pow(2)  # shape: (n, n)
    if h is None:
        h = th.median(pairwise_dists)
        h = h / th.log(th.tensor(x.shape[0], dtype=th.float32) + 1.0)
        h = th.clamp(h, min=1e-4)

    K = th.exp(-pairwise_dists / h)
    return K, h

def pskills(D_phi, alpha, beta_p):
    beta_dist = Beta(alpha, beta_p)
    log_p = beta_dist.pdf(D_phi)
    return log_p

def pgoals(goals,model,anomaly_model,a,b):
    log_p_skills = pskills(model(goals),a,b)
    log_p_val = log_p_valid(goals,anomaly_model,temperature=0.1)
    return log_p_val+log_p_skills

def log_pgoals(goals, model, anomaly_model, a, b, temperature=0.1):
    
    goals.requires_grad_(True)
    logp = pgoals(goals, model, anomaly_model, a, b, temperature)
    return logp

def svgd_step(goals, log_prob_fn, lr=1e-2):
    """
    Perform one SVGD update step.
    :param goals: Tensor of shape (n_particles, dim)
    :param log_prob_fn: Function that returns log-prob and requires grad
    :param lr: Learning rate
    :return: Updated goals
    """
    goals = th.tensor(goals,dtype=th.float32)
    goals = goals.clone().detach().requires_grad_(True)
    n_particles = goals.shape[0]

    log_probs = log_prob_fn(goals)  # shape: (n_particles,)
    grads = th.autograd.grad(log_probs.sum(), goals)[0]  # shape: (n_particles, dim)

    K, h = rbf_kernel(goals)
    dK = -2 * (goals.unsqueeze(1) - goals.unsqueeze(0)) / h * K.unsqueeze(2)  # shape: (n, n, dim)

    phi = (K @ grads + dK.sum(dim=0)) / n_particles  # shape: (n_particles, dim)

    with th.no_grad():
        goals += lr * phi

    return goals.detach().numpy()

def svgd(goals,num_svgd_steps,model,anomaly_model):
    for step in range(num_svgd_steps):
        goals = svgd_step(goals, model, anomaly_model, alpha, beta_p, lr_svgg, temperature)
        
        # Optionally print the progress of the optimization
        if step % 10 == 0:
            logp = log_pgoals(th.tensor(goals,dtype=th.float32), model, anomaly_model, alpha, beta_p, temperature)
            print(f"Step {step}: mean log p = {logp.mean().item():.4f}") 
    return goals  



def test_svgd_with_gaussian():
    # Target: standard 2D Gaussian
    n_particles = 100
    goals = np.random.uniform(low=0.0, high=5.0, size=(n_particles, 2))
    for step in range(1000):
        goals = svgd_step(goals, log_p_gaussian, lr=0.1)
        if step % 100 == 0:
            print(f"Step {step}: Mean = {np.mean(goals, axis=0)}")

            # Plot results
            plt.figure(figsize=(6, 6))
            plt.scatter(goals[:, 0], goals[:, 1], alpha=0.7)
            plt.title("SVGD: Particles approximating 2D Gaussian")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.show()


def log_p_gaussian(goals):
    return -0.5 * ((goals - 2.0)**2).sum(dim=1)
    

if __name__ == "__main__":
    test_svgd_with_gaussian()

    

