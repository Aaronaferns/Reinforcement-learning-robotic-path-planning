import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
from gridworld_env import gridworld_env

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def train(env_name='CartPole-v0', hidden_sizes=[32], pi_lr=1e-3,v_lr=1e-2, clip=0.2,
          epochs=200, batch_size=5000, render=False):

    # Set device to CUDA if available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # make environment, check spaces, get obs / act dims
    env = gridworld_env()

    # assert isinstance(env.observation_space, Box), \
    #     "This example only works for envs with continuous state spaces."
    # assert isinstance(env.action_space, Discrete), \
    #     "This example only works for envs with discrete action spaces."

    obs_dim = env.n_obs
    n_acts = env.n_acts

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts]).to(device)

    # value net
    val_net = mlp(sizes=[obs_dim]+hidden_sizes+[1]).to(device)

    # make function to compute action distribution
    def get_policy(obs, mask):
        logits = logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device))

        # mask to prevent agent from moving to a cell that has already been visited
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool).to(device)
        masked_logits = logits.masked_fill(mask_tensor, -float('inf'))
        return Categorical(logits=masked_logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs, mask):
        act = get_policy(obs, mask).sample()
        return act.item()
    def get_value(obs):
        val = val_net(torch.as_tensor(obs, dtype=torch.float32).to(device))
        return val.item()
    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss_pi(obs, mask, act, adv,logp, weights):
        logp_new = get_policy(obs, mask).log_prob(act)

        ratio = torch.exp(logp_new-logp)
        clip_adv = torch.clamp(ratio, 1-clip, 1+clip) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_pi

    def compute_loss_v(obs, rew):
        return ((val_net(torch.as_tensor(obs, dtype=torch.float32).to(device)) - rew)**2).mean()



    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=pi_lr)
    optimizer_v = Adam(val_net.parameters(), lr=v_lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_mask = []         # for action mask
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        batch_adv = []
        batch_logp=[]
        batch_rew = []
        # reset episode-specific variables
        obs, mask = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep
        ep_leng=0
        # render first episode of each epoch
        # finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            # if (not finished_rendering_this_epoch) and render:
            #     env.render()

            # save obs
            batch_obs.append(obs)
            batch_mask.append(mask)

            # act in the environment
            act = get_action(obs, env.action_mask)
            adv =get_value(obs)
            batch_adv.append(adv)
            batch_logp.append(get_policy(torch.tensor(obs).to(device), torch.tensor(mask).to(device)).log_prob(torch.tensor(act).to(device)).item())

            step_result = env.step(act)
            obs, mask, rew, done = step_result[0], step_result[1], step_result[2], step_result[3]
            ep_leng+=1
            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)
            batch_rew.append(rew*ep_leng)
            


            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                # print(f"batch_weights before: {batch_weights}")
                batch_weights += [ep_ret] * ep_len
                obs, mask = env.reset()  # first obs comes from starting distribution
                ep_leng=0
                done = False  # signal from environment that episode is over
                ep_rews = []

                # won't render again this epoch
                # finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                # input("Press any key to continue...")
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        # we don't tensorize batch_mask because that happens in the get_policy function
        batch_loss = compute_loss_pi(obs=torch.as_tensor(batch_obs, dtype=torch.float32).to(device),
                                  mask=batch_mask,
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32).to(device),
                                  adv=torch.as_tensor(batch_adv, dtype=torch.int32).to(device),
                                  logp=torch.as_tensor(batch_logp, dtype=torch.int32).to(device),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32).to(device)
                                  )
        batch_loss.backward()
        optimizer.step()


        optimizer_v.zero_grad()
        batch_loss_v = compute_loss_v(obs=torch.as_tensor(batch_obs, dtype=torch.float32).to(device),
                                      rew=torch.as_tensor(batch_rew, dtype=torch.float32).to(device))
        batch_loss_v.backward()
        optimizer_v.step()




        return batch_loss, batch_loss_v, batch_rets, batch_adv, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_loss_v, batch_rets, batch_adv, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t v_loss: %.3f \t advantage: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, batch_loss_v, np.mean(batch_rets),np.mean(batch_adv), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, pi_lr=args.lr)