import torch
import numpy as np
from matplotlib import pyplot as plt
from geomloss import SamplesLoss
wloss = SamplesLoss("sinkhorn", p=1, blur=0.01)
cmap ={ 'Max': 'tab:blue', 
        'Greedy': 'tab:blue',  
        'EO': 'tab:green', 
        'PPO': 'tab:orange',
        'G-PPO': 'tab:orange',
        'A-PPO': 'tab:purple', 
        'A-PPO-L': 'tab:pink',
        'F-PPO-S': 'tab:green',
        'F-PPO-L': 'tab:blue',
        'F-PPO': 'tab:red'}

def plot_rews_over_time(names, datas, save_dir):
    for name, data in zip(names, datas):
        aggregated_tot_ep_rews = data['tot_rews_over_time']
        timesteps = list(range(len(aggregated_tot_ep_rews[0])))

        means = np.mean(aggregated_tot_ep_rews, axis=0) # Shape: (num human_designed_policies, num timesteps)
        plt.plot(timesteps, means, alpha=0.8, label=f'{name}', c=cmap[name])

    # plt.title(f'Average Reward Over Time')
    plt.xlabel('Timestep', fontsize=15)
    plt.xticks([0, 5, 10, 15, 20])
    plt.ylabel('Reward', fontsize=15)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir + 'inf-reward.png')
    plt.close()


def plot_delta_over_time(names, datas, save_dir):
    for name, data in zip(names, datas):
        aggregated_tot_ep_deltas = data['tot_deltas_over_time']  # (num_eps, num timesteps)
        timesteps = list(range(len(aggregated_tot_ep_deltas[0])))

        means = np.mean(aggregated_tot_ep_deltas, axis=0) # Shape: (num human_designed_policies, num timesteps)
        plt.plot(timesteps, means, alpha=0.8, label=f'{name}', c=cmap[name])

    # plt.title(f'Average Delta Over Time')
    plt.xlabel('Timestep', fontsize=15)
    plt.xticks([0, 5, 10, 15, 20])
    plt.ylabel('Average Delta', fontsize=15)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir + 'inf-short-term.png')
    plt.close()


def plot_dist_disc_over_time(names, datas, save_dir):
    for name, data in zip(names, datas):
        tot_dist_disc = data['tot_dists']
        means = tot_dist_disc

        timesteps = list(range(tot_dist_disc.shape[1]))
        distance = np.zeros(tot_dist_disc.shape[1],)

        for i in timesteps:
            distance[i] = wloss(torch.tensor(means[:, i, 0]), torch.tensor(means[:, i, 1]))
        plt.plot(timesteps, distance, label=f'{name}', c=cmap[name])

    # plt.title(f'Average Dist Discrapency')
    plt.xlabel('Timestep', fontsize=15)
    plt.ylabel('Long-term Fairness', fontsize=15)
    plt.xticks([0, 5, 10, 15, 20])
    
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir + 'inf-long-term.png')
    plt.close()