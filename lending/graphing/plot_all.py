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

def plot_tpr_gap_over_time(names, datas, save_dir):
    for name, data in zip(names, datas):
        tpr_over_time = np.mean(data['tot_tpr_over_time'], axis=0)
        tpr_gap_over_time = np.abs(np.subtract(tpr_over_time[:, 0], tpr_over_time[:, 1]))
        timesteps = np.arange(tpr_gap_over_time.shape[0])
        plt.plot(timesteps, tpr_gap_over_time, c=cmap[name], label=f'{name}')

    # plt.title('Average Delta Over Time Across Agents')
    plt.xlabel('Timestep', fontsize=15)
    plt.ylim((0., .5))
    plt.ylabel('Short-term Fairness', fontsize=15)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir + 'short-term.png')
    plt.close()


def plot_bank_cash_over_time(names, datas, save_dir):
    for name, data in zip(names, datas):
        aggregated_tot_ep_bank_cash = data['tot_bank_cash_over_time']
        timesteps = list(range(aggregated_tot_ep_bank_cash.shape[1]))
        mean_val = np.mean(aggregated_tot_ep_bank_cash, axis=0) # Shape: (num human_designed_policies, num timesteps)
        plt.plot(timesteps, mean_val, label=f'{name}', c=cmap[name])

    # plt.title(f'Average Bank Cash Over Time')
    plt.xlabel('Timestep', fontsize=15)
    plt.ylabel('Bank Cash', fontsize=15)
    plt.ylim((9900, 12000))
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir + 'reward.png')
    plt.close()


def plot_loans_over_time(names, datas, save_dir):
    for name, data in zip(names, datas):
        tot_loans_over_time = np.mean(data['tot_loans_over_time'], axis=0)
        timesteps = np.arange(tot_loans_over_time.shape[0])
        plt.plot(timesteps, np.sum(tot_loans_over_time, axis=1), c=cmap[name], label=f'{name}')

    # plt.title('Cumulative loans')
    plt.xlabel('Timestep', fontsize=15)
    plt.ylabel('# Loans', fontsize=15)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir + 'loans.png')
    plt.close()

def plot_dist_disc_over_time(names, datas, save_dir):
    for name, data in zip(names, datas):
        tot_dist_disc = data['tot_dists']
        timesteps = list(range(tot_dist_disc.shape[1]))
        distance = np.zeros(tot_dist_disc.shape[1],)
        for i in timesteps:
            distance[i] = wloss(torch.tensor(tot_dist_disc[:, i, 0]), torch.tensor(tot_dist_disc[:, i, 1]))
        plt.plot(timesteps, distance, label=f'{name}', c=cmap[name])

    # plt.title(f'Average Distribution Discrapency')
    plt.xlabel('Timestep', fontsize=15)
    plt.ylabel('Long-term Fairness', fontsize=15)
    # plt.ylim((0.08, 0.45))
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir + 'long-term.png')
    plt.close()