import os
import torch
import pandas
import numpy as np
from matplotlib import pyplot as plt
from geomloss import SamplesLoss
wloss = SamplesLoss("sinkhorn", p=1, blur=0.01)


def plot_rets(exp_path):

    df = pandas.read_csv(f'./{exp_path}/progress.csv')
    xs = df['time/total_timesteps']
    ys = df['rollout/ep_rew_mean']

    plt.plot(xs, ys)
    plt.title('PPO Training: Average Episodic Return Over Time')
    plt.xlabel('Total Timesteps Trained So Far')
    plt.ylabel('Average Episodic Return')
    plt.savefig(exp_path + 'train_ret_over_time')
    plt.close()


def plot_delta_over_time(tot_eval_data, path):
    aggregated_tot_ep_deltas = tot_eval_data['tot_deltas_over_time']  # (num_eps, num timesteps)
    timesteps = list(range(len(aggregated_tot_ep_deltas[0])))

    plt.title(f'Average Delta Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Delta')

    means = np.mean(aggregated_tot_ep_deltas, axis=0) # Shape: (num human_designed_policies, num timesteps)

    # Plot means and add a legend
    plt.plot(timesteps, means, alpha=0.8)
    plt.legend()
    plt.savefig(path + 'delta_over_time.png')
    plt.close()

    # Next, show the average delta for each agent
    # plt.title('Average Delta Mean')
    # plt.ylabel('Delta')
    # plt.bar(agent_names, np.mean(means, axis=1))
    # plt.show()
    # plt.close()


def plot_rews_over_time(tot_eval_data, path):
    aggregated_tot_ep_rews = tot_eval_data['tot_rews_over_time']

    timesteps = list(range(len(aggregated_tot_ep_rews[0])))

    plt.title(f'Average Reward Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')

    means = np.mean(aggregated_tot_ep_rews, axis=0) # Shape: (num human_designed_policies, num timesteps)

    # Plot means and add a legend
    # colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(len(agent_names))]
    plt.plot(timesteps, means, alpha=0.8)
    plt.legend()
    plt.savefig(path + 'reward_over_time.png')
    plt.close()


def plot_dist(tot_eval_data, step, path):
    
    width = 0.25
    tot_dists = np.mean(tot_eval_data['tot_dists'], axis=0)
    x = np.array(list(range(7)))
    plt.bar(x-width/2, tot_dists[step, 0], width, label=f'Group {1}')
    plt.bar(x+width/2, tot_dists[step, 1], width, label=f'Group {2}')
    
    plt.title('Distribution')
    plt.xlabel('credit score')
    plt.ylabel('probability')
    plt.legend()
    plt.savefig(path + 'dist_'+str(step)+'.png')
    plt.close()


def plot_dist_disc_over_time(tot_eval_data, path):
    tot_dist_disc = tot_eval_data['tot_dists']
    means = tot_dist_disc

    timesteps = list(range(tot_dist_disc.shape[1]))
    distance = np.zeros(tot_dist_disc.shape[1],)

    for i in timesteps:
        distance[i] = wloss(torch.tensor(means[:, i, 0]), torch.tensor(means[:, i, 1]))

    plt.plot(timesteps, distance)
    plt.title(f'Average Dist Discrapency')
    plt.xlabel('Timestep')
    plt.ylabel('Dist_Disc')
    plt.savefig(path + 'dist_disc_over_time.png')
    plt.close()


def plot_training(name, exp_path, save_png=True):
    
    if not os.path.isdir(exp_path):
        exit(f"{exp_path} not found!!!")

    df = pandas.read_csv(f'{exp_path}/progress.csv')
    ys_mean = df[name + '_mean']
    ys_std = df[name + '_std']
    xs = np.arange(len(ys_mean))

    plt.plot(xs, ys_mean.values)
    plt.fill_between(xs, ys_mean.values-ys_std.values, ys_mean.values+ys_std.values, alpha=0.4)
    
    plt.xlabel('Iteration')
    plt.ylabel('Values')

    if save_png:
        plt.savefig(exp_path + "/" + name + '.png')
    else:
        plt.show()

    plt.close()

# plot_training("short", "../results/F-PPO-L/models")


def plot_training_short(name, exp_path, save_png=True):
    if not os.path.isdir(exp_path):
        exit(f"{exp_path} not found!!!")

    df = pandas.read_csv(f'{exp_path}/F-PPO/models/progress.csv')
    ys_mean = df[name + '_mean']
    ys_std = df[name + '_std']
    xs = np.arange(len(ys_mean))

    plt.plot(xs, ys_mean.values, label="F-PPO")
    plt.fill_between(xs, ys_mean.values-ys_std.values, ys_mean.values+ys_std.values, alpha=0.4)

    df = pandas.read_csv(f'{exp_path}/F-PPO-L/models/progress.csv')
    ys_mean = df[name + '_mean']
    ys_std = df[name + '_std']
    xs = np.arange(len(ys_mean))

    plt.plot(xs, ys_mean.values, label="F-PPO-L")
    plt.fill_between(xs, ys_mean.values-ys_std.values, ys_mean.values+ys_std.values, alpha=0.4)
    
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Short-term Fairness', fontsize=15)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if save_png:
        plt.savefig(exp_path + "/" + name + '.png')
    else:
        plt.show()

    plt.close()


plot_training_short("short", "../results/")