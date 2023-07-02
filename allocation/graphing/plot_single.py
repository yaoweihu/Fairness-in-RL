import os
import torch
import numpy as np
import pandas
from matplotlib import pyplot as plt
from geomloss import SamplesLoss
wloss = SamplesLoss("sinkhorn", p=1, blur=0.01)

def plot_bank_cash_over_time(tot_eval_data, path):
    aggregated_tot_ep_bank_cash = tot_eval_data['tot_bank_cash_over_time']

    timesteps = list(range(aggregated_tot_ep_bank_cash.shape[1]))
    means = np.mean(aggregated_tot_ep_bank_cash, axis=0) # Shape: (num human_designed_policies, num timesteps)

    plt.plot(timesteps, means, alpha=0.8)
    plt.title(f'Average Bank Cash Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Bank Cash')
    plt.savefig(path + 'bank_cash_over_time.png')
    plt.close()


def plot_confusion_matrix_over_time(tot_eval_data, path):

    fig, axs = plt.subplots()
    fig.tight_layout()

    tot_tp_over_time = tot_eval_data['tot_tp_over_time']  # (num_eps, num_timesteps, num_groups)
    tot_fp_over_time = tot_eval_data['tot_fp_over_time']
    tot_tn_over_time = tot_eval_data['tot_tn_over_time']
    tot_fn_over_time = tot_eval_data['tot_fn_over_time']

    timesteps = list(range(tot_tp_over_time.shape[1]))

    axs.set_title(f'Average Confusion Terms Over Time For Agent')
    axs.set_xlabel('Timestep')
    axs.set_ylabel('Number of Occurrences')

    tot_tp_means = np.mean(tot_tp_over_time, axis=0)
    tot_fp_means = np.mean(tot_fp_over_time, axis=0)
    tot_tn_means = np.mean(tot_tn_over_time, axis=0)
    tot_fn_means = np.mean(tot_fn_over_time, axis=0)

    # for means in range(means.shape[-1]):
    for metric, means in zip(['TP', 'FP', 'TN', 'FN'], [tot_tp_means, tot_fp_means, tot_tn_means, tot_fn_means]):
        for group_id in range(tot_tp_means.shape[1]):
            axs.plot(timesteps, means[:, group_id], label=f'{metric}: Group {group_id + 1}')

    axs.legend()
    plt.savefig(path + 'confusion_matrix.png')
    plt.close()


def plot_dist_disc_over_time(tot_eval_data, path):
    tot_dist_disc = tot_eval_data['tot_dists']
    # means = np.mean(tot_dist_disc, axis=0)
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


def plot_loans_over_time(tot_eval_data, path):
    agent_names = list(tot_eval_data.keys())
    
    tot_loans_over_time = np.mean(tot_eval_data['tot_loans_over_time'], axis=0)
    timesteps = np.arange(tot_loans_over_time.shape[0])

    for group in range(tot_loans_over_time.shape[1]):
        plt.plot(timesteps, tot_loans_over_time[:, group], label=f'Group {group + 1}')

    plt.title('Cumulative loans')
    plt.xlabel('Timestep ')
    plt.ylabel('# Loans')
    plt.legend()
    plt.savefig(path + 'cumulative loans')
    plt.close()


def plot_rets(exp_path, save_png=True):
    if not os.path.isdir(exp_path):
        exit(f"{exp_path} not found!!!")

    df = pandas.read_csv(f'{exp_path}/progress.csv')
    xs = df['time/total_timesteps']
    ys = df['rollout/ep_rew_mean']

    plt.plot(xs.values, ys.values)
    plt.title('PPO Training: Average Episodic Return Over Time')
    plt.xlabel('Total Timesteps Trained So Far')
    plt.ylabel('Average Episodic Return')
    if save_png:
        plt.savefig(exp_path + 'train_ret_over_time')
    else:
        plt.show()

    plt.close()


def plot_total_rews_over_time(tot_eval_data, path):
    aggregated_tot_rews_over_time = tot_eval_data['tot_rews_over_time']

    timesteps = list(range(aggregated_tot_rews_over_time.shape[1]))
    means = np.mean(aggregated_tot_rews_over_time, axis=0) # Shape: (num human_designed_policies, num timesteps)

    plt.plot(timesteps, means, alpha=0.8)
    plt.title(f'Total Rewards Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Rewards')
    plt.savefig(path + 'total_rewards_over_time.png')
    plt.close()


def plot_tpr_gap_over_time(tot_eval_data, path):
    agent_names = list(tot_eval_data.keys())

    # for an in agent_names:
    tpr_over_time = np.mean(tot_eval_data['tot_tpr_over_time'], axis=0)
    tpr_gap_over_time = np.abs(np.subtract(tpr_over_time[:, 0], tpr_over_time[:, 1]))
    timesteps = np.arange(tpr_gap_over_time.shape[0])
    plt.plot(timesteps, tpr_gap_over_time, label='Diff')

    plt.title('Average Delta Over Time Across Agents')
    plt.xlabel('Timestep')
    plt.ylabel('Average Delta')
    plt.legend()
    # plt.savefig(path + 'average delta over time.png')
    # plt.close()

def plot_tpr_over_time(tot_eval_data, path):

    # for an in agent_names:
    tpr_over_time = np.mean(tot_eval_data['tot_tpr_over_time'], axis=0)
    timesteps = np.arange(tpr_over_time.shape[0])
    for group in range(tpr_over_time.shape[1]):
        plt.plot(timesteps, tpr_over_time[:, group], label=f'Group {group + 1}')

    plt.title('Average TPR Over Time Across Agents')
    plt.xlabel('Timestep')
    plt.ylabel('Average TPR')
    plt.legend()
    plt.savefig(path + 'average tpr over time.png')
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


# plot_training("short", "../../lend_exp_1/Our/models")

def plot_training_short(name, exp_path, save_png=True):
    if not os.path.isdir(exp_path):
        exit(f"{exp_path} not found!!!")

    df = pandas.read_csv(f'{exp_path}/Our/models/progress.csv')
    ys_mean = df[name + '_mean']
    ys_std = df[name + '_std']
    xs = np.arange(len(ys_mean))

    plt.plot(xs, ys_mean.values, label="F-PPO")
    plt.fill_between(xs, ys_mean.values-ys_std.values, ys_mean.values+ys_std.values, alpha=0.4)

    df = pandas.read_csv(f'{exp_path}/Our-L/models/progress.csv')
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

