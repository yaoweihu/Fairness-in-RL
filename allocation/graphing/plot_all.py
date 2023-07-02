import os
import copy
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
cmap = { 
        'CPO': 'tab:green', 
        'PPO': 'tab:orange',
        'A-PPO': 'tab:purple', 
        'F-PPO': 'tab:red',
    }


def plot_att_all_over_time_across_agents(tot_eval_data, save_dir=None, smoothing_val=None):
    """
    Plots average episodic attention allocated over time.

    Parameters
    ----------
    tot_eval_data: {agent_name: {tot_att_all: np.array(num_episodes, num_timesteps, num_locations)}}
    save_dir: if not None, saves the plot to save_dir. Otherwise, it displays the plot
    smoothing_val: if not None, applies gaussian smoothing with std=smoothing_val for better visualization

    Notes
    -----
    aggregated_tot_ep_att_all: the attention allocated per timestep per agent, should be {agent_name: tot_ep_att_all} where
                            tot_ep_att_all shape is (num_episodes, num_timesteps, num_locations)
    Returns:
    """
    agent_names = list(tot_eval_data.keys())

    fig, axs = plt.subplots(len(agent_names))
    # fig.tight_layout()
    if len(agent_names) == 1:
        axs = [axs]

    for name, ax in zip(agent_names, axs):
        tot_ep_att_all = tot_eval_data[name]['tot_att_all']  # (num_episodes, num_timesteps, num_locations)
        timesteps = list(range(len(tot_ep_att_all[0])))
        num_eps = len(tot_ep_att_all)

        ax.set_title(name)
        # ax.set_xlabel('Timestep')
        ax.set_ylabel('Attn. Allocated')

        means = np.mean(tot_ep_att_all, axis=0)

        for i in range(means.shape[-1]):
            if smoothing_val is not None:
                ax.plot(timesteps, gaussian_filter1d(means[:, i],smoothing_val), label=f'Site {i}')
            else:
                ax.plot(timesteps, means[:, i], label=f'Site {i}')

    # fig.suptitle('Average Attention Allocated Over Time')
    plt.xlabel('Timestep')
    plt.subplots_adjust(top=0.85, hspace=0.75)
    plt.legend()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'att_all_over_time.png'))
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_deltas_over_time(tot_eval_data, save_dir=None):
    """
    Plots average deltas over time

    Parameters
    ----------
    tot_eval_data: {agent_name: {tot_deltas: np.array(num_episodes, num_timesteps)}}
    save_dir: if not None, saves the plot to save_dir. Otherwise, it displays the plot
    smoothing_val: if not None, smooths the graph with a 1d gaussian filter with std=smoothing_val for better visualization

    Notes
    -----
    aggregated_tot_ep_deltas: the delta per timestep per agent, should be {agent_name: tot_deltas} where
                            tot_deltas shape is (num_episodes, num_timesteps)
    """
    agent_names = list(tot_eval_data.keys())

    aggregated_tot_ep_deltas = [tot_eval_data[name]['tot_deltas'] for name in agent_names]

    timesteps = list(range(len(aggregated_tot_ep_deltas[0][0])))
    num_eps = len(aggregated_tot_ep_deltas[0])

    # plt.title(f'Average Delta Over Time')
    plt.xlabel('Timestep', fontsize=15)
    plt.ylabel('Short-term Fairness', fontsize=15)

    means = np.mean(aggregated_tot_ep_deltas, axis=1)

    # Plot means and add a legend
    for i in range(len(agent_names)):
        plt.plot(timesteps, means[i], alpha=0.8, label=agent_names[i], c=cmap[agent_names[i]])

    plt.legend(agent_names, loc=1, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'deltas_over_time_graph.png'))
        plt.close()
    else:
        plt.show()
        plt.close()

    # Next, show the average delta for each agent
    policy = agent_names[0].split('_')[0]
    plt.title(f'Average Delta Mean')
    plt.ylabel('Delta')
    agent_names = [name for name in agent_names]
    plt.bar(agent_names, np.mean(means, axis=1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'deltas_over_time_bar.png'))
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_incidents_occurred_over_time_across_agents(tot_eval_data, save_dir=None, smoothing_val=None):
    agent_names = list(tot_eval_data.keys())
    aggregated_tot_ep_incidents_missed = [tot_eval_data[name]['tot_incidents_occurred'] for name in agent_names]
    aggregated_tot_ep_incidents_missed = np.stack([np.sum(arr,axis=2) for arr in aggregated_tot_ep_incidents_missed], axis=0)

    timesteps = list(range(len(aggregated_tot_ep_incidents_missed[0][0])))
    num_eps = len(aggregated_tot_ep_incidents_missed[0])

    plt.title(f'Average Incidents Occurred Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Incidents Occurred')

    means = np.mean(aggregated_tot_ep_incidents_missed, axis=1)
    maxs = np.max(aggregated_tot_ep_incidents_missed, axis=1)
    mins = np.min(aggregated_tot_ep_incidents_missed, axis=1) 

    # Plot means and add a legend
    for i in range(len(agent_names)):
        if smoothing_val is not None:
            plt.plot(gaussian_filter1d(means[i],smoothing_val), alpha=0.8)
        else:
            plt.plot(timesteps, means[i], alpha=0.8)
    plt.legend(agent_names, fontsize='x-large')

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'incidents_Occurred_over_time.png'))
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_incidents_missed_over_time_across_agents(tot_eval_data, save_dir=None, smoothing_val=None):
    """
    Plots average incidents missed over time

    Parameters
    ----------
    tot_eval_data: {agent_name: {tot_incidents_missed: np.array(num_episodes, num_timesteps, num_locations)}}
    save_dir: if not None, saves the plot to save_dir. Otherwise, it displays the plot
    smoothing_val: if not None, smooths the graph with a 1d gaussian filter with std=smoothing_val for better visualization

    Notes
    -----
    aggregated_tot_ep_incidents_missed: the incidents missed per timestep per agent, should be {agent_name: tot_incidents_missed} where
                            tot_incidents_missed shape is (num_episodes, num_timesteps, num_locations)
    Returns:
    """
    agent_names = list(tot_eval_data.keys())
    aggregated_tot_ep_incidents_missed = [tot_eval_data[name]['tot_incidents_missed'] for name in agent_names]
    aggregated_tot_ep_incidents_missed = np.stack([np.sum(arr,axis=2) for arr in aggregated_tot_ep_incidents_missed], axis=0)

    timesteps = list(range(len(aggregated_tot_ep_incidents_missed[0][0])))
    num_eps = len(aggregated_tot_ep_incidents_missed[0])

    plt.title(f'Average Incidents Missed Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Incidents Missed')

    means = np.mean(aggregated_tot_ep_incidents_missed, axis=1)
    maxs = np.max(aggregated_tot_ep_incidents_missed, axis=1)
    mins = np.min(aggregated_tot_ep_incidents_missed, axis=1)

    # Plot means and add a legend
    for i in range(len(agent_names)):
        if smoothing_val is not None:
            plt.plot(gaussian_filter1d(means[i],smoothing_val), alpha=0.8)
        else:
            plt.plot(timesteps, means[i], alpha=0.8)
    plt.legend(agent_names, fontsize='x-large')

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'incidents_missed_over_time.png'))
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_incidents_seen_over_time_across_agents(tot_eval_data, save_dir=None, smoothing_val=None):
    """
    Plots incidents seen over time, averaged over number of eval episodes, for each agent

    Parameters
    ----------
    tot_eval_data: {agent_name: {tot_incidents_seen: np.array(num_episodes, num_timesteps, num_locations)}}
    save_dir: if not None, saves the plot to save_dir. Otherwise, it displays the plot
    smoothing_val: if not None, smooths the graph with a 1d gaussian filter with std=smoothing_val for better visualization

    aggregated_tot_ep_incidents_seen: the incidents per timestep per agent, should be {agent_name: tot_incidents_seen} where
                            tot_incidents_seen shape is (num_episodes, num_timesteps, num_locations)
    Returns:
    """
    agent_names = list(tot_eval_data.keys())
    aggregated_tot_ep_incidents_seen = [tot_eval_data[name]['tot_incidents_seen'] for name in agent_names]
    aggregated_tot_ep_incidents_seen = np.stack([np.sum(arr,axis=2) for arr in aggregated_tot_ep_incidents_seen], axis=0)

    timesteps = list(range(len(aggregated_tot_ep_incidents_seen[0][0])))
    num_eps = len(aggregated_tot_ep_incidents_seen[0])

    plt.title(f'Average Incidents Seen Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Incidents Seen')

    means = np.mean(aggregated_tot_ep_incidents_seen, axis=1)
    maxs = np.max(aggregated_tot_ep_incidents_seen, axis=1)
    mins = np.min(aggregated_tot_ep_incidents_seen, axis=1)

    # Plot means and add a legend
    for i in range(len(agent_names)):
        if smoothing_val is not None:
            plt.plot(gaussian_filter1d(means[i],smoothing_val), alpha=0.8)
        else:
            plt.plot(timesteps, means[i], alpha=0.8)
    plt.legend(agent_names, fontsize='x-large')

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'incidents_seen_over_time.png'))
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_rews_over_time(tot_eval_data, save_dir=None, smoothing_val=None):
    """
    Plots average episodic rewards over time, with maxs and mins.

    Parameters
    ----------
    tot_eval_data: {agent_name: {tot_rews: np.array(num_episodes, num_timesteps)}}
    smoothing_val: if not None, smooths the graph with a 1d gaussian filter with std=smoothing_val for better visualization
    save_dir: if not None, saves the plot to save_dir. Otherwise, it displays the plot

    Notes
    -----
    aggregated_tot_ep_rews: the reward per timestep over time per agent, should be {agent_name: tot_rews} where
                            tot_rews shape is (num_episodes, num_timesteps)
    """
    agent_names = list(tot_eval_data.keys())
    aggregated_tot_ep_rews = [tot_eval_data[name]['tot_rews'] for name in agent_names]

    timesteps = list(range(len(aggregated_tot_ep_rews[0][0])))

    # plt.title(f'Reward Over Time', fontsize=20)
    plt.xlabel('Timestep', fontsize=15)
    plt.ylabel('Reward', fontsize=15)

    means = np.mean(aggregated_tot_ep_rews, axis=1)

    # Plot means and add a legend
    for i in range(len(agent_names)):
        name = agent_names[i]
        if smoothing_val is not None:
            plt.plot(gaussian_filter1d(means[i],smoothing_val), alpha=0.8,label=f'{name}', c=cmap[name])
        else:
            plt.plot(timesteps, means[i], alpha=0.8,label=f'{name}', c=cmap[name])
    plt.legend(agent_names, loc=1, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'rew_over_time.png'))
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_rew_terms_over_time_across_agents(tot_eval_data, save_dir=None):
    """
    Plots average episodic reward terms over time, with maxs and mins.

    Parameters
    ----------
    tot_eval_data: {agent_name: {tot_rew_infos: list }}
    save_dir: if not None, saves the plot to save_dir. Otherwise, it displays the plot

    Notes
    -----
    aggregated_tot_infos: the info for reward terms per timestep over time per agent, should be {agent_name: tot_rew_infos}, where
                            tot_rew_infos is a list of lists of dicts, where each dict is the info for a timestep
                            tot_rew_infos "shape" is (num_episodes, num_timesteps, {})
    Returns:
    """
    agent_names = list(tot_eval_data.keys())

    fig, axs = plt.subplots(len(agent_names))
    # fig.tight_layout()
    if len(agent_names) == 1:
        axs = [axs]

    for name, ax in zip(agent_names, axs):
        tot_ep_infos = tot_eval_data[name]['tot_rew_infos']  # (num_episodes, num_timesteps, {})
        timesteps = list(range(len(tot_ep_infos[0])))
        num_eps = len(tot_ep_infos)

        term_names = list(tot_ep_infos[0][0].keys())

        # Extract out term values, might be slightly inefficient... oh wells ;P
        term_vals = {}
        for term_name in term_names:
            mean_term_vals = []
            for i in range(num_eps):
                ep_term_vals = []
                for info in tot_ep_infos[i]:
                    ep_term_vals.append(info[term_name])
                mean_term_vals.append(copy.deepcopy(ep_term_vals))
            term_vals[term_name] = np.mean(mean_term_vals, axis=0)

        ax.set_title(f'{name}')
        # ax.set_xlabel('Timestep')
        ax.set_ylabel('Reward Magnitude')

        for term_name, term_means in list(term_vals.items()):
            ax.plot(timesteps, term_vals[term_name], label=term_name)

        # ax.legend()
    # fig.suptitle('Avg Reward Term Values Over Time')
    plt.xlabel('Timestep')
    plt.subplots_adjust(top=0.85, hspace=0.75)
    plt.legend()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'rew_terms_over_time.png'))
        plt.close()
    else:
        plt.show()
        plt.close()



def plot_true_rates_over_time_across_agents(tot_eval_data, save_dir=None):
    """
    Plots average episodic true rates over time.

    Parameters
    ----------
    tot_eval_data: {agent_name: {tot_true_rates: np.array(num_episodes, num_timesteps, num_locations)}}
    save_dir: if not None, saves the plot to save_dir. Otherwise, it displays the plot

    Notes
    -----
    aggregated_tot_ep_true_rates: the true rates per timestep per site per agent, should be {agent_name: tot_true_rates} where
                            tot_true_rates shape is (num_episodes, num_timesteps, num_locations)
    Returns:
    """
    agent_names = list(tot_eval_data.keys())

    fig, axs = plt.subplots(len(agent_names))
    # fig.tight_layout()
    if len(agent_names) == 1:
        axs = [axs]

    for name, ax in zip(agent_names, axs):
        tot_ep_true_rates = tot_eval_data[name]['tot_true_rates']  # (num_episodes, num_timesteps, num_locations)
        timesteps = list(range(len(tot_ep_true_rates[0])))
        num_eps = len(tot_ep_true_rates)

        ax.set_title(name)
        # ax.set_xlabel('Timestep')
        ax.set_ylabel('True Rates')

        means = np.mean(tot_ep_true_rates, axis=0)

        for i in range(means.shape[-1]):
            ax.plot(timesteps, means[:, i], label=f'Site {i}')

        # ax.legend()
    # fig.suptitle('True Rates Over Time')
    plt.xlabel('Timestep')
    plt.subplots_adjust(top=0.85, hspace=0.75)
    plt.legend()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'true_rates_over_time.png'))
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_dist_disc_over_time(tot_eval_data, save_dir=None):
    agent_names = list(tot_eval_data.keys())

    mean_list = []
    for name in agent_names:
        tot_dist_disc = tot_eval_data[name]['tot_dists']
        means = tot_dist_disc

        timesteps = list(range(tot_dist_disc.shape[1]))
        distance = np.mean(tot_dist_disc, axis=0)
        mean_list.append(distance.mean().item())
        plt.plot(timesteps, distance, label=name, c=cmap[name])

    # plt.title(f'Average Dist Discrapency')
    plt.xlabel('Timestep', fontsize=15)
    plt.ylabel('Long-term Fairness', fontsize=15)
    plt.legend(loc=1,fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'dist_disc_over_time.png'))
        plt.close()
    else:
        plt.show()
        plt.close()

    plt.title(f'Average Dist Discrapency')
    plt.ylabel('Dist_Disc')
    plt.bar(agent_names, mean_list)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'dist_disc_over_time_bar.png'))
        plt.close()
    else:
        plt.show()
        plt.close()
    
