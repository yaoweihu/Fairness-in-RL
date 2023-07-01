import argparse
import copy
import os
import random
import shutil
from pathlib import Path
import sys
sys.path.append('..')
import networkx as nx
import numpy as np
import torch
import tqdm
import pickle
from absl import flags
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from networkx.algorithms import community

from infectious.agents.human_designed_policies import infectious_disease_agents
from infectious.agents.human_designed_policies.infectious_disease_agents import CentralityAgent, RandomAgent, MaxNeighborsAgent
from infectious.config import Setting_params, MODEL, SEED, INFECTION_PROBABILITY, INFECTED_EXIT_PROBABILITY, NUM_TREATMENTS, BURNIN, \
    EVAL_DIR, SAVE_DIR, EXP_DIR, POLICY_KWARGS, LEARNING_RATE, SAVE_FREQ, TRAIN_TIMESTEPS, GRAPH_NAME, EVAL_ZETA_1, EVAL_ZETA_0, EVAL_MODEL_PATHS
from infectious.environments import infectious_disease, rewards
from infectious.environments.rewards import InfectiousReward, calc_percent_healthy
from infectious.graphing.plot_single import *
from infectious.agents.ppo.ppo_wrapper_env import PPOEnvWrapper
from infectious.agents.ppo.sb3.ppo import PPO



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)

GRAPHS = {'karate': nx.karate_club_graph()}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def train(train_timesteps, env):

    exp_exists = False
    if os.path.isdir(SAVE_DIR):
        exp_exists = True
        if input(f'{SAVE_DIR} already exists; do you want to retrain / continue training? (y/n): ') != 'y':
            exit()

        print('Training from start...')

    print('env_params: ', env.state.params)

    env = PPOEnvWrapper(env=env, reward_fn=InfectiousReward)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    model = None
    should_load = False
    if exp_exists:
        resp = input(f'\nWould you like to load the previous model to continue training? If you do not select yes, you will start a new training. (y/n): ')
        if resp != 'y' and resp != 'n':
            exit('Invalid response for resp: ' + resp)
        should_load = resp == 'y'

    if should_load:
        model_name = input(f'Specify the model you would like to load in. Do not include the .zip: ')
        model = PPO.load(EXP_DIR + "models/" + model_name, verbose=1, device=device)
        model.set_env(env)
    else:
        model = PPO("MlpPolicy", env,
                    policy_kwargs=POLICY_KWARGS,
                    verbose=1,
                    learning_rate=LEARNING_RATE,
                    device=device)

        shutil.rmtree(EXP_DIR, ignore_errors=True)
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

        checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=SAVE_DIR,
                                                name_prefix='rl_model')

        model.set_logger(configure(folder=SAVE_DIR))
        model.learn(total_timesteps=train_timesteps, callback=checkpoint_callback)
        model.save(SAVE_DIR + '/final_model')

    # Once we finish learning, plot the returns over time and save into the experiments directory
    plot_rets(SAVE_DIR)
    print("Finish Training")


def cal_dist(states, com_map):
    dists = np.zeros([2, 3])
    for i in range(len(states)):
        dists[com_map[i]][states[i]] += 1
    return dists


def evaluate(env, agent, num_eps, num_timesteps, seeds, eval_path):
    print()

    communities_generator = community.girvan_newman(env.state.population_graph)
    communities = tuple(sorted(c) for c in next(communities_generator))
    num_communities = len(communities)
    # Map individuals in the graph to a community
    communities_map = {
        individual: comm_i for comm_i, comm in enumerate(communities) for individual in comm
    }

    eval_data = {
        'tot_rews_over_time': np.zeros((num_eps, num_timesteps)),  # The reward per timestep per episode
        'tot_percent_sick_over_time': np.zeros((num_eps, num_timesteps)),  # The percentage of sick per timestep per episode
        'tot_percent_healthy_over_time': np.zeros((num_eps, num_timesteps)),  # The percentage of sick per timestep per episode
        'tot_deltas_over_time': np.zeros((num_eps, num_timesteps)),  # The delta per timestep per episode
        'tot_dists': np.zeros((num_eps, num_timesteps, 2, 3))
    }

    reward_fn = InfectiousReward()
    
    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])
        env.seed(seeds[ep])

        # Keep track of how many vaccines go to each community
        num_vaccines_per_community = np.zeros(num_communities)
        # Newly infected in each community
        num_newly_infected_per_community = np.zeros(num_communities)

        obs = env.reset()
        done = False

        print(f'Episode {ep}:')
        for t in tqdm.trange(num_timesteps):
            prev_health_states = copy.deepcopy(env.state.health_states)

            eval_data['tot_dists'][ep][t] = cal_dist(prev_health_states, communities_map)

            action = None
            a = None  # action placeholder for PPO after processing it
            if isinstance(agent, PPO):
                action = agent.predict(obs)[0]
                a = env.process_action(action)
            else:
                action = agent.act(obs, done)
                a = action


            obs, _, done, _ = env.step(action)

            # Update the number of vaccines in each community
            if a is not None:
                comm_i = communities_map[np.array([a]).flatten()[0]]
                num_vaccines_per_community[comm_i] += 1
            # Compute newly infected
            for i, (health_state, prev_health_state) in enumerate(
                    zip(env.state.health_states, prev_health_states)):
                # 1 is the index in self.env.state.params.state_names for infected
                if health_state == 1 and health_state != prev_health_state:
                    comm_i = communities_map[i]
                    num_newly_infected_per_community[comm_i] += 1

            r = reward_fn(health_states=env.state.health_states,
                          num_vaccines_per_community=num_vaccines_per_community,
                          num_newly_infected_per_community=num_newly_infected_per_community,
                          eta0=EVAL_ZETA_0,
                          eta1=EVAL_ZETA_1)


            percent_healthy = calc_percent_healthy(env.state.health_states)
            eval_data['tot_rews_over_time'][ep][t] = r
            eval_data['tot_percent_sick_over_time'][ep][t] = 1 - percent_healthy
            eval_data['tot_percent_healthy_over_time'][ep][t] = percent_healthy
            eval_data['tot_deltas_over_time'][ep][t] = reward_fn.calc_delta(num_vaccines_per_community=num_vaccines_per_community,
                                                                            num_newly_infected_per_community=num_newly_infected_per_community)

            if done:
                break

    with open(f'{eval_path}/tot_eval_data.pkl', 'wb') as f:
        pickle.dump(eval_data, f)

    return eval_data

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True)
    parser.add_argument('--eval', default=True)
    args = parser.parse_args()

    graph = GRAPHS[GRAPH_NAME]
    # Randomly initialize a node to infected
    initial_health_state = [0 for _ in range(graph.number_of_nodes())]
    initial_health_state[0] = 1
    env = infectious_disease.build_sir_model(
        population_graph=graph,
        infection_probability=INFECTION_PROBABILITY,
        infected_exit_probability=INFECTED_EXIT_PROBABILITY,
        num_treatments=NUM_TREATMENTS,
        max_treatments=1,
        burn_in=BURNIN,
        # Treatments turn susceptible people into recovered without having them
        # get sick.
        treatment_transition_matrix=np.array([[0, 0, 1],
                                              [0, 1, 0],
                                              [0, 0, 1]]),
        initial_health_state = copy.deepcopy(initial_health_state)
    )
    env.seed(SEED)

    if args.train:
        train(train_timesteps=TRAIN_TIMESTEPS, env=env)
        plot_rets(exp_path=SAVE_DIR)

    if args.eval:
        # Initialize eval directory to store eval information
        shutil.rmtree(EVAL_DIR, ignore_errors=True)
        Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)

        # Get random seeds
        eval_eps = 200
        eval_timesteps = 20
        seeds = [random.randint(0, 10000) for _ in range(eval_eps)]

        with open(EVAL_DIR + '/seeds.txt', 'w') as f:
            f.write(str(seeds) + "\n")
            f.write(str(Setting_params))


        if MODEL not in ['RANDOM', 'MAX']:
            # First, evaluate PPO human_designed_policies
            for name, model_path in EVAL_MODEL_PATHS.items():
                # Set up agent render directory
                agent = PPO.load(model_path, verbose=1)
                evaluate(env=PPOEnvWrapper(env=env, reward_fn=InfectiousReward, ep_timesteps=eval_timesteps),
                        agent=agent,
                        num_eps=eval_eps,
                        num_timesteps=eval_timesteps,
                        seeds=seeds,
                        eval_path=EVAL_DIR)
        else:
            if MODEL == 'RANDOM':
                agent_class = RandomAgent
            if MODEL == 'MAX':
                agent_class = MaxNeighborsAgent
            
            agent = agent_class(
                env.action_space,
                rewards.NullReward(),
                env.observation_space,
                params=infectious_disease_agents.env_to_agent_params(env.initial_params)
            )
            evaluate(env=env,
                     agent=agent,
                     num_eps=eval_eps,
                     num_timesteps=eval_timesteps,
                     seeds=seeds,
                    eval_path=EVAL_DIR)


def plot_figures():
    with open(EVAL_DIR + 'tot_eval_data.pkl', 'rb') as f:
        eval_data = pickle.load(f)

    plot_delta_over_time(eval_data, EVAL_DIR)
    plot_rews_over_time(eval_data, EVAL_DIR)
    plot_dist_disc_over_time(eval_data, EVAL_DIR)
 

if __name__ == '__main__':
    main()
    plot_figures()