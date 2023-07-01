import argparse
import os
import random
import shutil
import pickle
from pathlib import Path
from copy import deepcopy
from collections import deque

import numpy as np
import torch
import tqdm
from absl import flags
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import sys; sys.path.append('..')


from lending.config import MODEL, CLUSTER_PROBABILITIES, GROUP_0_PROB, BANK_STARTING_CASH, INTEREST_RATE, \
    CLUSTER_SHIFT_INCREMENT, BURNIN, EP_TIMESTEPS, NUM_GROUPS, EVAL_ZETA_0, EVAL_DIR, SEED, WINDOW, \
    EVAL_ZETA_1, TRAIN_TIMESTEPS, SAVE_DIR, EXP_DIR, POLICY_KWARGS, LEARNING_RATE, SAVE_FREQ, EVAL_MODEL_PATHS,\
    EQUALIZE_OPPORTUNITY, MAXIMIZE_REWARD, Setting_params
from lending.agents.human_designed_policies import oracle_lending_agent
from lending.agents.human_designed_policies.classifier_agents import ScoringAgentParams
from lending.agents.human_designed_policies.threshold_policies import ThresholdPolicy
from lending.environments import params, rewards
from lending.environments.lending import DelayedImpactEnv
from lending.environments.lending_params import DelayedImpactParams, two_group_credit_clusters
from lending.environments.rewards import LendingReward
from lending.agents.ppo.ppo_wrapper_env import PPOEnvWrapper
from lending.agents.ppo.sb3.ppo import PPO
from lending.graphing.plot_single import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
torch.cuda.empty_cache()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)


def train(train_timesteps, env):

    exp_exists = False
    if os.path.isdir(SAVE_DIR):
        exp_exists = True
        if input(f'{SAVE_DIR} already exists; do you want to retrain / continue training? (y/n): ') != 'y':
            exit()

        print('Training from start...')

    print('env_params: ', env.state.params)

    env = PPOEnvWrapper(env=env, reward_fn=LendingReward)
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
                    n_steps=EP_TIMESTEPS,
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


def evaluate(env, agent, num_eps, num_timesteps, name, seeds, eval_path, algorithm=None):
    print(f"Evaluating {name}")
    eval_data = {
        'tot_loans': np.zeros((num_eps, NUM_GROUPS)),  # The number of loans per group per episode
        'tot_tp': np.zeros((num_eps, NUM_GROUPS)),  # The number of true positives, or no default given loan accepted, per group per episode
        'tot_fp': np.zeros((num_eps, NUM_GROUPS)),  # The number of false positives, or default given loan accepted, per group per episode
        'tot_tn': np.zeros((num_eps, NUM_GROUPS)),  # The number of true negatives, or default given loan rejected, per group per episode
        'tot_fn': np.zeros((num_eps, NUM_GROUPS)),  # The number of false negatives, or no default given loan rejected, per group per episode
        'tot_tpr': np.zeros((num_eps, NUM_GROUPS)),  # The TPR per group per episode
        'tot_rews_over_time': np.zeros((num_eps, num_timesteps)),  # The reward per timestep per episode
        'tot_loans_over_time': np.zeros((num_eps, num_timesteps,  NUM_GROUPS)),  # The number of loans per group per timestep per episode
        'tot_bank_cash_over_time': np.zeros((num_eps, num_timesteps)),  # The amount of bank cash per timestep per episode
        'tot_tp_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The TP per group per timestep per episode
        'tot_fp_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The FP per group per timestep per episode
        'tot_tn_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The TN per group per timestep per episode
        'tot_fn_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The FN per group per timestep per episode
        'tot_tpr_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The TPR per group per timestep per episode
        # ------------------------------ new part ---------------------------
        'tot_dists': np.zeros((num_eps, num_timesteps, NUM_GROUPS, 7))
        # -------------------------------------------------------------------
    }

    reward_fn = LendingReward()

    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])
        # ------------------------------ new part --------------------------
        env.seed(seeds[ep])
        population = deque(maxlen=WINDOW)
        # -------------------------------------------------------------------

        obs = env.reset()
        done = False
        print(f'Episode {ep}:')
        for t in tqdm.trange(num_timesteps):
            
            # -------------------------------- new part ---------------------------
            if len(population) == WINDOW:
                old_id, old_default, old_action = population.popleft()

                if old_action == 1:
                    if old_default:
                        eval_data['tot_fp'][ep][old_id] -= 1
                    else:
                        eval_data['tot_tp'][ep][old_id] -= 1
                elif old_action == 0:
                    if old_default:
                        eval_data['tot_tn'][ep][old_id] -= 1
                    else:
                        eval_data['tot_fn'][ep][old_id] -= 1

            eval_data['tot_dists'][ep][t][0] = env.state.params.applicant_distribution.components[0].weights
            eval_data['tot_dists'][ep][t][1] = env.state.params.applicant_distribution.components[1].weights
            # ---------------------------------------------------------------------
            
            will_default = env.state.will_default

            action = None
            if algorithm == 'cpo':
                action = int(agent(torch.FloatTensor(obs).squeeze()).sample().item())
            else:
                if isinstance(agent, PPO):
                    action = agent.predict(obs)[0]
                else:
                    action = agent.act(obs, done)

            # Logging
            group_id = np.argmax(env.state.group)
            # --------------------------- new part ----------------------------------
            state_default = deepcopy(env.state.will_default)
            population.append((group_id, state_default, action))
            # ------------------------------------------------------------------------
            # Add to loans if the agent wants to loan
            if action == 1:
                eval_data['tot_loans'][ep][group_id] += 1
                # Check if agent would default
                if env.state.will_default:
                    eval_data['tot_fp'][ep][group_id] += 1
                else:
                    eval_data['tot_tp'][ep][group_id] += 1
            elif action == 0:
                if env.state.will_default:
                    eval_data['tot_tn'][ep][group_id] += 1
                else:
                    eval_data['tot_fn'][ep][group_id] += 1

            # Update TPR for both groups per timestep.
            # In the edge case where denom is 0, set TPR to 0
            eval_data['tot_tpr'][ep] = np.divide(
                eval_data['tot_tp'][ep],
                eval_data['tot_tp'][ep] + eval_data['tot_fn'][ep],
                out=np.zeros_like(eval_data['tot_tp'][ep]),
                where=(eval_data['tot_tp'][ep] + eval_data['tot_fn'][ep])!=0)
            # Update total loans and TPR observed so far
            eval_data['tot_loans_over_time'][ep][t] = eval_data['tot_loans'][ep]
            eval_data['tot_tpr_over_time'][ep][t] = eval_data['tot_tpr'][ep]

            eval_data['tot_tp_over_time'][ep][t] = eval_data['tot_tp'][ep]
            eval_data['tot_fp_over_time'][ep][t] = eval_data['tot_fp'][ep]
            eval_data['tot_tn_over_time'][ep][t] = eval_data['tot_tn'][ep]
            eval_data['tot_fn_over_time'][ep][t] = eval_data['tot_fn'][ep]

            old_bank_cash = env.state.bank_cash

            obs, _, done, _ = env.step(action)

            bank_cash = env.state.bank_cash

            r = reward_fn(old_bank_cash=old_bank_cash,
                          bank_cash=bank_cash,
                          tpr=eval_data['tot_tpr'][ep],
                          zeta0=EVAL_ZETA_0,
                          zeta1=EVAL_ZETA_1)

            eval_data['tot_rews_over_time'][ep][t] = r
            eval_data['tot_bank_cash_over_time'][ep][t] = bank_cash

            if done:
                break

    # ------------------- new part -------------------------
    with open(f'{eval_path}/tot_eval_data.pkl', 'wb') as f:
        pickle.dump(eval_data, f)
    # ------------------------------------------------------

    return eval_data


def plot_figures():
    with open(EVAL_DIR + 'tot_eval_data.pkl', 'rb') as f:
        eval_data = pickle.load(f)

    plot_tpr_gap_over_time(eval_data, EVAL_DIR)
    plot_tpr_over_time(eval_data, EVAL_DIR)
    plot_loans_over_time(eval_data, EVAL_DIR)
    plot_bank_cash_over_time(eval_data, EVAL_DIR)
    plot_confusion_matrix_over_time(eval_data,EVAL_DIR)
    plot_total_rews_over_time(eval_data, EVAL_DIR)
    plot_dist_disc_over_time(eval_data, EVAL_DIR)

    plot_dist(eval_data, 0, EVAL_DIR)
    plot_dist(eval_data, 1000, EVAL_DIR)
    plot_dist(eval_data, 2999, EVAL_DIR)
    plot_dist(eval_data, 5000, EVAL_DIR)
    plot_dist(eval_data, 9999, EVAL_DIR)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'cpo'])
    parser.add_argument('--eval_path', dest='eval_path', type=str, default=EVAL_DIR)
    args = parser.parse_args()

    env_params = DelayedImpactParams(
        applicant_distribution=two_group_credit_clusters(
            cluster_probabilities=CLUSTER_PROBABILITIES,
            group_likelihoods=[GROUP_0_PROB, 1 - GROUP_0_PROB]),
        bank_starting_cash=BANK_STARTING_CASH,
        interest_rate=INTEREST_RATE,
        cluster_shift_increment=CLUSTER_SHIFT_INCREMENT,
    )
    env = DelayedImpactEnv(env_params)
    env.seed(SEED)

    if args.train:
        train(train_timesteps=TRAIN_TIMESTEPS, env=env)
        plot_rets(exp_path=SAVE_DIR, save_png=True)


    if args.eval: 
        # Initialize eval directory to store eval information
        shutil.rmtree(EVAL_DIR, ignore_errors=True)
        Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)

        # Get random seeds
        eval_eps = 10
        eval_timesteps = 10000
        seeds = [random.randint(0, 10000) for _ in range(eval_eps)]

        with open(EVAL_DIR + '/seeds.txt', 'w') as f:
            f.write(str(seeds)+"\n")
            f.write(str(Setting_params))

        if MODEL not in ['EO', 'Greedy']:
            # First, evaluate PPO human_designed_policies
            for name, model_path in EVAL_MODEL_PATHS.items():
                env = DelayedImpactEnv(env_params)
                agent = PPO.load(model_path, verbose=1)
                evaluate(env=PPOEnvWrapper(env=env, reward_fn=LendingReward, ep_timesteps=eval_timesteps),
                        agent=agent,
                        num_eps=eval_eps,
                        num_timesteps=eval_timesteps,
                        name=name,
                        seeds=seeds,
                        eval_path=args.eval_path)
        else:
            # Evaluate threshold policies
            if MODEL == 'EO':
                threshold_policy = EQUALIZE_OPPORTUNITY
            elif MODEL == 'Greedy':
                threshold_policy = MAXIMIZE_REWARD

            env = DelayedImpactEnv(env_params)
            agent_params = ScoringAgentParams(
                feature_keys=['applicant_features'],
                group_key='group',
                default_action_fn=(lambda: 1),
                burnin=BURNIN,
                convert_one_hot_to_integer=True,
                threshold_policy=threshold_policy,
                skip_retraining_fn=lambda action, observation: action == 0,
                cost_matrix=params.CostMatrix(
                    fn=0, fp=-1, tp=env_params.interest_rate, tn=0))

            agent = oracle_lending_agent.OracleThresholdAgent(
                action_space=env.action_space,
                reward_fn=rewards.BinarizedScalarDeltaReward(
                    'bank_cash', baseline=env.initial_params.bank_starting_cash),
                observation_space=env.observation_space,
                params=agent_params,
                env=env)

            evaluate(env=env,
                    agent=agent,
                    num_eps=eval_eps,
                    num_timesteps=eval_timesteps,
                    name=threshold_policy.name,
                    seeds=seeds,
                    eval_path=args.eval_path)



if __name__ == '__main__':
    main()
    plot_figures()