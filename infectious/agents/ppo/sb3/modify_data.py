import os
import copy
import torch
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor
import time
from infectious.environments.rewards import InfectiousReward
reward_fn = InfectiousReward()


def modify_data_for_short_term_fairness(self, env, obs_tensor, iteration, n_steps, actions, log_probs, counter):
    from infectious.config import START_ITERATION, START_THRESH, END_THRESH, RATIO, SAVE_DIR
    
    thresh0 = START_THRESH if iteration < START_ITERATION else min(END_THRESH, START_THRESH * RATIO ** (iteration - START_ITERATION))
    probs = torch.exp(log_probs)
    # if n_steps in [500, 1000, 1500]:
    #     print(probs)

    if probs <= thresh0 and iteration > START_ITERATION:

        community2action = env.get_attr('community2action')[0]
        community = env.get_attr('communities_map')[0]
        community[34] = 2
        num_vaccines_per_community = env.get_attr('num_vaccines_per_community')[0]
        num_newly_infected_per_community = env.get_attr('num_newly_infected_per_community')[0]
        
        short_fairness = [0] * 3
        short_fairness[0] = reward_fn.calc_delta(num_vaccines_per_community=num_vaccines_per_community + [1, 0],
                                    num_newly_infected_per_community=num_newly_infected_per_community)
        short_fairness[1] = reward_fn.calc_delta(num_vaccines_per_community=num_vaccines_per_community + [0, 1],
                                    num_newly_infected_per_community=num_newly_infected_per_community)
        short_fairness[2] = reward_fn.calc_delta(num_vaccines_per_community=num_vaccines_per_community,
                                    num_newly_infected_per_community=num_newly_infected_per_community)
        
        def choose_best_action(candi_group):
            best_action, best_log_prob = None, float('-inf')
            action = torch.FloatTensor([0.]).to(self.device)
            for num in candi_group:
                action[0] = num
                _, log_prob, _ = self.policy.evaluate_actions(obs_tensor, action)
                if log_prob > best_log_prob:
                    best_action = num
                    best_log_prob = log_prob
            return best_action
            
        best_group = np.array(short_fairness).argmin(axis=-1)   
        if best_group != community[actions[0]]:
            with torch.no_grad():
                if best_group == 2:
                    actions = torch.tensor([34]).to(self.device)
                else:
                    group = community2action[best_group]
                    health_group = env.get_attr("state")[0].health_states
                    candidate_group = [g for g in group if health_group[g] == 0]
                    if len(candidate_group) > 0:
                        # action = np.random.choice(candidate_group)
                        action = choose_best_action(candidate_group)
                    else:
                        # action = np.random.choice(community2action[best_group])
                        action = choose_best_action(community2action[best_group])

                    actions = torch.tensor([action]).to(self.device)
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                values, log_probs, entropy = self.policy.evaluate_actions(obs_tensor, actions)

                actions = actions.cpu().numpy()
                counter += 1

    if n_steps in [500, 1000, 1500]:
        print(n_steps, " - ", counter)
        print(thresh0, probs)
        file_path = SAVE_DIR + '/counter.txt'
        if os.path.exists(file_path):
            with open(file_path, 'a') as f:
                f.write(str(iteration) + " - " + str(n_steps) + " - " + str(counter) + " - " + str(thresh0) + "\n")
        else:
            with open(file_path, 'w') as f:
                f.write(str(iteration) + " - " + str(n_steps) + " - " + str(counter) + " - " + str(thresh0) + "\n")
    # print(short_fairness, actions[0], num_vaccines_per_community, num_newly_infected_per_community)
    return actions, log_probs, counter



"""
def modify_data_for_short_term_fairness(self, env, iteration, n_steps, action_probs, actions, log_probs, counter):
    from infectious.config import START_ITERATION, START_THRESH, END_THRESH, RATIO
    
    thresh0 = START_THRESH if iteration < START_ITERATION else min(END_THRESH, START_THRESH * RATIO ** (iteration - START_ITERATION))
    prob = torch.exp(log_probs)
    print(prob, thresh0)

    if prob < thresh0:
        print("sss")
        new_action = compute_short_term_fairness(env, action_probs, actions)
        with torch.no_grad():
            obs_tensor = obs_as_tensor(self._last_obs, self.device)
            actions = torch.tensor([new_action]).to(self.device)
            values, log_probs, entropy = self.policy.evaluate_actions(obs_tensor, actions)
        actions = actions.cpu().numpy()
        counter += 1
    
    if n_steps in [500, 1000, 1500, 1900]:
        print('iteration:', iteration, 'n_steps:', n_steps, 'counter:', counter, 'threshold:', [thresh0])

    return actions, log_probs, counter


def compute_short_term_fairness(env, action_probs, action):
    prev_health_states = copy.deepcopy(env.get_attr('state')[0].health_states)
    communities_map = env.get_attr('communities_map')[0]
    num_vaccines_per_community = env.get_attr('num_vaccines_per_community')[0]
    num_newly_infected_per_community = env.get_attr('num_newly_infected_per_community')[0]
    fairness = []

    for i in range(len(action_probs[0])):
        num_per_com = copy.deepcopy(num_vaccines_per_community)
        num_new_per_com = copy.deepcopy(num_newly_infected_per_community)
        new_env = copy.deepcopy(env)

        act = process_action(i, action_probs)
    
        if act is not None and act[0] < 34:
            obs, _, done, info = new_env.step(act)
            comm_i = communities_map[act[0]]
            num_per_com[comm_i] += 1

        # Compute newly infected
        for i, (health_state, prev_health_state) in enumerate(zip(new_env.get_attr('state')[0].health_states, prev_health_states)):
            # 1 is the index in self.env.state.params.state_names for infected
            if health_state == 1 and health_state != prev_health_state:
                comm_i = communities_map[i]
                num_new_per_com[comm_i] += 1

        delta = reward_fn.calc_delta(num_vaccines_per_community=num_per_com,
                                     num_newly_infected_per_community=num_new_per_com)
        fairness.append(delta)

    inds = []
    min_val = min(fairness)
    for idx, val in enumerate(fairness):
        if val == min_val:
            inds.append(idx)
    min_idx = np.random.choice(inds)
    return min_idx



def process_action(action, action_probs):
    if action == len(action_probs[0]) - 1:
        return None
    return np.array([action])



def modify_data_for_short_term_fairness(self, env, iteration, n_steps, log_probs, actions, counter):
    from infectious.config import START_ITERATION, START_THRESH, END_THRESH, RATIO
    
    thresh0 = START_THRESH if iteration < START_ITERATION else min(END_THRESH, START_THRESH * RATIO ** (iteration - START_ITERATION))
    # thresh1 = 1 - thresh0
    probs = torch.exp(log_probs)
    if n_steps in [500, 1000, 1500]:
        print(probs)

    if probs <= thresh0 and iteration > START_ITERATION:
        # t1 = time.time()
        delta_list = []
        reward_list = []
        new_action = copy.deepcopy(actions)
        for i in range(35):
            new_action[0] = i
            old_env = copy.deepcopy(env)
            old_env.step(new_action)
            delta_list.append(old_env.get_attr('delta')[0])
            reward_list.append(old_env.get_attr('tmp_r')[0])
        
        min_val = min(delta_list[:-1])
        if delta_list[actions[0]] != min_val:
            candidate_ids = [j for j, val in enumerate(delta_list) if val == min_val]
            max_reward = max([reward_list[j] for j in candidate_ids])
            can_ids = [j for j in candidate_ids if reward_list[j] == max_reward]
            new_action = np.random.choice(can_ids)
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions = torch.tensor([new_action]).to(self.device)
                values, log_probs, entropy = self.policy.evaluate_actions(obs_tensor, actions)
            actions = actions.cpu().numpy()
            counter += 1
        # t2 = time.time()
        # print("Steps:", n_steps, "Time:", t2-t1, "Prob:", probs)
    
    if n_steps in [500, 1000, 1500, 1900]:
        print('iteration:', iteration, 'n_steps:', n_steps, 'counter:', counter, 'threshold:', [thresh0])

    return actions, log_probs, counter
"""
        