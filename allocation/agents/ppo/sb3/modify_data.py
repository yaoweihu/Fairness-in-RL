import os
import torch
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor

from allocation.config import N_ATTENTION_UNITS, START_ITERATION, TAU, SAVE_DIR


def get_alloc(actions):
    logits_orig = actions.flatten()
    p = np.exp(logits_orig) / np.exp(logits_orig).sum()
    
    n_left = N_ATTENTION_UNITS
    allocs = [0 for _ in range(len(logits_orig))]
    while n_left > 0:
        idx = np.argmax(p)
        allocs[idx] += 1
        n_left -= 1
        p[idx] -= 1 / N_ATTENTION_UNITS

    return np.array(allocs)


def recover_logits(logits, r_idxs, a_idxs, eps):
    num, denom = np.exp(logits), np.exp(logits).sum()
    probs = num / denom
    probs = np.tile(probs, (r_idxs.shape[0],1))

    rows = np.arange(r_idxs.shape[0])
    
    # sigma is the amount of probability mass we are going to move from the action idx we are removing
    # a unit from to the action idx we are adding a unit to
    sigma = probs[rows,r_idxs] - np.maximum(eps, probs[rows,r_idxs] - 1 / N_ATTENTION_UNITS)

    logits_out = np.tile(logits, (r_idxs.shape[0],1))
    logits_out[rows,r_idxs] = np.log(denom) + np.log(probs[rows,r_idxs] - sigma)
    logits_out[rows,a_idxs] = np.log(denom) + np.log(probs[rows,a_idxs] + sigma)
    
    return logits_out


# when we remove or add an attention unit from a location, we need to recompute the logits for those
# locations. if 'l' is the original logit, 'l_p' is the new logit, then we only want to consider
# actions where |P(l) - P(l_p)| <= eps
def find_close_actions(self, logits_orig, r_idxs, min_alloc_prob, thresh):
    with torch.no_grad():
        obs_tensor = obs_as_tensor(self._last_obs, self.device)
        dist = self.policy.get_distribution(obs_tensor)
        log_probs_orig = dist.distribution.log_prob(torch.tensor(logits_orig).to(self.device))

        qual_idxs = []
        qual_logits = []
        
        ar1 = np.array(list(r_idxs))
        ar2 = np.array(list(range(logits_orig.shape[0])))
        ar3 = np.array(np.meshgrid(ar1,ar2)).T.reshape(-1, 2)
        idxs = ar3[np.where(ar3[:,0] != ar3[:,1])]
        rows = np.arange(idxs.shape[0])
        r_idxs = idxs[:,0]
        a_idxs = idxs[:,1]

        logits = recover_logits(logits_orig, r_idxs, a_idxs, min_alloc_prob)
        log_probs = dist.distribution.log_prob(torch.tensor(logits).to(self.device))
        diff = torch.absolute(torch.exp(log_probs_orig) - torch.exp(log_probs))

        qual_idxs = ((diff[rows,r_idxs]<=thresh)*(diff[rows,a_idxs]<=thresh)).nonzero().cpu().numpy().flatten()
        qual_logits = logits[qual_idxs].tolist()
        
        qual_idxs = [(r_idxs[i], a_idxs[i]) for i in qual_idxs]
    
    assert len(qual_idxs) == len(qual_logits)

    return qual_idxs, qual_logits


def modify_data_for_short_term_fairness(self, env, iteration, n_steps, log_probs, actions, counter):
    logits_orig = actions.flatten()
    p = np.exp(logits_orig) / np.exp(logits_orig).sum()
    p_static = np.exp(logits_orig) / np.exp(logits_orig).sum()

    min_alloc_prob = 0.005
    if iteration > START_ITERATION: 
        n_left = N_ATTENTION_UNITS
        r_idxs = set()
        allocs = [0 for _ in range(len(logits_orig))]
        while n_left > 0:
            idx = np.argmax(p)
            # location attn unit candidates above min_alloc_prob
            if p_static[idx] > min_alloc_prob:
                r_idxs.add(idx)

            allocs[idx] += 1
            n_left -= 1
            p[idx] -= 1 / N_ATTENTION_UNITS
        
        qual_idxs, qual_logits = find_close_actions(self, logits_orig, r_idxs, min_alloc_prob, TAU)

        attn_alloc_history = env.get_attr('attn_alloc_history')[0]
        if len(qual_logits) != 0 and attn_alloc_history.sum() != 0:

            attn_alloc_history = np.concatenate((attn_alloc_history[1:], np.expand_dims(allocs, axis=0)))
            orig_delta = env.get_attr('reward_fn')[0].calc_delta(attn_alloc_history)
        
            best_delta = orig_delta

            for (r_idx, a_idx), logits in zip(qual_idxs, qual_logits):
                new_allocs = np.array(allocs)
                new_allocs[r_idx] -= 1
                new_allocs[a_idx] += 1
                attn_alloc_history[-1] = new_allocs
                delta = env.get_attr('reward_fn')[0].calc_delta(attn_alloc_history)
                if delta < best_delta:
                    best_delta = delta
                    actions = np.array(logits).reshape(actions.shape)

            if best_delta < orig_delta:
                with torch.no_grad():
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    values, log_probs, entropy = self.policy.evaluate_actions(obs_tensor, torch.tensor(actions).to(self.device))
                counter += 1

    if n_steps in [100, 500, 1000, 1999]:
        print('iteration:', iteration,'n_steps:', n_steps, 'counter:', counter)
        file_path = os.path.join(SAVE_DIR, 'counter.txt')
        if os.path.exists(file_path):
            with open(file_path, 'a') as f:
                f.write(str(iteration) + " - " + str(n_steps) + " - " + str(counter)  + "\n")
        else:
            with open(file_path, 'w') as f:
                f.write(str(iteration) + " - " + str(n_steps) + " - " + str(counter)  + "\n")

    return actions, log_probs, counter
