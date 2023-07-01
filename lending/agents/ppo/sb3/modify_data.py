import torch
import copy
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor


def modify_data_for_short_term_fairness(self, env, iteration, n_steps, log_probs, actions, counter):
    from lending.config import START_ITERATION, START_THRESH, END_THRESH, RATIO

    thresh0 = START_THRESH if iteration < START_ITERATION else max(END_THRESH, START_THRESH * RATIO ** (iteration - START_ITERATION))
    thresh1 = 1 - thresh0
    probs = torch.exp(log_probs)

    if thresh0 <= probs <= thresh1 and iteration > START_ITERATION:
        default = env.get_attr('state')[0].will_default
        group_id = np.argmax(env.get_attr('state')[0].group)
        tp, fn = copy.deepcopy(env.get_attr('tp')[0]), copy.deepcopy(env.get_attr('fn')[0])
        ctp, cfn = copy.deepcopy(env.get_attr('tp')[0]), copy.deepcopy(env.get_attr('fn')[0])
        
        if actions[0] == 1 and not default:
            tp[group_id] += 1
            cfn[group_id] += 1
        elif actions[0] == 0 and not default:
            fn[group_id] += 1
            ctp[group_id] += 1

        tpr = env.get_attr('compute_tpr')[0](tp, fn)
        ctpr = env.get_attr('compute_tpr')[0](ctp, cfn)
        delta = np.abs(tpr[0] - tpr[1])
        cdelta = np.abs(ctpr[0] - ctpr[1])
        if delta >= cdelta:
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions = torch.tensor(1 - actions).to(self.device)
                values, log_probs, entropy = self.policy.evaluate_actions(obs_tensor, actions)
            actions = actions.cpu().numpy()
            counter += 1

    if n_steps in [500, 1000, 1999]:
        print('iteration:', iteration, 'counter:', counter, 'threshold:', [thresh0, thresh1])

    return actions, log_probs, counter