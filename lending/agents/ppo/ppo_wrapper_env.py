import gym
import numpy as np
import torch
from gym import spaces
from copy import deepcopy
from collections import deque
from geomloss import SamplesLoss
from lending.config import EP_TIMESTEPS, ZETA_0, ZETA_1, WINDOW

wloss = SamplesLoss("sinkhorn", p=1, blur=0.01)


class PPOEnvWrapper(gym.Wrapper):
  def __init__(self,
               env,
               reward_fn,
               ep_timesteps=EP_TIMESTEPS):
    super(PPOEnvWrapper, self).__init__(env)

    self.observation_space = spaces.Box(
      low=np.inf,
      high=np.inf,
      # (7) OHE of credit score + (2) group +  (2) TPRs of each group
      # shape=(env.observation_space['applicant_features'].shape[0] + 2 * env.state.params.num_groups,),
      # -------------------- add new part ---------------------------
      shape=(3 * env.observation_space['applicant_features'].shape[0] + 2 * env.state.params.num_groups,),
      # shape=(env.observation_space['applicant_features'].shape[0] + env.state.params.num_groups + 1,),
      # -------------------------------------------------------------
    )

    self.action_space = spaces.Discrete(n=2)

    self.env = env
    self.reward_fn = reward_fn()

    self.timestep = 0
    self.ep_timesteps = ep_timesteps

    self.tp = np.zeros(self.env.state.params.num_groups,)
    self.fp = np.zeros(self.env.state.params.num_groups,)
    self.tn = np.zeros(self.env.state.params.num_groups,)
    self.fn = np.zeros(self.env.state.params.num_groups,)
    self.tpr = np.zeros(self.env.state.params.num_groups,)
    self.delta = np.zeros(1, )
    self.old_bank_cash = 0

  def process_observation(self, obs):
    credit_score = obs['applicant_features']
    group = obs['group']
    hist0 = self.history[0]
    hist1 = self.history[1]
    norm = sum(hist0 + hist1) + 1.
    
    return np.concatenate(
      (credit_score,
       group,
       self.tpr,
      # -------------------- add new history features --------------
      hist0 / norm,
      hist1 / norm,
      # -------------------------------------------------------------
       ),
      axis=0
    )

  def compute_tpr(self, tp, fn):
    # tp: true positive, 2-dimensional for 2 groups
    # fn: false negative, 2-dimensional for 2 groups
    return np.divide(
      tp,
      tp + fn,
      out=np.zeros_like(tp),
      where=(tp + fn) != 0)

  def reset(self):
    self.timestep = 0
    self.tp = np.zeros(self.env.state.params.num_groups,)
    self.fp = np.zeros(self.env.state.params.num_groups,)
    self.tn = np.zeros(self.env.state.params.num_groups,)
    self.fn = np.zeros(self.env.state.params.num_groups,)
    self.tpr = np.zeros(self.env.state.params.num_groups,)
    self.delta = np.zeros(1, )
    self.old_bank_cash = 0
    self.delta_delta = 0

    # ----------------------------------- add history and population ----------------------------
    self.history = np.zeros((self.env.state.params.num_groups, self.env.observation_space['applicant_features'].shape[0]))
    self.population = deque(maxlen=WINDOW)
    self.dist = 0
    self.dist_dist = 0
    # --------------------------------------------------------------------------------------------

    return self.process_observation(self.env.reset())

  def step(self, action):
    old_delta = self.delta

    # Update instance variables before we step the environment
    group_id = np.argmax(self.env.state.group)
    if action == 1:
      # Check if agent would default
      if self.env.state.will_default:
        self.fp[group_id] += 1
      else:
        self.tp[group_id] += 1
    elif action == 0:
      if self.env.state.will_default:
        self.tn[group_id] += 1
      else:
        self.fn[group_id] += 1
    self.tpr = self.compute_tpr(tp=self.tp,
                                fn=self.fn)
    self.old_bank_cash = self.env.state.bank_cash

    # Update delta terms
    self.delta = np.abs(self.tpr[0] - self.tpr[1])
    self.delta_delta = self.delta - old_delta

    # ------------------- update population and distribution -----------------
    old_dist = deepcopy(self.dist)

    state_feats = deepcopy(self.env.state.applicant_features)
    state_default = deepcopy(self.env.state.will_default)
    if len(self.population) == WINDOW:
      old_id, old_feats, old_default, old_action = self.population.popleft()
      self.history[old_id] -= old_feats

      if old_action == 1:
        if old_default:
          self.fp[old_id] -= 1
        else:
          self.tp[old_id] -= 1
      elif old_action == 0:
        if old_default:
          self.tn[old_id] -= 1
        else:
          self.fn[old_id] -= 1

    self.population.append((group_id, state_feats, state_default, action))
    self.history[group_id] = self.history[group_id] + state_feats

    # Update dist
    self.dist = wloss(torch.tensor(self.history[0]).view(-1, 1), torch.tensor(self.history[1]).view(-1, 1)).item()
    self.dist_dist = self.dist - old_dist
    # -------------------------------------------------------------------------

    obs, _, done, info = self.env.step(action)

    r = self.reward_fn(old_bank_cash=self.old_bank_cash,
                       bank_cash=self.env.state.bank_cash,
                       tpr=self.tpr,
                       zeta0=ZETA_0,
                       zeta1=ZETA_1)

    self.timestep += 1
    if self.timestep >= self.ep_timesteps:
      done = True

    return self.process_observation(obs), r, done, info