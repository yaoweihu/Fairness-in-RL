import gym
import numpy as np
import torch
from gym import spaces
# from collections import deque
from copy import deepcopy


from allocation.config import EP_TIMESTEPS, OBS_HIST_LEN, ZETA_0, ZETA_1, ZETA_2, WINDOW, \
  SHORT_TERM_WINDOW


class PPOEnvWrapper(gym.Wrapper):
  def __init__(self,
               env,
               reward_fn,
               ep_timesteps=EP_TIMESTEPS,):
    super(PPOEnvWrapper, self).__init__(env)

    # It's best to turn the observations into a single vector in box space, rather than dict.
    self.observation_space = spaces.Box(
      low=-np.inf,
      high=np.inf,
      shape=(env.state.params.n_locations * 4 * OBS_HIST_LEN,),
      dtype=np.float32
    )

    self.action_space = spaces.Box(
      low=-3,
      high=3,
      shape=(env.state.params.n_locations,),
      dtype=np.float64
    )

    self.env = env
    self.reward_fn = reward_fn()

    self.ep_timesteps = ep_timesteps
    self.timestep = 0

    self.ep_deltas = []
    self.ep_delta_deltas = []
    self.ep_long_deltas = []
    self.ep_long_delta_deltas = []

    # --------------------- new part -------------------------
    self.ep_incidents_seen = np.zeros((WINDOW, self.env.state.params.n_locations))
    self.ep_incidents_occurred = np.zeros((WINDOW, self.env.state.params.n_locations))
    # --------------------------------------------------------

    # Observation history of incidents seen, occurred, attention allocated, delta terms per site
    self.observation_history = np.zeros((OBS_HIST_LEN, self.env.state.params.n_locations * 4))

    self.attn_alloc_history = np.zeros((SHORT_TERM_WINDOW, self.env.state.params.n_locations))
    self.delta = 0  # The delta term
    self.delta_delta = 0  # The delta(s') - delta(s) part of the decrease-in-violation term 

  def reset(self):
    self.timestep = 0
    self.observation_history = np.zeros((OBS_HIST_LEN, self.env.state.params.n_locations * 4))
    self.delta = 0
    self.delta_delta = 0

    # ----------------------------------- new part -------------------------------------------
    self.ep_incidents_seen = np.zeros((WINDOW, self.env.state.params.n_locations))
    self.ep_incidents_occurred = np.zeros((WINDOW, self.env.state.params.n_locations))

    self.attn_alloc_history = np.zeros((SHORT_TERM_WINDOW, self.env.state.params.n_locations))

    self.history = np.zeros(self.env.state.params.n_locations)
    self.dist = 0
    self.dist_dist = 0

    self.ep_deltas = []
    self.ep_delta_deltas = []
    self.ep_long_deltas = []
    self.ep_long_delta_deltas = []
    # --------------------------------------------------------------------------------------------

    _ = self.env.reset()

    return self.observation_history.flatten()

  def step(self, action, process=True):

    if process == True:
      action = self.process_action(action)
    
    obs, reward, done, info = self.env.step(action)

    # ------------------------------- new part -------------------------------
    old_dist = deepcopy(self.dist)
    
    self.history += self.env.state.incidents_occurred
    if self.timestep >= WINDOW:
      self.history -= self.ep_incidents_occurred[0]

    self.dist = max(abs(self.history - self.history.mean())) / max(self.history)
    self.dist_dist = self.dist - old_dist

    self.ep_incidents_seen = np.concatenate((self.ep_incidents_seen[1:], np.expand_dims(self.env.state.incidents_seen, axis=0)))
    self.ep_incidents_occurred = np.concatenate((self.ep_incidents_occurred[1:], np.expand_dims(self.env.state.incidents_occurred, axis=0)))
    # -----------------------------------------------------------------------

    self.timestep += 1

    if self.timestep >= self.ep_timesteps:
      done = True

    # Form observation history
    deltas = np.sum(self.ep_incidents_seen, axis=0) / (np.sum(self.ep_incidents_occurred, axis=0) + 1)
    current_obs = np.concatenate((self.env.state.incidents_seen, self.env.state.incidents_occurred, action, deltas), axis=0)

    self.observation_history = np.concatenate((self.observation_history[1:], np.expand_dims(current_obs, axis=0)))
    obs = self.observation_history.flatten()

    # Store custom reward
    self.attn_alloc_history = np.concatenate((self.attn_alloc_history[1:], np.expand_dims(action, axis=0)))
    reward = self.reward_fn(incidents_seen=self.env.state.incidents_seen,
                          incidents_occurred=self.env.state.incidents_occurred,
                          attn_alloc_hist=self.attn_alloc_history,
                          zeta0=ZETA_0,
                          zeta1=ZETA_1,
                          zeta2=ZETA_2)
    
    # Update delta terms
    old_delta = self.delta

    self.delta = self.reward_fn.calc_delta(self.attn_alloc_history)
    
    self.delta_delta = self.delta - old_delta

    return obs, reward, done, info

  def process_action(self, action):
    """
    Convert actions from logits to attention allocation over sites through the following steps:
    1. Convert logits vector into a probability distribution using softmax
    2. Convert probability distribution into allocation distribution with multinomial distribution
    Args:
      action: n_locations vector of logits
    Returns: n_locations vector of allocations
    """
    action = torch.tensor(action)
    probs = torch.nn.functional.softmax(action, dim=-1)
    allocs = [0 for _ in range(self.env.state.params.n_locations)]
    n_left = self.env.state.params.n_attention_units
    
    while n_left > 0:
      ind = np.argmax(probs)
      allocs[ind] += 1
      n_left -= 1
      probs[ind] -= 1 / self.env.state.params.n_attention_units

    action = np.array(allocs)
    return action
  


























#   import inspect
# import gym
# import numpy as np
# import torch
# from gym import spaces
# from collections import deque
# from copy import deepcopy

# from allocation.config import EP_TIMESTEPS, OBS_HIST_LEN, ZETA_0, ZETA_1, ZETA_2, WINDOW


# class PPOEnvWrapper(gym.Wrapper):
#   def __init__(self,
#                env,
#                reward_fn,
#                ep_timesteps=EP_TIMESTEPS):
#     super(PPOEnvWrapper, self).__init__(env)

#     # It's best to turn the observations into a single vector in box space, rather than dict.
#     self.observation_space = spaces.Box(
#       low=-np.inf,
#       high=np.inf,
#       shape=(env.state.params.n_locations * 4 * OBS_HIST_LEN,),
#       dtype=np.float32
#     )

#     self.action_space = spaces.Box(
#       low=-3,
#       high=3,
#       shape=(env.state.params.n_locations,),
#       dtype=np.float64
#     )

#     self.frame_list = []
#     self.env = env
#     self.reward_fn = reward_fn()

#     self.ep_timesteps = ep_timesteps
#     self.timestep = 0
#     self.reset_count =0
#     # --------------------- new part -------------------------
#     # self.ep_incidents_seen = np.zeros((self.ep_timesteps, self.env.state.params.n_locations))
#     # self.ep_incidents_occurred = np.zeros((self.ep_timesteps, self.env.state.params.n_locations))
    
#     self.ep_incidents_seen = np.zeros((WINDOW, self.env.state.params.n_locations))
#     self.ep_incidents_occurred = np.zeros((WINDOW, self.env.state.params.n_locations))
#     # --------------------------------------------------------

#     # Observation history of incidents seen, occurred, attention allocated, delta terms per site
#     self.observation_history = np.zeros((OBS_HIST_LEN, self.env.state.params.n_locations * 4))

#     self.delta = 0  # The delta term
#     self.delta_delta = 0  # The delta(s') - delta(s) part of the decrease-in-violation term of Eq. 3 from the paper, where s' is the consecutive state of s


#   def reset(self):
#     self.timestep = 0
#     # self.ep_incidents_seen = np.zeros((self.ep_timesteps, self.env.state.params.n_locations))
#     # self.ep_incidents_occurred = np.zeros((self.ep_timesteps, self.env.state.params.n_locations))
#     self.observation_history = np.zeros((OBS_HIST_LEN, self.env.state.params.n_locations * 4))
#     self.delta = 0
#     self.delta_delta = 0

#     self.reset_count += 1
#     # ----------------------------------- new part -------------------------------------------
#     self.ep_incidents_seen = np.zeros((WINDOW, self.env.state.params.n_locations))
#     self.ep_incidents_occurred = np.zeros((WINDOW, self.env.state.params.n_locations))
#     self.frame_list.append(inspect.currentframe())
#     if self.reset_count > 2:
#       import pdb; pdb.set_trace()

#     self.history = np.zeros(self.env.state.params.n_locations)
#     self.dist = 0
#     self.dist_dist = 0
#     # --------------------------------------------------------------------------------------------

#     _ = self.env.reset()

#     return self.observation_history.flatten()

#   def step(self, action, process=True, n_step=None):

#     if process == True:
#       action = self.process_action(action)

#     obs, reward, done, info = self.env.step(action)
#     # self.ep_incidents_seen[self.timestep] = self.env.state.incidents_seen
#     # self.ep_incidents_occurred[self.timestep] = self.env.state.incidents_occurred

#     # ------------------------------- new part -------------------------------
#     old_dist = deepcopy(self.dist)
    
#     self.history += self.env.state.incidents_occurred
#     if self.timestep >= WINDOW:
#       self.history -= self.ep_incidents_occurred[0]

#     self.dist = max(abs(self.history - self.history.mean()))
#     self.dist_dist = self.dist - old_dist
#     # import pdb; pdb.set_trace()
#     self.ep_incidents_seen = np.concatenate((self.ep_incidents_seen[1:], np.expand_dims(self.env.state.incidents_seen, axis=0)))
#     self.ep_incidents_occurred = np.concatenate((self.ep_incidents_occurred[1:], np.expand_dims(self.env.state.incidents_occurred, axis=0)))
#     # -----------------------------------------------------------------------

#     self.timestep += 1

#     if self.timestep >= self.ep_timesteps:
#       done = True

#     # Form observation history
#     deltas = np.sum(self.ep_incidents_seen, axis=0) / (np.sum(self.ep_incidents_occurred, axis=0) + 1)
#     current_obs = np.concatenate((self.env.state.incidents_seen, self.env.state.incidents_occurred, action, deltas), axis=0)

#     self.observation_history = np.concatenate((self.observation_history[1:], np.expand_dims(current_obs, axis=0)))
#     obs = self.observation_history.flatten()

#     # if n_step is not None and n_step > 350:
#     #   import pdb; pdb.set_trace()

#     # Store custom reward
#     reward = self.reward_fn(incidents_seen=self.env.state.incidents_seen,
#                             incidents_occurred=self.env.state.incidents_occurred,
#                             ep_incidents_seen=self.ep_incidents_seen,
#                             ep_incidents_occurred=self.ep_incidents_occurred,
#                             zeta0=ZETA_0,
#                             zeta1=ZETA_1,
#                             zeta2=ZETA_2)

#     # Update delta terms
#     old_delta = self.delta
#     self.delta = self.reward_fn.calc_delta(self.ep_incidents_seen, self.ep_incidents_occurred)
#     self.delta_delta = self.delta - old_delta

#     return obs, reward, done, info

#   def process_action(self, action):
#     """
#     Convert actions from logits to attention allocation over sites through the following steps:
#     1. Convert logits vector into a probability distribution using softmax
#     2. Convert probability distribution into allocation distribution with multinomial distribution
#     Args:
#       action: n_locations vector of logits
#     Returns: n_locations vector of allocations
#     """
#     action = torch.tensor(action)
#     probs = torch.nn.functional.softmax(action, dim=-1)
#     allocs = [0 for _ in range(self.env.state.params.n_locations)]
#     n_left = self.env.state.params.n_attention_units
    
#     while n_left > 0:
#       ind = np.argmax(probs)
#       allocs[ind] += 1
#       n_left -= 1
#       probs[ind] -= 1 / self.env.state.params.n_attention_units

#     action = np.array(allocs)
#     return action