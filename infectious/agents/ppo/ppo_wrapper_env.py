import copy
import torch
import gym
import numpy as np
from gym import spaces

from networkx.algorithms import community

from infectious.environments import core
from infectious.config import EP_TIMESTEPS, ZETA_1, ZETA_0
from geomloss import SamplesLoss
wloss = SamplesLoss("sinkhorn", p=1, blur=0.01)


class PPOEnvWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 reward_fn,
                 ep_timesteps=EP_TIMESTEPS):
        super(PPOEnvWrapper, self).__init__(env)

        self.env = env
        # ----------------------------------- new part --------------------------------
        self.history_actions = []
        self.dist = np.zeros(1, )
        # -----------------------------------------------------------------------------

        shape = self.format_observation(
            self.env.observation_space.sample()).shape

        
        self.observation_space = spaces.Box(
            low=np.inf,
            high=np.inf,
            shape=shape,
        )
        
        self.action_space = spaces.Discrete(
            n=shape[0] + 1,
        )

        self.reward_fn = reward_fn()

        self.timestep = 0
        self.ep_timesteps = ep_timesteps

        communities_generator = community.girvan_newman(self.env.state.population_graph)
        self.communities = tuple(sorted(c) for c in next(communities_generator))
        self.num_communities = len(self.communities)
        # Map individuals in the graph to a community
        self.communities_map = {
            individual: comm_i for comm_i, comm in enumerate(self.communities) for individual in comm
        }
        # ------------------- new part -------------------------
        self.community2action = {0: [], 1: []}
        for individual, comm in self.communities_map.items():
            self.community2action[comm].append(individual)
        # ------------------------------------------------------

        # Keep track of how many vaccines go to each community
        self.num_vaccines_per_community = np.zeros(self.num_communities)
        # Keep track of previous health states to compute newly infected number
        self.prev_health_states = copy.deepcopy(self.env.state.health_states)
        # Newly infected in each community
        self.num_newly_infected_per_community = np.zeros(self.num_communities)

        self.delta = 0
        self.delta_delta = 0
       

    def format_observation(self, obs, padding=2):
        """Formats health state observations into a numpy array.
        The health-states are one-hot encoded as row vectors, and then stacked
        together vertically to create a |population| x |health states| array.
        The population is padded on top and bottom with "recovered" indivduals,
        which don't affect the disease spread but make convolutions simpler.
        Args:
          obs: An observation dictionary.
          padding: An integer indicating how many people to use for padding.
        Returns:
          A numpy array suitable for passing to a DQN agent.
        REMOVED PADDING
        """
        vecs = []
        initial_params = self.env.initial_params
        num_states = len(initial_params.state_names)
        recovered_state = initial_params.state_names.index('recovered')
        for state in obs['health_states']:
            vecs.append(np.zeros((num_states + 1, 1), dtype=float))
            vecs[-1][state] = 1.0

        # ---------- new part --------------------
        for act in self.history_actions:
            vecs[act-1][-1] += 1.0
        return np.hstack(vecs).T
        # ----------------------------------------
        # return np.hstack(vecs).T

    def process_action(self, action):
        if action == self.action_space.n - 1:
            return None
        return np.array([action])

    def reset(self):
        self.timestep = 0
        # Keep track of how many vaccines go to each community
        self.num_vaccines_per_community = np.zeros(self.num_communities)
        # Keep track of previous health states to compute newly infected number
        self.prev_health_states = copy.deepcopy(self.env.state.health_states)
        # Newly infected in each community
        self.num_newly_infected_per_community = np.zeros(self.num_communities)
        self.delta = 0
        self.delta_delta = 0
        # ----------------------------------- new part --------------------------------
        self.history_actions = []
        self.dist = np.zeros(1, )
        # -----------------------------------------------------------------------------

        return self.format_observation(self.env.reset())

    def step(self, action):
        self.prev_health_states = copy.deepcopy(self.env.state.health_states)
        self.history_actions.append(action)

        action = self.process_action(action)
        obs, _, done, info = self.env.step(action)

        # Update the number of vaccines in each community
        if action is not None:
            comm_i = self.communities_map[action[0]]
            self.num_vaccines_per_community[comm_i] += 1
        # Compute newly infected
        for i, (health_state, prev_health_state) in enumerate(zip(self.env.state.health_states, self.prev_health_states)):
            # 1 is the index in self.env.state.params.state_names for infected
            if health_state == 1 and health_state != prev_health_state:
                comm_i = self.communities_map[i]
                self.num_newly_infected_per_community[comm_i] += 1

        r = self.reward_fn(health_states=self.env.state.health_states,
                           num_vaccines_per_community=self.num_vaccines_per_community,
                           num_newly_infected_per_community=self.num_newly_infected_per_community,
                           eta0=ZETA_0,
                           eta1=ZETA_1)
        tmp_r = r

        old_delta = self.delta
        self.delta = self.reward_fn.calc_delta(num_vaccines_per_community=self.num_vaccines_per_community,
                                               num_newly_infected_per_community=self.num_newly_infected_per_community)
        self.delta_delta = self.delta - old_delta

        self.timestep += 1
        if self.timestep == self.ep_timesteps:
            done = True

        # ----------------------------------- new part --------------------------------
        # Long-term Distance
        old_dist = cal_dist(self.prev_health_states, self.communities_map)
        self.dist = cal_dist(self.env.state.health_states, self.communities_map)
        self.dist_dist = self.dist - old_dist
        # -----------------------------------------------------------------------------

        return self.format_observation(obs), r, done, info
    
# ----------------------------------- new part --------------------------------
def cal_dist(states, com_map):
    dists = np.zeros([2, 3])
    for i in range(len(states)):
        dists[com_map[i]][states[i]] += 1
    wd = wloss(torch.tensor(dists[0].reshape(1,-1)), torch.tensor(dists[1].reshape(1,-1)))
    return abs(wd.item())
# -----------------------------------------------------------------------------