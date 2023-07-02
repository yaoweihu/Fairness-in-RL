# coding=utf-8
# Copyright 2022 The ML Fairness Gym Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Reward functions for ML fairness gym.
These transforms are used to extract scalar rewards from state variables.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Optional

import torch

import numpy as np

from allocation.config import OMEGA
from allocation.environments import core


class NullReward(core.RewardFn):
  """Reward is always 0."""

  # TODO(): Find a better type for observations than Any.
  def __call__(self, observation):
    del observation  # Unused.
    return 0


class ScalarDeltaReward(core.RewardFn):
  """Extracts a scalar reward from the change in a scalar state variable."""

  def __init__(self, dict_key, baseline=0):
    """Initializes ScalarDeltaReward.
    Args:
      dict_key: String key for the observation used to compute the reward.
      baseline: value to consider baseline when first computing reward delta.
    """
    self.dict_key = dict_key
    self.last_val = float(baseline)

  # TODO(): Find a better type for observations than Any.
  def __call__(self, observation):
    """Computes a scalar reward from observation.
    The scalar reward is computed from the change in a scalar observed variable.
    Args:
      observation: A dict containing observations.
    Returns:
      scalar reward.
    Raises:
      TypeError if the observed variable indicated with self.dict_key is not a
        scalar.
    """
    # Validates that the state variable is a scalar with this float() call.
    current_val = float(observation[self.dict_key])
    retval = current_val - self.last_val
    self.last_val = current_val
    return retval


class BinarizedScalarDeltaReward(ScalarDeltaReward):
  """Extracts a binary reward from the sign of the change in a state variable."""

  # TODO(): Find a better type for observations than Any.
  def __call__(self, observation):
    """Computes binary reward from state.
    Args:
      observation: A dict containing observations.
    Returns:
      1 - if the state variable has gone up.
      0 - if the state variable has gone down.
      None - if the state variable has not changed.
    Raises:
      TypeError if the state variable indicated with self.dict_key is not a
        scalar.
    """
    delta = super(BinarizedScalarDeltaReward, self).__call__(observation)
    # Validate that delta is a scalar.
    _ = float(delta)
    if delta == 0:
      return None
    return int(delta > 0)


class VectorSumReward(core.RewardFn):
  """Extracts scalar reward that is the sum of a vector state variable.
  e.g.if state.my_vector = [1, 2, 4, 6], then
  VectorSumReward('my_vector')(state) returns 13.
  """

  def __init__(self, dict_key):
    """Initializes VectorSumReward.
    Args:
      dict_key: String key for the state variable used to compute the reward.
    """
    self.dict_key = dict_key

  # TODO(): Find a better type for observations than Any.
  def __call__(self, observation):
    """Computes scalar sum reward from state.
    Args:
      observation: An observation containing dict_key.
    Returns:
      Scalar sum of the vector observation defined by dict_key.
    Raises:
      ValueError if the dict_key is not in the observation.
    """
    if self.dict_key not in observation:
      raise ValueError("dict_key %s not in observation" % self.dict_key)
    return np.sum(observation[self.dict_key])


class AttentionAllocationReward(core.RewardFn):
  """
  Computes r(s_t) defined in the attention allocation experiments section of the paper
  """
  def __init__(self, dict_key=''):
    # Max number of incidents seen.
    self.max_incidents_seen_sum = -np.inf


  def calc_delta(self, attn_alloc_hist):
    
    total = np.sum(attn_alloc_hist)
    
    loc_total = np.sum(attn_alloc_hist, axis=0)
    loc_proportion = loc_total / total
    
    avg_proportion = np.mean(loc_proportion)
    
    delta = max(np.abs(loc_proportion - avg_proportion))
    
    return delta

  def __call__(self,
               incidents_seen,
               incidents_occurred,
               attn_alloc_hist,
               zeta0,
               zeta1,
               zeta2):

    # Compute incidents missed
    incidents_missed = incidents_occurred - incidents_seen

    incidents_seen_term = zeta0 * np.sum(incidents_seen)
    incidents_missed_term = -zeta1 * np.sum(incidents_missed)
    delta_term = zeta2 * min(0, -self.calc_delta(attn_alloc_hist) + OMEGA)

    # Compute modified reward from attention allocation environment
    tot_rew = incidents_seen_term + incidents_missed_term + delta_term

    self.rew_info = {
      'incidents_seen_term': incidents_seen_term,
      'incidents_missed_term': incidents_missed_term,
      'delta_term': delta_term,
      'tot_rew': tot_rew
    }

    return tot_rew
    

