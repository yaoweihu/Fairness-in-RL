import os
import torch


MODEL = 'F-PPO' 
########## Experiment Setup Parameters ##########
RESULTS_DIR = './results' 
EXP_DIR = os.path.join(RESULTS_DIR, MODEL)
SAVE_DIR = os.path.join(EXP_DIR, 'models')
EVAL_DIR = os.path.join(RESULTS_DIR,'evaluation')

# PPO model paths to evaluate
EVAL_MODEL_PATHS = {
    'F-PPO': "results/F-PPO/models/final_model",
    # 'A-PPO': "results/A-PPO/models/final_model",
}
    
CPO_EVAL_MODEL_PATHS = {
    # 'CPO': 'results/CPO/models/cpo_agent_ep600',
}

########## Env Parameters ##########
N_LOCATIONS = 5
N_ATTENTION_UNITS = 6
EP_TIMESTEPS = 1000
INCIDENT_RATES = [8, 6, 4, 3, 1.5]
DYNAMIC_RATE = 0.1
DYNAMIC_INCREASE_RATES = [0.2, 0.26, 0.4, 0.6, 0.3]

########## PPO Train Parameters ##########
TRAIN_TIMESTEPS = 2_000_000  # Total train time
LEARNING_RATE = 0.00001
POLICY_KWARGS = dict(activation_fn=torch.nn.ReLU,
                     net_arch = [128, 128, dict(vf=[128, 64], pi=[128, 64])])  # actor-critic architecture
SAVE_FREQ = 250_000  # save frequency in timesteps

WINDOW = 300
SHORT_TERM_WINDOW = WINDOW
# Weights for incidents seen, missed incidents, and delta in reward for the attention allocation environment
ZETA_0 = 1
ZETA_1 = 0.25
ZETA_2 = 0  # 0 means no delta penalty in the reward (should only be non-zero for R-PPO)

SEED = 2023

if MODEL.startswith('A-PPO'):
    REGULARIZE_ADVANTAGE = True  # Regularize advantage?
    BETA_0 = 1
    BETA_1 = 0.15
    BETA_2 = 0.15
    BETA_3 = 0.
    BETA_4 = 0.
    OMEGA = 0.05
    TAU = None
    SHORT_TERM = False
elif MODEL.startswith('PPO'):
    REGULARIZE_ADVANTAGE = False
    BETA_0 = 1
    BETA_1 = 0.
    BETA_2 = 0.
    BETA_3 = 0.
    BETA_4 = 0.
    OMEGA = 0.05
    TAU = None
    SHORT_TERM = False
elif MODEL.startswith('F-PPO-S'):
    REGULARIZE_ADVANTAGE = False
    BETA_0 = 1
    BETA_1 = 0.
    BETA_2 = 0.
    BETA_3 = 0.0
    BETA_4 = 0.0
    OMEGA = 0.1
    TAU = 0.08
    SHORT_TERM = True
elif MODEL.startswith('F-PPO-L'):
    REGULARIZE_ADVANTAGE = True 
    BETA_0 = 1
    BETA_1 = 0.
    BETA_2 = 0.
    BETA_3 = 0.15
    BETA_4 = 0.15
    OMEGA = 0.1
    TAU = None
    SHORT_TERM = False
elif MODEL.startswith('F-PPO'):
    REGULARIZE_ADVANTAGE = True
    BETA_0 = 1
    BETA_1 = 0.
    BETA_2 = 0.
    BETA_3 = 0.15
    BETA_4 = 0.15
    # Threshold for delta
    OMEGA = 0.1
    TAU = 0.08
    SHORT_TERM = True

START_ITERATION = 15 

# Number of timesteps remembered in observation history
OBS_HIST_LEN = 8

########## Eval Parameters ##########
# Weights for incidents seen, missed incidents, and delta in reward for the attention allocation environment
EVAL_ZETA_0 = 1
EVAL_ZETA_1 = 0.25
EVAL_ZETA_2 = 0
# How many timesteps in the past the observation history should include
EVAL_OBS_HIST_LEN = 8


Setting_params = {
    'seed': SEED,
    'window': WINDOW,
    'long-term': [BETA_0, BETA_1, BETA_2, BETA_3, BETA_4],
    'short-term': [OMEGA, TAU, SHORT_TERM],
    'n_locations': N_LOCATIONS,
    'n_attention_units': N_ATTENTION_UNITS,
    'ep_timesteps': EP_TIMESTEPS,
    'incident_rates': INCIDENT_RATES,
    'dynamic_rate': DYNAMIC_RATE,
    'dynamic_increase_rates': DYNAMIC_INCREASE_RATES,
}
