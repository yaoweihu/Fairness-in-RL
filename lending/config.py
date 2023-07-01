import torch
from lending.agents.human_designed_policies.threshold_policies import ThresholdPolicy


MODEL = 'F-PPO'
########## Experiment Setup Parameters ##########
# EXP_DIR = './experiments/advantage_regularized_ppo/'
EXP_DIR = './results/' + MODEL + '/'
SAVE_DIR = f'{EXP_DIR}/models/'
EVAL_DIR = f'{EXP_DIR}/evaluation/'
# PPO model paths to evaluate
EVAL_MODEL_PATHS = {
    # Sample model evaluation path
    'A-PPO': SAVE_DIR + 'rl_model_700000_steps',
}

########## Env Parameters ##########
DELAYED_IMPACT_CLUSTER_PROBS = (
    (0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.0),
    (0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.0),
)
NUM_GROUPS = 2
GROUP_0_PROB = 0.5
BANK_STARTING_CASH= 10000
INTEREST_RATE = 1
CLUSTER_SHIFT_INCREMENT= 0.01
CLUSTER_PROBABILITIES = DELAYED_IMPACT_CLUSTER_PROBS
EP_TIMESTEPS = 2000
MAXIMIZE_REWARD = ThresholdPolicy.MAXIMIZE_REWARD
EQUALIZE_OPPORTUNITY = ThresholdPolicy.EQUALIZE_OPPORTUNITY
WINDOW = 300

########## PPO Train Parameters ##########
TRAIN_TIMESTEPS = 700_000  # Total train time
LEARNING_RATE = 0.00001
POLICY_KWARGS = dict(activation_fn=torch.nn.ReLU,
                     net_arch = [256, 256, dict(vf=[256, 128], pi=[256, 128])])
SAVE_FREQ = 50000
# Weights for delta bank cash and delta terms in the reward for the lending environment
ZETA_0 = 1
ZETA_1 = 0  # 0 means no delta penalty in the reward (should only be non-zero for R-PPO)

# Weights for advantage, value-thresholding, and decrease-in-violation terms in Eq. 3 of the paper

SEED = 2023
if MODEL == 'A-PPO':
    # 200_000
    REGULARIZE_ADVANTAGE = True
    BETA_0 = 1
    BETA_1 = 0.25
    BETA_2 = 0.25
    BETA_3 = 0.
    BETA_4 = 0.
    # Threshold for delta
    OMEGA = 0.005
elif MODEL == 'PPO':
    # 200_000
    REGULARIZE_ADVANTAGE = False
    BETA_0 = 1
    BETA_1 = 0.
    BETA_2 = 0.
    BETA_3 = 0.
    BETA_4 = 0.
    # Threshold for delta
    OMEGA = 0.005
elif MODEL in ['Greedy', 'EO']:
    REGULARIZE_ADVANTAGE = False
    BETA_0 = 1
    BETA_1 = 0.
    BETA_2 = 0.
    BETA_3 = 0.
    BETA_4 = 0.
    # Threshold for delta
    OMEGA = 0.005
elif MODEL in ['F-PPO']:
    # 700_000
    REGULARIZE_ADVANTAGE = True
    ZETA_0 = 1
    ZETA_1 = 0
    BETA_0 = 1
    BETA_1 = 0.
    BETA_2 = 0.
    BETA_3 = 1.12                  
    BETA_4 = 1.                    
    START_ITERATION = 17
    START_THRESH = 0.5     
    END_THRESH = 0.
    RATIO = 0.985  
    OMEGA = 0.005

elif MODEL in ['F-PPO-L']:
    # 700_000
    REGULARIZE_ADVANTAGE = True
    ZETA_0 = 1
    ZETA_1 = 0
    BETA_0 = 1
    BETA_1 = 0.
    BETA_2 = 0.
    BETA_3 = 1.12                  
    BETA_4 = 1.                    
    START_ITERATION = 17
    START_THRESH = 0.5     
    END_THRESH = 0.
    RATIO = 0.985  
    OMEGA = 0.005

elif MODEL in ['F-PPO-S']:
    # 700_000
    REGULARIZE_ADVANTAGE = True
    ZETA_0 = 1
    ZETA_1 = 0
    BETA_0 = 1
    BETA_1 = 0.
    BETA_2 = 0.
    BETA_3 = 1.12                  
    BETA_4 = 1.                    
    START_ITERATION = 17
    START_THRESH = 0.5     
    END_THRESH = 0.
    RATIO = 0.985  
    OMEGA = 0.005

    

########## Eval Parameters ##########
# Weights for delta bank cash and delta terms in the reward for the lending environment
EVAL_ZETA_0 = 1
EVAL_ZETA_1 = 0
BURNIN = 0  # Number of steps before applying the threshold policy.

if MODEL not in ['F-PPO']:
    START_ITERATION = 0
    START_THRESH = 0.    
    END_THRESH = 0.      
    RATIO = 0.

Setting_params = {
    'seed': SEED,
    'window': WINDOW,
    'long-term': [BETA_0, BETA_1, BETA_2, BETA_3, BETA_4],
    'shor-term': [START_ITERATION, START_THRESH, END_THRESH, RATIO, OMEGA]
}

