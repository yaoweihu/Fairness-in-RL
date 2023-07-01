import torch

MODEL = 'F-PPO'
########## Experiment Setup Parameters ##########
EXP_DIR = './results/' + MODEL +'/'
SAVE_DIR = f'{EXP_DIR}/models/'
EVAL_DIR = f'{EXP_DIR}/evaluation/'

EVAL_FILE = f'rl_model_500000_steps'
# PPO model paths to evaluate
EVAL_MODEL_PATHS = {
    # Sample model evaluation path
    'A-PPO': SAVE_DIR + EVAL_FILE,
}

########## Env Parameters ##########
INFECTION_PROBABILITY =  0.5
INFECTED_EXIT_PROBABILITY = 0.005
NUM_TREATMENTS = 1
BURNIN = 1
GRAPH_NAME = 'karate'
EP_TIMESTEPS = 20  # Number of steps in the experiment.

########## PPO Train Parameters ##########
TRAIN_TIMESTEPS = 500_000  # Total train time
LEARNING_RATE = 0.00001
POLICY_KWARGS = dict(activation_fn=torch.nn.ReLU,
                     net_arch = [512, 512, dict(vf=[512, 256], pi=[512, 256])])
SAVE_FREQ = 50000
# Threshold for delta
OMEGA = 0.05
# Weights for percent healthy and delta term in reward
ZETA_0 = 1
ZETA_1 = 0

SEED = 2023
if MODEL == 'A-PPO':
    REGULARIZE_ADVANTAGE = True  # Regularize advantage?
    # Weights for advantage, value-thresholding, and decrease-in-violation terms in Eq. 3 of the paper
    BETA_0 = 1
    BETA_1 = 0.1
    BETA_2 = 0.1
    BETA_3 = 0.
    BETA_4 = 0.
elif MODEL == 'PPO':
    REGULARIZE_ADVANTAGE = False
    BETA_0 = 1
    BETA_1 = 0.
    BETA_2 = 0.
    BETA_3 = 0.
    BETA_4 = 0.
elif MODEL in ['Random', 'Max']:
    REGULARIZE_ADVANTAGE = False
    BETA_0 = 1
    BETA_1 = 0.
    BETA_2 = 0.
    BETA_3 = 0.
    BETA_4 = 0.
elif MODEL == 'F-PPO':
    REGULARIZE_ADVANTAGE = True
    BETA_0 = 1             
    BETA_1 = 0.
    BETA_2 = 0.
    BETA_3 = 0.25
    BETA_4 = 0.25
    START_ITERATION = 50    
    START_THRESH = 0.01     
    END_THRESH = 0.35        
    RATIO = 1.2 #1.05 
elif MODEL == 'F-PPO-L':
    REGULARIZE_ADVANTAGE = True
    BETA_0 = 1             
    BETA_1 = 0.
    BETA_2 = 0.
    BETA_3 = 0.25
    BETA_4 = 0.25
    START_ITERATION = 50    
    START_THRESH = 0.01     
    END_THRESH = 0.35        
    RATIO = 1.2 #1.05            


########## Eval Parameters ##########
# Weights for percent healthy term and delta term in reward
EVAL_ZETA_0 = 1
EVAL_ZETA_1 = 0


if MODEL not in ['F-PPO']:
    START_ITERATION = 0
    START_THRESH = 0.    
    END_THRESH = 0.      
    RATIO = 0.

Setting_params = {
    'seed': SEED,
    'file': EVAL_FILE,
    'long-term': [BETA_0, BETA_1, BETA_2, BETA_3, BETA_4],
    'shor-term': [START_ITERATION, START_THRESH, END_THRESH, RATIO, OMEGA]
}
