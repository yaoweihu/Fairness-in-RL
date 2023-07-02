import os
import sys
sys.path.append('..')
import argparse
from pathlib import Path
import pickle
from graphing.plot_all import *


def plot_figures(model_names, draw_path, eval_dir='./results/evaluation'):

    tot_eval_data = {}
    for agent_name in model_names:
        tot_eval_data[agent_name] = {}
        with open(os.path.join(eval_dir, agent_name, 'tot_eval_data.pkl'), 'rb') as f:
            tot_eval_data[agent_name] = pickle.load(f)

    Path(draw_path).mkdir(parents=True, exist_ok=True)
    plot_rews_over_time(tot_eval_data, draw_path, 6)
    plot_deltas_over_time(tot_eval_data, draw_path)
    plot_dist_disc_over_time(tot_eval_data, draw_path)


if __name__ == "__main__":
    result_dir = './results'
    DRAW_PATH = os.path.join(result_dir, 'Summary')
    EVAL_PATH = os.path.join(result_dir, 'evaluation')
    EVAL_MODELS = ['A-PPO', 'F-PPO']
    plot_figures(EVAL_MODELS, DRAW_PATH, EVAL_PATH)