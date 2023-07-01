import sys
sys.path.append('..')
import pickle
from pathlib import Path
from graphing.plot_all import *


def plot_figures(model_names):
    datas = []
    for model in model_names:
        EVAL_PATH = './results/' + model +'/evaluation/tot_eval_data.pkl'
        with open(EVAL_PATH, 'rb') as f:
            datas.append(pickle.load(f))

    DRAW_PATH = './results/Summary/'
    Path(DRAW_PATH).mkdir(parents=True, exist_ok=True)
    plot_tpr_gap_over_time(model_names, datas, DRAW_PATH)
    plot_bank_cash_over_time(model_names, datas, DRAW_PATH)
    plot_loans_over_time(model_names, datas, DRAW_PATH)
    plot_dist_disc_over_time(model_names, datas, DRAW_PATH)


if __name__ == "__main__":

    EVAL_MODELS = ['EO', 'PPO', 'A-PPO', 'F-PPO']
    # EVAL_MODELS = ["F-PPO", "F-PPO-S", "F-PPO-L"]
    plot_figures(EVAL_MODELS)