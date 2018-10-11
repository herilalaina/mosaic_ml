import warnings
import time

# Mosaic library
from mosaic.mosaic import Search
from mosaic_ml.evaluator import evaluate
from mosaic_ml.utils import balanced_accuracy


# Metric
from sklearn.metrics import roc_auc_score

# Config space
from ConfigSpace.read_and_write import pcs


from functools import partial

class AutoML():
    def __init__(self, time_budget = None, time_limit_for_evaluation = None, training_log_file = "", info_training = {}, n_jobs = 1):
        self.time_budget = time_budget
        self.time_limit_for_evaluation = time_limit_for_evaluation
        self.training_log_file = training_log_file
        self.info_training = info_training
        self.n_jobs = n_jobs

        # Load config space file
        self.config_space = cs = pcs.read(open("./mosaic_ml/model_config/1_0.pcs", "r"))


    def fit(self, X, y):
        eval_func = partial(evaluate, X=X, y=y, info = self.info_training, score_func=roc_auc_score)

        self.searcher = Search(eval_func=eval_func, config_space=self.config_space)
        self.searcher.run(nb_simulation = 100000000000, generate_image_path = self.info_training["images_directory"])
