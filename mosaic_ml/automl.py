# Mosaic library
from mosaic.mosaic import Search
from mosaic_ml.evaluator import evaluate, test_function

# pynisher
import pynisher

# scipy
from scipy.sparse import issparse
import numpy as np

# Metric
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score

# Config space
from ConfigSpace.read_and_write import pcs

import os
from functools import partial

class AutoML():
    def __init__(self, time_budget = 3600,
                 time_limit_for_evaluation = 360,
                 memory_limit = 3024,
                 multi_fidelity=False,
                 use_parameter_importance=False,
                 use_rave=False,
                 scoring_func="balanced_accuracy",
                 seed=1
                 ):
        self.time_budget = time_budget
        self.time_limit_for_evaluation = time_limit_for_evaluation
        self.memory_limit = memory_limit
        self.multi_fidelity = multi_fidelity
        self.use_parameter_importance = use_parameter_importance
        self.use_rave = use_rave

        if scoring_func == "balanced_accuracy":
            self.scoring_func = balanced_accuracy_score
        elif scoring_func == "accuracy":
            self.scoring_func = accuracy_score
        elif scoring_func == "roc_auc":
            self.scoring_func = roc_auc_score
        else:
            raise Exception("Score func {0} unknown".format(scoring_func))

        self.seed = seed
        np.random.seed(seed)

    def fit(self, X, y, X_test=None, y_test=None, categorical_features=None):
        print("-> X shape: {0}".format(str(X.shape)))
        print("-> y shape: {0}".format(str(y.shape)))
        print("-> X_test shape: {0}".format(str(X_test.shape)))
        print("-> y_test shape: {0}".format(str(y_test.shape)))
        print("-> Categorical features: {0}".format(str(categorical_features)))

        if issparse(X):
            self.config_space = pcs.read(open(os.path.dirname(os.path.abspath(__file__)) + "/model_config/1_1.pcs", "r"))
            print("-> Data is sparse")
        else:
            self.config_space = pcs.read(open(os.path.dirname(os.path.abspath(__file__)) + "/model_config/1_0.pcs", "r"))
            print("-> Data is dense")

        eval_func = partial(evaluate, X=X, y=y, score_func=self.scoring_func,
                            categorical_features=categorical_features, seed=self.seed)

        # This function may hang indefinitely
        self.searcher = Search(eval_func=eval_func,
                               config_space=self.config_space,
                               mem_in_mb=self.memory_limit,
                               cpu_time_in_s=self.time_limit_for_evaluation,
                               #logfile=self.info_training["scoring_path"] if "scoring_path"  in self.info_training else "",
                               time_budget=self.time_budget,
                               multi_fidelity=self.multi_fidelity,
                               use_parameter_importance=self.use_parameter_importance,
                               use_rave=self.use_rave)
        self.searcher.print_config()

        self.searcher.run(nb_simulation=100000000000)

        test_func = pynisher.enforce_limits(mem_in_mb=self.memory_limit,
                                            cpu_time_in_s=self.time_limit_for_evaluation * 3
                                            )(test_function)
        return self.searcher.test_performance(X, y, X_test, y_test, test_func)
