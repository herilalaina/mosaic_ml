# Mosaic library
import os
import json
from functools import partial
import simplejson as json

import numpy as np
# pynisher
import pynisher
# Config space
from ConfigSpace.read_and_write import pcs_new as pcs
from mosaic.mosaic import Search
from mosaic_ml.metafeatures import get_dataset_metafeature_from_openml
# scipy
from scipy.sparse import issparse
# Metric
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from sklearn.metrics import balanced_accuracy_score
except:
    pass

from networkx.readwrite.gpickle import write_gpickle
from networkx.readwrite import json_graph
from mosaic_ml.evaluator import evaluate, test_function, evaluate_competition


class AutoML():
    def __init__(self, time_budget=3600,
                 time_limit_for_evaluation=360,
                 memory_limit=3024,
                 multi_fidelity=False,
                 use_parameter_importance=False,
                 scoring_func="balanced_accuracy",
                 seed=1,
                 data_manager=None,
                 exec_dir=""
                 ):
        self.time_budget = time_budget
        self.time_limit_for_evaluation = time_limit_for_evaluation
        self.memory_limit = memory_limit
        self.multi_fidelity = multi_fidelity
        self.use_parameter_importance = use_parameter_importance
        self.config_space = None

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

        self.searcher = None
        self.data_manager = data_manager
        self.exec_dir = exec_dir
        try:
            os.mkdir(self.exec_dir)
        except Exception as e:
            raise(e)



    def fit(self, X, y, X_test=None, y_test=None, categorical_features=None, intial_configurations = [], id_task = None):
        print("-> X shape: {0}".format(str(X.shape)))
        print("-> y shape: {0}".format(str(y.shape)))
        if X_test is not None:
            print("-> X_test shape: {0}".format(str(X_test.shape)))
            print("-> y_test shape: {0}".format(str(y_test.shape)))
        print("-> Categorical features: {0}".format(str([i for i, x in enumerate(categorical_features) if x == "categorical"])))

        if issparse(X):
            self.config_space = pcs.read(
                open(os.path.dirname(os.path4.abspath(__file__)) + "/model_config/1_1.pcs", "r"))
            print("-> Data is sparse")
        else:
            self.config_space = pcs.read(
                open(os.path.dirname(os.path.abspath(__file__)) + "/model_config/1_0.pcs", "r"))
            print("-> Data is dense")

        dataset_features = get_dataset_metafeature_from_openml(id_task)

        eval_func = partial(evaluate, X=X, y=y, score_func=self.scoring_func,
                            categorical_features=categorical_features, seed=self.seed,
                            test_data = {"X_test": X_test, "y_test": y_test})

        # This function may hang indefinitely
        self.searcher = Search(eval_func=eval_func,
                          config_space=self.config_space,
                          mem_in_mb=self.memory_limit,
                          cpu_time_in_s=self.time_limit_for_evaluation,
                          time_budget=self.time_budget,
                          multi_fidelity=self.multi_fidelity,
                          use_parameter_importance=self.use_parameter_importance,
                          seed=self.seed)

        try:
            self.searcher.run(nb_simulation=100000000000, intial_configuration=intial_configurations)
        except Exception as e:
            pass

        # Save X, y, y_time performance
        self.searcher.mcts.env.score_model.save_data(self.exec_dir)
        # Save tree
        write_gpickle(self.searcher.mcts.tree.tree, os.path.join(self.exec_dir, "tree.json"))
        # Save full log
        self.save_full_log(os.path.join(self.exec_dir, "full_log.json"))



    def refit(self, X, y, X_test=None, y_test=None, categorical_features=None, cpu_time_in_s=360, time_budget=3600):
        print("Fit with warmstart")
        print("-> X shape: {0}".format(str(X.shape)))
        print("-> y shape: {0}".format(str(y.shape)))
        if X_test is not None:
            print("-> X_test shape: {0}".format(str(X_test.shape)))
            print("-> y_test shape: {0}".format(str(y_test.shape)))
        print("-> Categorical features: {0}".format(str(categorical_features)))
        self.time_budget = time_budget
        self.time_limit_for_evaluation = cpu_time_in_s

        eval_func = partial(evaluate, X=X, y=y, score_func=self.scoring_func,
                            categorical_features=categorical_features, seed=self.seed,
                            test_data = {"X_data": X_test, "y_test": y_test})

        self.searcher.run_warmstrat(eval_func,
                      mem_in_mb=self.memory_limit,
                      cpu_time_in_s=cpu_time_in_s,
                      time_budget=self.time_limit_for_evaluation,
                      nb_simulation = 100000000000)

    def get_history(self):
        return self.searcher.get_history_run()

    def save_full_log(self, file):
        with open(file, 'w') as outfile:
            json.dump(self.searcher.mcts.env.history_score, outfile)


    def get_test_performance(self, X, y, categorical_features, X_test=None, y_test=None):
        test_func = pynisher.enforce_limits(mem_in_mb=self.memory_limit,
                                            cpu_time_in_s=self.time_limit_for_evaluation * 3
                                            )(test_function)
        print("Get test performance ...")
        return self.searcher.test_performance(X, y, X_test, y_test, test_func, categorical_features)
