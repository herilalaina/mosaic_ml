# Mosaic library
import os
import json
import pynisher
from functools import partial
import simplejson as json

import numpy as np
# pynisher
# Config space
from mosaic.external.ConfigSpace import pcs_new as pcs
from mosaic_ml.mosaic_wrapper.mosaic import Search
from mosaic_ml.metafeatures import get_dataset_metafeature_from_openml
# scipy
from scipy.sparse import issparse
# Metric
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score

from networkx.readwrite.gpickle import write_gpickle
from networkx.readwrite import json_graph
from mosaic_ml.evaluator import evaluate, test_function, evaluate_competition, evaluate_generate_metadata

from mosaic_ml.model_config.encoding import OneHotEncoding
from mosaic_ml.sklearn_env import SklearnEnv


class AutoML():
    def __init__(self, time_budget=3600,
                 time_limit_for_evaluation=300,
                 memory_limit=3024,
                 scoring_func="roc_auc",
                 seed=1,
                 data_manager=None,
                 exec_dir=""
                 ):
        self.time_budget = time_budget
        self.time_limit_for_evaluation = time_limit_for_evaluation
        self.memory_limit = memory_limit
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

    def adapt_search_space(self, X, y):
        import ConfigSpace.hyperparameters as CSH
        self.problem_dependant_parameter = ["preprocessor:feature_agglomeration:n_clusters",
                                            "preprocessor:kernel_pca:n_components",
                                            "preprocessor:kitchen_sinks:n_components",
                                            "preprocessor:nystroem_sampler:n_components",
                                            # "preprocessor:fast_ica:n_components"
                                            ]

        try:
            enc = OneHotEncoding.OneHotEncoder()
            nb_normal, nb_onehot_enc = np.shape(
                X)[1], np.shape(enc.fit_transform(X))[1]
        except:
            nb_normal, nb_onehot_enc = np.shape(X)[1], np.shape(X)[1]

        self.searcher.mcts.env.problem_dependant_param = self.problem_dependant_parameter

        try:
            from sklearn.naive_bayes import MultinomialNB
            MultinomialNB().fit(X, y)
            is_positive = True
        except:
            is_positive = False

        self.searcher.mcts.env.problem_dependant_value = {
            "no_encoding": nb_normal,
            "one_hot_encoding": nb_onehot_enc,
            "is_positive": is_positive
        }
        print(self.searcher.mcts.env.problem_dependant_value)

    def prepare_ensemble(self, X, y):
        from sklearn.model_selection import train_test_split

        self.ensemble_dir = os.path.join(self.exec_dir, "ensemble_files")
        try:
            os.mkdir(self.ensemble_dir)
        except Exception as e:
            raise (e)

        _, _, y_train, y_test = train_test_split(
            X, y, test_size=0.329, random_state=self.seed)
        np.save(os.path.join(self.ensemble_dir, "y_valid.npy"), y_test)
        np.save(os.path.join(self.ensemble_dir, "y_test.npy"), y)

    def fit(self, X, y, X_test=None, y_test=None, categorical_features=None, intial_configurations=[], id_task=None, policy_arg={}):
        X = np.array(X)
        y = np.array(y)
        if X_test is not None:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
        print("-> X shape: {0}".format(str(X.shape)))
        print("-> y shape: {0}".format(str(y.shape)))
        if X_test is not None:
            print("-> X_test shape: {0}".format(str(X_test.shape)))
            print("-> y_test shape: {0}".format(str(y_test.shape)))
        print("-> Categorical features: {0}".format(
            str([i for i, x in enumerate(categorical_features) if x == "categorical"])))

        if issparse(X):
            self.config_space = pcs.read(
                open(os.path.dirname(os.path4.abspath(__file__)) + "/model_config/1_1.pcs", "r"))
            print("-> Data is sparse")
        else:
            self.config_space = pcs.read(
                open(os.path.dirname(os.path.abspath(__file__)) + "/model_config/1_0.pcs", "r"))
            print("-> Data is dense")

        #dataset_features = get_dataset_metafeature_from_openml(id_task)
        #self.prepare_ensemble(X, y)

        eval_func = partial(evaluate, X=X, y=y, score_func=self.scoring_func,
                            categorical_features=categorical_features, seed=self.seed,
                            test_data={"X_test": X_test, "y_test": y_test},)
        # store_directory=self.ensemble_dir)

        # Create environment
        environment = SklearnEnv(eval_func=eval_func,
                                 config_space=self.config_space,
                                 mem_in_mb=self.memory_limit,
                                 cpu_time_in_s=self.time_limit_for_evaluation,
                                 seed=self.seed)

        # This function may hang indefinitely
        self.searcher = Search(environment=environment,
                               time_budget=self.time_budget,
                               seed=self.seed,
                               policy_arg=policy_arg,
                               exec_dir=self.exec_dir)

        self.adapt_search_space(X, y)

        try:
            res = self.searcher.run(
                nb_simulation=100000000000, intial_configuration=intial_configurations)
        except Exception as e:
            raise(e)

        return res

    def get_history(self):
        return self.searcher.get_history_run()

    def save_full_log(self, file):
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
    with open(file, 'w') as outfile:
            json.dump(self.searcher.mcts.env.history_score, outfile)

    def get_test_performance(self, X, y, categorical_features, X_test=None, y_test=None):
        test_func = pynisher.enforce_limits(mem_in_mb=self.memory_limit,
                                            cpu_time_in_s=self.time_limit_for_evaluation * 3
                                            )(test_function)
        print("Get test performance ...")
        return self.searcher.test_performance(X, y, X_test, y_test, test_func, categorical_features)
