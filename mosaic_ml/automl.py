# Mosaic library
from mosaic.mosaic import Search
from mosaic_ml.evaluator import evaluate, test_function

# pynisher
import pynisher

# scipy
from scipy.sparse import issparse


# Metric
from sklearn.metrics import balanced_accuracy_score

# Config space
from ConfigSpace.read_and_write import pcs


from functools import partial

class AutoML():
    def __init__(self, time_budget = 3600,
                 time_limit_for_evaluation = 360,
                 memory_limit = 3024,
                 info_training = {},
                 multi_fidelity=False,
                 use_parameter_importance=False,
                 use_rave=False
                 ):
        self.time_budget = time_budget
        self.time_limit_for_evaluation = time_limit_for_evaluation
        self.memory_limit = memory_limit
        self.info_training = info_training
        self.multi_fidelity = multi_fidelity
        self.use_parameter_importance = use_parameter_importance
        self.use_rave = use_rave

    def fit(self, X, y, X_test=None, y_test=None):
        if issparse(X):
            self.config_space = pcs.read(open("mosaic_ml/model_config/1_1.pcs", "r"))
        else:
            self.config_space = pcs.read(open("mosaic_ml/model_config/1_0.pcs", "r"))

        eval_func = partial(evaluate, X=X, y=y, X_TEST=X_test, Y_TEST=y_test, info = self.info_training, score_func=balanced_accuracy_score)

        # This function may hang indefinitely
        self.searcher = Search(eval_func=eval_func,
                               config_space=self.config_space,
                               mem_in_mb=self.memory_limit,
                               cpu_time_in_s=self.time_limit_for_evaluation,
                               logfile=self.info_training["scoring_path"] if "scoring_path"  in self.info_training else "",
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
