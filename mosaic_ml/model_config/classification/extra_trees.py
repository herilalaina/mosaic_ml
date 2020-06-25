import numpy as np

from mosaic_ml.model_config.util import convert_multioutput_multiclass_to_multilabel
from mosaic_ml.model_config.util import check_for_bool, check_none


class ExtraTreesClassifier:

    def __init__(self, criterion, min_samples_leaf,
                 min_samples_split,  max_features, bootstrap, max_leaf_nodes,
                 max_depth, min_weight_fraction_leaf, min_impurity_decrease,
                 oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 class_weight=None):

        self.n_estimators = self.get_max_iter()
        self.estimator_increment = 10
        if criterion not in ("gini", "entropy"):
            raise ValueError("'criterion' is not in ('gini', 'entropy'): "
                             "%s" % criterion)
        self.criterion = criterion

        if check_none(max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(max_depth)
        if check_none(max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(max_leaf_nodes)

        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)
        self.max_features = float(max_features)
        self.bootstrap = check_for_bool(bootstrap)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.oob_score = oob_score
        self.n_jobs = int(n_jobs)
        self.random_state = random_state
        self.verbose = int(verbose)
        self.class_weight = class_weight
        self.estimator = None

    def fit(self, X, y, sample_weight=None):
        self.iterative_fit(
            X, y, n_iter=2, refit=True, sample_weight=sample_weight
        )
        iteration = 2
        while not self.configuration_fully_fitted():
            n_iter = int(2 ** iteration / 2)
            self.iterative_fit(X, y, n_iter=n_iter,
                               sample_weight=sample_weight)
            iteration += 1
        return self

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):
        from sklearn.ensemble import ExtraTreesClassifier as ETC

        if refit:
            self.estimator = None

        if self.estimator is None:
            max_features = int(X.shape[1] ** float(self.max_features))
            self.estimator = ETC(n_estimators=n_iter,
                                 criterion=self.criterion,
                                 max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                 bootstrap=self.bootstrap,
                                 max_features=max_features,
                                 max_leaf_nodes=self.max_leaf_nodes,
                                 min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                 min_impurity_decrease=self.min_impurity_decrease,
                                 oob_score=self.oob_score,
                                 n_jobs=self.n_jobs,
                                 verbose=self.verbose,
                                 random_state=self.random_state,
                                 class_weight=self.class_weight,
                                 warm_start=True)

        else:
            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(self.estimator.n_estimators,
                                              self.n_estimators)

        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        return not len(self.estimator.estimators_) < self.n_estimators

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_max_iter():
        return 512


def get_model(name, config, random_state):
    list_param = {"random_state": random_state,
                  "class_weight": "weighting" if config["class_weight"] == "weighting" else None}
    for k in config:
        if k.startswith("classifier:extra_trees:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = ExtraTreesClassifier(**list_param)
    return (name, model)
