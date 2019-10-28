
import numpy as np

from mosaic_ml.model_config.util import check_for_bool, check_none


class ExtraTreesPreprocessorClassification:

    def __init__(self, n_estimators, criterion, min_samples_leaf,
                 min_samples_split, max_features, bootstrap, max_leaf_nodes,
                 max_depth, min_weight_fraction_leaf, min_impurity_decrease,
                 oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 class_weight=None):

        self.n_estimators = n_estimators
        self.estimator_increment = 10
        if criterion not in ("gini", "entropy"):
            raise ValueError("'criterion' is not in ('gini', 'entropy'): "
                             "%s" % criterion)
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease

        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight
        self.preprocessor = None

    def fit(self, X, Y, sample_weight=None):
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel

        self.n_estimators = int(self.n_estimators)

        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)

        if check_none(self.max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(self.max_depth)

        self.bootstrap = check_for_bool(self.bootstrap)
        self.n_jobs = int(self.n_jobs)
        self.min_impurity_decrease = float(self.min_impurity_decrease)
        self.max_features = self.max_features
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.min_samples_split = int(self.min_samples_split)
        self.verbose = int(self.verbose)

        max_features = int(X.shape[1] ** float(self.max_features))
        estimator = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            max_features=max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            class_weight=self.class_weight)
        estimator.fit(X, Y, sample_weight=sample_weight)
        self.preprocessor = SelectFromModel(estimator=estimator,
                                            threshold='mean',
                                            prefit=True)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError
        return self.preprocessor.transform(X)



def get_model(name, config, random_state):
    list_param = {"random_state": random_state,
                  "class_weight": "weighting" if config["class_weight"] == "weighting" else None}
    for k in config:
        if k.startswith("preprocessor:extra_trees_preproc_for_classification:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = ExtraTreesPreprocessorClassification(**list_param)
    return (name, model)
