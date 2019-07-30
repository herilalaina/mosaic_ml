from sklearn.tree import DecisionTreeClassifier


from mosaic_ml.model_config.util import convert_multioutput_multiclass_to_multilabel
from mosaic_ml.model_config.util import check_none


class AdaboostClassifier:

    def __init__(self, n_estimators, learning_rate, algorithm, max_depth,
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
        self.max_depth = max_depth
        self.estimator = None

    def fit(self, X, Y, sample_weight=None):
        import sklearn.ensemble
        import sklearn.tree

        self.n_estimators = int(self.n_estimators)
        self.learning_rate = float(self.learning_rate)
        self.max_depth = int(self.max_depth)
        base_estimator = sklearn.tree.DecisionTreeClassifier(
            max_depth=self.max_depth)

        estimator = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state
        )

        estimator.fit(X, Y, sample_weight=sample_weight)

        self.estimator = estimator
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:adaboost:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = AdaboostClassifier(**list_param)
    return (name, model)
