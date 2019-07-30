import numpy as np


class TruncatedSVD:
    def __init__(self, target_dim, random_state=None):
        self.target_dim = target_dim
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, Y):
        import sklearn.decomposition

        self.target_dim = int(self.target_dim)
        target_dim = min(self.target_dim, X.shape[1] - 1)
        self.preprocessor = sklearn.decomposition.TruncatedSVD(
            target_dim, algorithm='randomized')
        # TODO: remove when migrating to sklearn 0.16
        # Circumvents a bug in sklearn
        # https://github.com/scikit-learn/scikit-learn/commit/f08b8c8e52663167819f242f605db39f3b5a6d0c
        # X = X.astype(np.float64)
        self.preprocessor.fit(X, Y)

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("preprocessor:truncatedSVD:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = TruncatedSVD(**list_param)
    return (name, model)
