import warnings

from mosaic_ml.model_config.util import check_for_bool, check_none


class FastICA:
    def __init__(self, algorithm, whiten, fun, n_components=None,
                 random_state=None):
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.n_components = n_components

        self.random_state = random_state

    def _fit(self, X, Y=None):
        import sklearn.decomposition

        self.whiten = check_for_bool(self.whiten)
        if check_none(self.n_components):
            self.n_components = None
        else:
            self.n_components = int(self.n_components)

        self.preprocessor = sklearn.decomposition.FastICA(
            n_components=self.n_components, algorithm=self.algorithm,
            fun=self.fun, whiten=self.whiten, random_state=self.random_state
        )
        # Make the RuntimeWarning an Exception!
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message='array must not contain infs or NaNs')
            try:
                return self.preprocessor.fit_transform(X)
            except ValueError as e:
                if 'array must not contain infs or NaNs' in e.args[0]:
                    raise ValueError(
                        "Bug in scikit-learn: https://github.com/scikit-learn/scikit-learn/pull/2738")

        return self

    def fit(self, X, y):
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        return self._fit(X)

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("feature_preprocessor:fast_ica:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = FastICA(**list_param)
    return (name, model)
