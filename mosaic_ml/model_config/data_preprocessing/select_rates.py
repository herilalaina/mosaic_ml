
class SelectRates:
    def __init__(self, alpha, mode='fpr',
                 score_func="chi2", random_state=None):
        import sklearn.feature_selection

        self.random_state = random_state  # We don't use this
        self.alpha = alpha

        if callable(score_func):
            self.score_func = score_func
        elif score_func == "chi2":
            self.score_func = sklearn.feature_selection.chi2
        elif score_func == "f_classif":
            self.score_func = sklearn.feature_selection.f_classif
        else:
            raise ValueError("score_func must be in ('chi2, 'f_classif', 'mutual_info'), "
                             "but is: %s" % score_func)

        self.mode = mode

    def fit(self, X, y):
        import scipy.sparse
        import sklearn.feature_selection

        self.alpha = float(self.alpha)

        self.preprocessor = sklearn.feature_selection.GenericUnivariateSelect(
            score_func=self.score_func, param=self.alpha, mode=self.mode)

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0.0
            else:
                X[X < 0] = 0.0

        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        import scipy.sparse
        import sklearn.feature_selection

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0.0
            else:
                X[X < 0] = 0.0

        if self.preprocessor is None:
            raise NotImplementedError()
        try:
            Xt = self.preprocessor.transform(X)
        except ValueError as e:
            if "zero-size array to reduction operation maximum which has no " \
                    "identity" in e.message:
                raise ValueError(
                    "%s removed all features." % self.__class__.__name__)
            else:
                raise e

        if Xt.shape[1] == 0:
            raise ValueError(
                "%s removed all features." % self.__class__.__name__)
        return Xt


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("preprocessor:select_rates:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = SelectRates(**list_param)

    return (name, model)
