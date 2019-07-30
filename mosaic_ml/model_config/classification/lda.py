
from mosaic_ml.model_config.util import softmax, check_none


class LDA:
    def __init__(self, shrinkage, n_components, tol, shrinkage_factor=0.5,
                 random_state=None):
        self.shrinkage = shrinkage
        self.n_components = n_components
        self.tol = tol
        self.shrinkage_factor = shrinkage_factor
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.discriminant_analysis
        import sklearn.multiclass

        if check_none(self.shrinkage):
            self.shrinkage_ = None
            solver = 'svd'
        elif self.shrinkage == "auto":
            self.shrinkage_ = 'auto'
            solver = 'lsqr'
        elif self.shrinkage == "manual":
            self.shrinkage_ = float(self.shrinkage_factor)
            solver = 'lsqr'
        else:
            raise ValueError(self.shrinkage)

        self.n_components = int(self.n_components)
        self.tol = float(self.tol)

        estimator = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
            n_components=self.n_components, shrinkage=self.shrinkage_,
            tol=self.tol, solver=solver)

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(
                estimator, n_jobs=1)
        else:
            self.estimator = estimator

        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        df = self.estimator.predict_proba(X)
        return softmax(df)


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:lda:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = LDA(**list_param)
    return (name, model)
