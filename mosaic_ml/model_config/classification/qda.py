
import numpy as np
from mosaic_ml.model_config.util import softmax


class QDA:

    def __init__(self, reg_param, random_state=None):
        self.reg_param = float(reg_param)
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.discriminant_analysis

        estimator = sklearn.discriminant_analysis.\
            QuadraticDiscriminantAnalysis(reg_param=self.reg_param)

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(
                estimator, n_jobs=1)
        else:
            self.estimator = estimator

        self.estimator.fit(X, Y)

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            problems = []
            for est in self.estimator.estimators_:
                problem = np.any(np.any([np.any(s <= 0.0) for s in
                                         est.scalings_]))
                problems.append(problem)
            problem = np.any(problems)
        else:
            problem = np.any(np.any([np.any(s <= 0.0) for s in
                                     self.estimator.scalings_]))
        if problem:
            raise ValueError('Numerical problems in QDA. QDA.scalings_ '
                             'contains values <= 0.0')
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
        if k.startswith("classifier:qda:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = QDA(**list_param)
    return (name, model)
