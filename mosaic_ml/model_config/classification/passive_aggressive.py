
import numpy as np

from mosaic_ml.model_config.util import softmax, check_for_bool


class PassiveAggressive:
    def __init__(self, C, fit_intercept, tol, loss, average, random_state=None):
        self.C = C
        self.fit_intercept = fit_intercept
        self.average = average
        self.tol = tol
        self.loss = loss
        self.random_state = random_state
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

    def iterative_fit(self, X, y, n_iter=2, refit=False, sample_weight=None):
        from sklearn.linear_model.passive_aggressive import \
            PassiveAggressiveClassifier

        # Need to fit at least two iterations, otherwise early stopping will not
        # work because we cannot determine whether the algorithm actually
        # converged. The only way of finding this out is if the sgd spends less
        # iterations than max_iter. If max_iter == 1, it has to spend at least
        # one iteration and will always spend at least one iteration, so we
        # cannot know about convergence.

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.fully_fit_ = False

            self.average = check_for_bool(self.average)
            self.fit_intercept = check_for_bool(self.fit_intercept)
            self.tol = float(self.tol)
            self.C = float(self.C)

            call_fit = True
            self.estimator = PassiveAggressiveClassifier(
                C=self.C,
                fit_intercept=self.fit_intercept,
                max_iter=n_iter,
                tol=self.tol,
                loss=self.loss,
                shuffle=True,
                random_state=self.random_state,
                warm_start=True,
                average=self.average,
            )
            self.classes_ = np.unique(y.astype(int))
        else:
            call_fit = False

        # Fallback for multilabel classification
        if len(y.shape) > 1 and y.shape[1] > 1:
            import sklearn.multiclass
            self.estimator.max_iter = 50
            self.estimator = sklearn.multiclass.OneVsRestClassifier(
                self.estimator, n_jobs=1)
            self.estimator.fit(X, y)
            self.fully_fit_ = True
        else:
            if call_fit:
                self.estimator.fit(X, y)
            else:
                self.estimator.max_iter += n_iter
                self.estimator.max_iter = min(self.estimator.max_iter,
                                              1000)
                self.estimator._validate_params()
                lr = "pa1" if self.estimator.loss == "hinge" else "pa2"
                self.estimator._partial_fit(
                    X, y,
                    alpha=1.0,
                    C=self.estimator.C,
                    loss="hinge",
                    learning_rate=lr,
                    max_iter=n_iter,
                    classes=None,
                    sample_weight=sample_weight,
                    coef_init=None,
                    intercept_init=None
                )
                if (
                    self.estimator._max_iter >= 1000
                    or n_iter > self.estimator.n_iter_
                ):
                    self.fully_fit_ = True

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, 'fully_fit_'):
            return False
        else:
            return self.fully_fit_

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        df = self.estimator.decision_function(X)
        return softmax(df)


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:passive_aggressive:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = PassiveAggressive(**list_param)
    return (name, model)
