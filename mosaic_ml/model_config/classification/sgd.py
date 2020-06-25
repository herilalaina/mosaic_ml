import numpy as np

from mosaic_ml.model_config.util import softmax, check_for_bool


class SGD:
    def __init__(self, loss, penalty, alpha, fit_intercept, tol,
                 learning_rate, l1_ratio=0.15, epsilon=0.1,
                 eta0=0.01, power_t=0.5, average=False, random_state=None):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio
        self.epsilon = epsilon
        self.eta0 = eta0
        self.power_t = power_t
        self.random_state = random_state
        self.average = average
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
        from sklearn.linear_model.stochastic_gradient import SGDClassifier

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

            self.alpha = float(self.alpha)
            self.l1_ratio = float(self.l1_ratio) if self.l1_ratio is not None \
                else 0.15
            self.epsilon = float(self.epsilon) if self.epsilon is not None \
                else 0.1
            self.eta0 = float(self.eta0)
            self.power_t = float(self.power_t) if self.power_t is not None \
                else 0.5
            self.average = check_for_bool(self.average)
            self.fit_intercept = check_for_bool(self.fit_intercept)
            self.tol = float(self.tol)

            self.estimator = SGDClassifier(loss=self.loss,
                                           penalty=self.penalty,
                                           alpha=self.alpha,
                                           fit_intercept=self.fit_intercept,
                                           max_iter=n_iter,
                                           tol=self.tol,
                                           learning_rate=self.learning_rate,
                                           l1_ratio=self.l1_ratio,
                                           epsilon=self.epsilon,
                                           eta0=self.eta0,
                                           power_t=self.power_t,
                                           shuffle=True,
                                           average=self.average,
                                           random_state=self.random_state,
                                           warm_start=True)
            self.estimator.fit(X, y, sample_weight=sample_weight)
        else:
            self.estimator.max_iter += n_iter
            self.estimator.max_iter = min(self.estimator.max_iter, 512)
            self.estimator._validate_params()
            self.estimator._partial_fit(
                X, y,
                alpha=self.estimator.alpha,
                C=1.0,
                loss=self.estimator.loss,
                learning_rate=self.estimator.learning_rate,
                max_iter=n_iter,
                sample_weight=sample_weight,
                classes=None,
                coef_init=None,
                intercept_init=None
            )

        if self.estimator.max_iter >= 512 or n_iter > self.estimator.n_iter_:
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

        if self.loss in ["log", "modified_huber"]:
            return self.estimator.predict_proba(X)
        else:
            df = self.estimator.decision_function(X)
            return softmax(df)


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:sgd:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = SGD(**list_param)
    return (name, model)
