import numpy as np

from mosaic_ml.model_config.util import check_for_bool


class MultinomialNB:

    def __init__(self, alpha, fit_prior, random_state=None, verbose=0):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None

    def fit(self, X, y, sample_weight=None):
        self.iterative_fit(X, y, n_iter=2, refit=True)
        iteration = 2
        while not self.configuration_fully_fitted():
            n_iter = int(2 ** iteration / 2)
            self.iterative_fit(X, y, n_iter=n_iter, refit=False)
            iteration += 1
        return self

    def iterative_fit(self, X, y, n_iter=1, refit=False):
        import sklearn.naive_bayes
        import scipy.sparse

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.fit_prior = check_for_bool(self.fit_prior)
            self.alpha = float(self.alpha)
            self.n_iter = 0
            self.fully_fit_ = False
            self.estimator = sklearn.naive_bayes.MultinomialNB(
                alpha=self.alpha, fit_prior=self.fit_prior)
            self.classes_ = np.unique(y.astype(int))

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if scipy.sparse.issparse(X):
            X.data[X.data < 0] = 0.0
        else:
            X[X < 0] = 0.0

        # Fallback for multilabel classification
        if len(y.shape) > 1 and y.shape[1] > 1:
            import sklearn.multiclass
            self.estimator.n_iter = self.n_iter
            self.estimator = sklearn.multiclass.OneVsRestClassifier(
                self.estimator, n_jobs=1)
            self.estimator.fit(X, y)
            self.fully_fit_ = True
        else:
            for iter in range(n_iter):
                start = min(self.n_iter * 1000, y.shape[0])
                stop = min((self.n_iter + 1) * 1000, y.shape[0])
                if X[start:stop].shape[0] == 0:
                    self.fully_fit_ = True
                    break

                self.estimator.partial_fit(X[start:stop], y[start:stop],
                                           self.classes_)
                self.n_iter += 1

                if stop >= len(y):
                    self.fully_fit_ = True
                    break

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
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:multinomial_nb:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = MultinomialNB(**list_param)
    return (name, model)
