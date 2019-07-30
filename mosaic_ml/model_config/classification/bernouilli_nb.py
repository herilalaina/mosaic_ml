import numpy as np

from mosaic_ml.model_config.util import check_for_bool


class BernoulliNB:
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

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.n_iter = 0
            self.fully_fit_ = False
            self.fit_prior = check_for_bool(self.fit_prior)
            self.estimator = sklearn.naive_bayes.BernoulliNB(
                alpha=self.alpha, fit_prior=self.fit_prior)
            self.classes_ = np.unique(y.astype(int))

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
                # Upper limit, scipy.sparse doesn't seem to handle max > len(matrix)
                stop = min(stop, y.shape[0])
                if X[start:stop].shape[0] == 0:
                    self.fully_fit_ = True
                    break

                self.estimator.partial_fit(
                    X[start:stop], y[start:stop], self.classes_)
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
    model = BernoulliNB(
        alpha=config["classifier:bernoulli_nb:alpha"],
        fit_prior=config["classifier:bernoulli_nb:fit_prior"],
        random_state=random_state
    )
    return (name, model)
