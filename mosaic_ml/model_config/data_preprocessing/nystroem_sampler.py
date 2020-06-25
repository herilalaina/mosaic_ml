import numpy as np


class Nystroem:
    def __init__(self, kernel, n_components, gamma=1.0, degree=3,
                 coef0=1, random_state=None):
        self.kernel = kernel
        self.n_components = n_components
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state

    def fit(self, X, Y=None):
        import scipy.sparse
        import sklearn.kernel_approximation

        self.n_components = int(self.n_components)
        self.gamma = float(self.gamma)
        self.degree = int(self.degree)
        self.coef0 = float(self.coef0)

        self.preprocessor = sklearn.kernel_approximation.Nystroem(
            kernel=self.kernel, n_components=self.n_components,
            gamma=self.gamma, degree=self.degree, coef0=self.coef0,
            random_state=self.random_state)

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.kernel == 'chi2':
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0.0
            else:
                X[X < 0] = 0.0

        self.preprocessor.fit(X.astype(np.float64))
        return self

    def transform(self, X):
        import scipy.sparse

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.kernel == 'chi2':
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0.0
            else:
                X[X < 0] = 0.0

        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("feature_preprocessor:nystroem_sampler:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = Nystroem(**list_param)
    return (name, model)
