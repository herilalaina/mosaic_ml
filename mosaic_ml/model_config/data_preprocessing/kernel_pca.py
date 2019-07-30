import warnings
import numpy as np


class KernelPCA:
    def __init__(self, n_components, kernel, degree=3, gamma=0.25, coef0=0.0,
                 random_state=None):
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.random_state = random_state

    def fit(self, X, Y=None):
        import scipy.sparse
        import sklearn.decomposition

        self.n_components = int(self.n_components)
        self.degree = int(self.degree)
        self.gamma = float(self.gamma)
        self.coef0 = float(self.coef0)

        self.preprocessor = sklearn.decomposition.KernelPCA(
            n_components=self.n_components, kernel=self.kernel,
            degree=self.degree, gamma=self.gamma, coef0=self.coef0,
            remove_zero_eig=True)
        if scipy.sparse.issparse(X):
            X = X.astype(np.float64)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            self.preprocessor.fit(X)
        # Raise an informative error message, equation is based ~line 249 in
        # kernel_pca.py in scikit-learn
        if len(self.preprocessor.alphas_ / self.preprocessor.lambdas_) == 0:
            raise ValueError('KernelPCA removed all features!')
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            X_new = self.preprocessor.transform(X)

            # TODO write a unittest for this case
            if X_new.shape[1] == 0:
                raise ValueError("KernelPCA removed all features!")

            return X_new


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("preprocessor:kernel_pca:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = KernelPCA(**list_param)
    return (name, model)
