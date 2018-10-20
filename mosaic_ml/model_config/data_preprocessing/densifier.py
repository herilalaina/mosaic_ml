from sklearn.base import BaseEstimator


class Densifier(BaseEstimator):
    def __init__(self, random_state=None):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from scipy import sparse
        if sparse.issparse(X):
            return X.todense().getA()
        else:
            return X
