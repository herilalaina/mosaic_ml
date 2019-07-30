import numpy as np

from mosaic_ml.model_config.util import check_for_bool


class PCA:
    def __init__(self, keep_variance, whiten, random_state=None):
        self.keep_variance = keep_variance
        self.whiten = whiten

        self.random_state = random_state

    def fit(self, X, Y=None):
        import sklearn.decomposition
        n_components = float(self.keep_variance)
        self.whiten = check_for_bool(self.whiten)

        self.preprocessor = sklearn.decomposition.PCA(n_components=n_components,
                                                      whiten=self.whiten,
                                                      copy=True)
        self.preprocessor.fit(X)

        if not np.isfinite(self.preprocessor.components_).all():
            raise ValueError("PCA found non-finite components.")

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)


def get_model(name, config, random_state):
    model = PCA(
        keep_variance=config["preprocessor:pca:keep_variance"],
        whiten=config["preprocessor:pca:whiten"],
        random_state=random_state
    )
    return (name, model)
