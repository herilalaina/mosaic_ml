import numpy as np


class FeatureAgglomeration:
    def __init__(self, n_clusters, affinity, linkage, pooling_func,
                 random_state=None):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.pooling_func = pooling_func
        self.random_state = random_state

        self.pooling_func_mapping = dict(mean=np.mean,
                                         median=np.median,
                                         max=np.max)

    def fit(self, X, Y=None):
        import sklearn.cluster

        self.n_clusters = int(self.n_clusters)

        n_clusters = min(self.n_clusters, X.shape[1])
        if not callable(self.pooling_func):
            self.pooling_func = self.pooling_func_mapping[self.pooling_func]

        self.preprocessor = sklearn.cluster.FeatureAgglomeration(
            n_clusters=n_clusters, affinity=self.affinity,
            linkage=self.linkage, pooling_func=self.pooling_func)
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("feature_preprocessor:feature_agglomeration:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = FeatureAgglomeration(**list_param)
    return (name, model)
