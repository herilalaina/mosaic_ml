
from mosaic_ml.model_config.util import check_none, check_for_bool

class RandomTreesEmbedding:

    def __init__(self, n_estimators, max_depth, min_samples_split,
                 min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes,
                 bootstrap, sparse_output=True, n_jobs=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.sparse_output = sparse_output
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _fit(self, X, Y=None):
        import sklearn.ensemble

        self.n_estimators = int(self.n_estimators)
        if check_none(self.max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(self.max_depth)
        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
        self.bootstrap = check_for_bool(self.bootstrap)

        self.preprocessor = sklearn.ensemble.RandomTreesEmbedding(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            sparse_output=self.sparse_output,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.preprocessor.fit(X, Y)
        return self

    def fit(self, X, y):
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        self._fit(X)
        return self.transform(X)

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("preprocessor:random_trees_embedding:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = RandomTreesEmbedding(**list_param)
    return (name, model)
