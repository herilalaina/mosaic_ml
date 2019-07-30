
from mosaic_ml.model_config.util import check_for_bool


class PolynomialFeatures:
    def __init__(self, degree, interaction_only, include_bias, random_state=None):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, Y):
        import sklearn.preprocessing

        self.degree = int(self.degree)
        self.interaction_only = check_for_bool(self.interaction_only)
        self.include_bias = check_for_bool(self.include_bias)

        self.preprocessor = sklearn.preprocessing.PolynomialFeatures(
            degree=self.degree, interaction_only=self.interaction_only,
            include_bias=self.include_bias)
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)


def get_model(name, config, random_state):
    model = PolynomialFeatures(
        degree=config["preprocessor:polynomial:degree"],
        include_bias=config["preprocessor:polynomial:include_bias"],
        interaction_only=config["preprocessor:polynomial:interaction_only"],
        random_state=random_state
    )
    return (name, model)
