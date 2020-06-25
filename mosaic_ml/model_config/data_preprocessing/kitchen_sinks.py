
class RandomKitchenSinks():

    def __init__(self, gamma, n_components, random_state=None):
        """ Parameters:
        gamma: float
               Parameter of the rbf kernel to be approximated exp(-gamma * x^2)

        n_components: int
               Number of components (output dimensionality) used to approximate the kernel
        """
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, Y=None):
        import sklearn.kernel_approximation

        self.n_components = int(self.n_components)
        self.gamma = float(self.gamma)

        self.preprocessor = sklearn.kernel_approximation.RBFSampler(
            self.gamma, self.n_components, self.random_state)
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("feature_preprocessor:kitchen_sinks:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = RandomKitchenSinks(**list_param)
    return (name, model)
