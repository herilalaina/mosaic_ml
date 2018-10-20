from sklearn.kernel_approximation import RBFSampler


def get_model(name, config):
    model = RBFSampler(
        gamma=float(config["preprocessor:kitchen_sinks:gamma"]),
        n_components=int(config["preprocessor:kitchen_sinks:n_components"])
    )
    return (name, model)
