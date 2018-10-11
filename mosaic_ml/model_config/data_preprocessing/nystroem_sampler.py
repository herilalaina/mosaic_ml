from sklearn.kernel_approximation import Nystroem

def get_model(name, config):
    model = Nystroem(
        n_components=int(config["preprocessor:nystroem_sampler:n_components"]),
        kernel=config["preprocessor:nystroem_sampler:kernel"],
        gamma=float(config["preprocessor:nystroem_sampler:gamma"]),
        degree=int(config["preprocessor:nystroem_sampler:degree"]),
        coef0=int(config["preprocessor:nystroem_sampler:coef0"])
    )
    return (name, config)