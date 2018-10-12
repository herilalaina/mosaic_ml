from sklearn.kernel_approximation import Nystroem

def get_model(name, config):
    kernel = config["preprocessor:nystroem_sampler:kernel"]
    if kernel in ["poly", "sigmoid"]:
        coef0 = int(config["preprocessor:nystroem_sampler:coef0"])
    else:
        coef0 = None

    if kernel == "poly":
        degree = int(config["preprocessor:nystroem_sampler:degree"])
    else:
        degree = None

    if kernel in ["poly", "rbf", "sigmoid"]:
        gamma = float(config["preprocessor:nystroem_sampler:gamma"])
    else:
        gamma = None


    model = Nystroem(
        n_components=int(config["preprocessor:nystroem_sampler:n_components"]),
        kernel=config["preprocessor:nystroem_sampler:kernel"],
        gamma=gamma,
        degree=degree,
        coef0=coef0
    )
    return (name, model)