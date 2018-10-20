from sklearn.decomposition import KernelPCA


def get_model(name, config):
    kernel = config["preprocessor:kernel_pca:kernel"]
    if kernel in ["poly", "sigmoid"]:
        coef0 = int(config["preprocessor:kernel_pca:coef0"])
    else:
        coef0 = None

    if kernel == "poly":
        degree = int(config["preprocessor:kernel_pca:degree"])
    else:
        degree = None

    if kernel in ["poly", "rbf"]:
        gamma = float(config["preprocessor:kernel_pca:gamma"])
    else:
        gamma = None

    model = KernelPCA(
        n_components=int(config["preprocessor:kernel_pca:n_components"]),
        kernel=config["preprocessor:kernel_pca:kernel"],
        gamma=gamma,
        degree=degree,
        coef0=coef0
    )
    return (name, model)
