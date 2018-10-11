from sklearn.decomposition import KernelPCA


def get_model(name, config):
    model = KernelPCA(
        n_components=int(config["preprocessor:kernel_pca:n_components"]),
        kernel=config["preprocessor:kernel_pca:kernel"],
        gamma=float(config["preprocessor:kernel_pca:gamma"]),
        degree=int(config["preprocessor:kernel_pca:degree"]),
        coef0=int(config["preprocessor:kernel_pca:coef0"])
    )
    return (name, model)