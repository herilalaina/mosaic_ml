from sklearn.decomposition import PCA


def get_model(name, config):
    model = PCA(
        n_components=float(config["preprocessor:pca:keep_variance"]),
        whiten=eval(config["preprocessor:pca:whiten"])
    )
    return (name, model)
