from sklearn.decomposition import FastICA


def get_model(name, config):
    if eval(config["preprocessor:fast_ica:whiten"]):
        n_components = int(config["preprocessor:fast_ica:n_components"])
    else:
        n_components = None
    model = FastICA(
        algorithm=config["preprocessor:fast_ica:algorithm"],
        fun=config["preprocessor:fast_ica:fun"],
        whiten=eval(config["preprocessor:fast_ica:whiten"]),
        n_components=n_components
    )
    return (name, model)
