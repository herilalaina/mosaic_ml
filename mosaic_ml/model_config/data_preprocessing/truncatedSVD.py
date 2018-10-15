from sklearn.decomposition import TruncatedSVD

def get_model(name, config):
    model = TruncatedSVD(
        n_components = config["preprocessor:truncatedSVD:target_dim"]
    )
    return (name, model)