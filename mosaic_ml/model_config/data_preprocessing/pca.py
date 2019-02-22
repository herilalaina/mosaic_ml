from autosklearn.pipeline.components.feature_preprocessing.pca import PCA


def get_model(name, config, random_state):
    model = PCA(
        keep_variance=config["preprocessor:pca:keep_variance"],
        whiten=config["preprocessor:pca:whiten"],
        random_state=random_state
    )
    return (name, model)
