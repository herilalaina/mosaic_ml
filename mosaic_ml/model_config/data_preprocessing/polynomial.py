from autosklearn.pipeline.components.feature_preprocessing.polynomial import PolynomialFeatures


def get_model(name, config, random_state):
    model = PolynomialFeatures(
        degree=config["preprocessor:polynomial:degree"],
        include_bias=config["preprocessor:polynomial:include_bias"],
        interaction_only=config["preprocessor:polynomial:interaction_only"],
        random_state=random_state
    )
    return (name, model)
