from sklearn.preprocessing import PolynomialFeatures


def get_model(name, config):
    model = PolynomialFeatures(
        degree=int(config["preprocessor:polynomial:degree"]),
        include_bias=eval(config["preprocessor:polynomial:include_bias"]),
        interaction_only=eval(config["preprocessor:polynomial:interaction_only"])
    )
    return (name, model)
