from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def get_model(name, config):
    model = QuadraticDiscriminantAnalysis(
        reg_param=float(config["classifier:qda:reg_param"])
    )
    return (name, model)