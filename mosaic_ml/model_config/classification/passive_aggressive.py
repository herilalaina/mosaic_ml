from sklearn.linear_model import PassiveAggressiveClassifier


def get_model(name, config):
    model = PassiveAggressiveClassifier(
        C=float(config["classifier:passive_aggressive:C"]),
        average=eval(config["classifier:passive_aggressive:average"]),
        fit_intercept=eval(config["classifier:passive_aggressive:fit_intercept"]),
        loss=config["classifier:passive_aggressive:loss"],
        tol=float(config["classifier:passive_aggressive:tol"]),
        n_jobs=-1
    )
    return (name, model)
