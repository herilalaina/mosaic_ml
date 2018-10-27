from sklearn.linear_model import SGDClassifier


def get_model(name, config):
    model = SGDClassifier(
        alpha=float(config["classifier:sgd:alpha"]),
        average=eval(config["classifier:sgd:average"]),
        learning_rate=config["classifier:sgd:learning_rate"],
        fit_intercept=eval(config["classifier:sgd:fit_intercept"]),
        loss=config["classifier:sgd:loss"],
        penalty=config["classifier:sgd:penalty"],
        tol=float(config["classifier:sgd:tol"]),
        eta0=0.1,
        n_jobs=-1
    )
    return (name, model)
