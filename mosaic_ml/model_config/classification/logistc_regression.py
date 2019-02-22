from sklearn.linear_model import LogisticRegression


def get_model(name, config, random_state):
    return (name, LogisticRegression(penalty=config["classifier:logistic_regression:penalty"]))
