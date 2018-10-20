from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def get_model(name, config):
    model = AdaBoostClassifier(
        algorithm=config["classifier:adaboost:algorithm"],
        learning_rate=float(config["classifier:adaboost:learning_rate"]),
        base_estimator=DecisionTreeClassifier(max_depth=float(config["classifier:adaboost:max_depth"])),
        n_estimators=int(config["classifier:adaboost:n_estimators"])
    )
    return (name, model)
