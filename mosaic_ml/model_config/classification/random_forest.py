from sklearn.ensemble import RandomForestClassifier


def get_model(name, config):
    model = RandomForestClassifier(
        bootstrap=eval(config["classifier:random_forest:bootstrap"]),
        criterion=config["classifier:random_forest:criterion"],
        max_depth=None,
        max_features=float(config["classifier:random_forest:max_features"]),
        max_leaf_nodes=None,
        min_impurity_decrease=float(config["classifier:random_forest:min_impurity_decrease"]),
        min_samples_leaf=int(config["classifier:random_forest:min_samples_leaf"]),
        min_samples_split=int(config["classifier:random_forest:min_samples_split"]),
        # TODO min_weight_fraction_leaf=float(config["classifier:random_forest:min_weight_fraction_leaf"]),
        n_estimators=int(config["classifier:random_forest:n_estimators"])
    )
    return (name, model)
