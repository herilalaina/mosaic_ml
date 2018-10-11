from sklearn.ensemble import GradientBoostingClassifier

def get_model(name, config):
    model = GradientBoostingClassifier(
        criterion=config["classifier:gradient_boosting:criterion"],
        learning_rate=float(config["classifier:gradient_boosting:learning_rate"]),
        loss=config["classifier:gradient_boosting:loss"],
        max_depth=int(config["classifier:gradient_boosting:max_depth"]),
        max_features=float(config["classifier:gradient_boosting:max_features"]),
        max_leaf_nodes=None,
        min_impurity_decrease=float(config["classifier:gradient_boosting:min_impurity_decrease"]),
        min_samples_leaf=int(config["classifier:gradient_boosting:min_samples_leaf"]),
        min_samples_split=int(config["classifier:gradient_boosting:min_samples_split"]),
        min_weight_fraction_leaf=float(config["classifier:gradient_boosting:min_weight_fraction_leaf"]),
        n_estimators=int(config["classifier:gradient_boosting:n_estimators"]),
        subsample=float(config["classifier:gradient_boosting:subsample"])
    )
    return (name, model)