from sklearn.ensemble import ExtraTreesClassifier


def get_model(name, config):
    model = ExtraTreesClassifier(
        bootstrap=bool(config["classifier:extra_trees:bootstrap"]),
        criterion=config["classifier:extra_trees:criterion"],
        max_depth=None,
        max_features=float(config["classifier:extra_trees:max_features"]),
        max_leaf_nodes=None,
        min_impurity_decrease=float(config["classifier:extra_trees:min_impurity_decrease"]),
        min_samples_leaf=int(config["classifier:extra_trees:min_samples_leaf"]),
        min_samples_split=int(config["classifier:extra_trees:min_samples_split"]),
        min_weight_fraction_leaf=float(config["classifier:extra_trees:min_weight_fraction_leaf"]),
        n_estimators=int(config["classifier:extra_trees:n_estimators"])
    )
    return (name, model)
