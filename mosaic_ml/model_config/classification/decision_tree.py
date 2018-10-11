from sklearn.tree import DecisionTreeClassifier

def get_model(name, config):
    model = DecisionTreeClassifier(
        criterion=config["classifier:decision_tree:criterion"],
        max_depth=float(config["classifier:decision_tree:max_depth"]),
        max_features=float(config["classifier:decision_tree:max_features"]),
        max_leaf_nodes=None,
        min_impurity_decrease=float(config["classifier:decision_tree:min_impurity_decrease"]),
        min_samples_leaf=int(config["classifier:decision_tree:min_samples_leaf"]),
        min_samples_split=int(config["classifier:decision_tree:min_samples_split"]),
        min_weight_fraction_leaf=float(config["classifier:decision_tree:min_weight_fraction_leaf"])
    )
    return (name, model)