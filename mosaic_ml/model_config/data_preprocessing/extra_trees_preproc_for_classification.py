from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def get_model(name, config):
    model = SelectFromModel(
        estimator=ExtraTreesClassifier(
            bootstrap=bool(config["preprocessor:extra_trees_preproc_for_classification:bootstrap"]),
            criterion=config["preprocessor:extra_trees_preproc_for_classification:criterion"],
            max_depth=None,
            max_features=float(config["preprocessor:extra_trees_preproc_for_classification:max_features"]),
            max_leaf_nodes=None,
            min_impurity_decrease=float(
                config["preprocessor:extra_trees_preproc_for_classification:min_impurity_decrease"]),
            min_samples_leaf=int(config["preprocessor:extra_trees_preproc_for_classification:min_samples_leaf"]),
            min_samples_split=int(config["preprocessor:extra_trees_preproc_for_classification:min_samples_split"]),
            min_weight_fraction_leaf=float(
                config["preprocessor:extra_trees_preproc_for_classification:min_weight_fraction_leaf"]),
            n_estimators=int(config["preprocessor:extra_trees_preproc_for_classification:n_estimators"])
        )
    )
    return (name, model)
