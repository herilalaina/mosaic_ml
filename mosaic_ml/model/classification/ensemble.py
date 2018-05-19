from mosaic.space import Parameter
from mosaic.scenario import ListTask
from mosaic.rules import ChildRule, ValueRule

from sklearn.tree import DecisionTreeClassifier


def get_configuration_AdaBoostClassifier():
    AdaBoostClassifier = ListTask(is_ordered=False, name = "AdaBoostClassifier",
                                  tasks = ["AdaBoostClassifier__base_estimator",
                                            "AdaBoostClassifier__n_estimators",
                                            "AdaBoostClassifier__learning_rate",
                                            "AdaBoostClassifier__algorithm"])
    sampler = {
          "AdaBoostClassifier__base_estimator": Parameter("AdaBoostClassifier__base_estimator", DecisionTreeClassifier(), "constant", "func"),
          "AdaBoostClassifier__n_estimators": Parameter("AdaBoostClassifier__n_estimators", [50, 500], "uniform", "int"),
          "AdaBoostClassifier__learning_rate": Parameter("AdaBoostClassifier__learning_rate", [0.01, 2], "uniform", "float"),
          "AdaBoostClassifier__algorithm": Parameter("AdaBoostClassifier__algorithm", ["SAMME", "SAMME.R"], "choice", "string")
    }
    return AdaBoostClassifier, sampler, []


def get_configuration_BaggingClassifier():
    BaggingClassifier = ListTask(is_ordered=False, name = "BaggingClassifier",
                                  tasks = ["BaggingClassifier__base_estimator",
                                            "BaggingClassifier__n_estimators",
                                            "BaggingClassifier__max_samples",
                                            "BaggingClassifier__max_features",
                                            "BaggingClassifier__bootstrap",
                                            "BaggingClassifier__bootstrap_features",
                                            "BaggingClassifier__oob_score",
                                            "BaggingClassifier__warm_start",
                                            "BaggingClassifier__n_jobs"])
    sampler = {
          "BaggingClassifier__base_estimator": Parameter("BaggingClassifier__base_estimator", DecisionTreeClassifier(), "constant", "func"),
          "BaggingClassifier__n_estimators": Parameter("BaggingClassifier__n_estimators", [50, 500], "uniform", "int"),
          "BaggingClassifier__max_samples": Parameter("BaggingClassifier__max_samples", [0.1, 0.5], "uniform", "float"),
          "BaggingClassifier__max_features": Parameter("BaggingClassifier__max_features", [0.1, 0.5], "uniform", "float"),
          "BaggingClassifier__bootstrap": Parameter("BaggingClassifier__bootstrap", [True, False], "choice", "bool"),
          "BaggingClassifier__bootstrap_features": Parameter("BaggingClassifier__bootstrap_features", [True, False], "choice", "bool"),
          "BaggingClassifier__oob_score": Parameter("BaggingClassifier__oob_score", [True, False], "choice", "bool"),
          "BaggingClassifier__warm_start":Parameter("BaggingClassifier__warm_start", [True, False], "choice", "bool"),
          "BaggingClassifier__n_jobs": Parameter("BaggingClassifier__n_jobs", -1, "constant", "int")
    }

    rules = [
        ValueRule([("BaggingClassifier__oob_score", True), ("BaggingClassifier__warm_start", False), ("BaggingClassifier__bootstrap", True)])
    ]
    return BaggingClassifier, sampler, rules


def get_configuration_ExtraTreesClassifier():
    ExtraTreesClassifier = ListTask(is_ordered=False, name = "ExtraTreesClassifier",
                                  tasks = ["ExtraTreesClassifier__n_estimators",
                                           "ExtraTreesClassifier__criterion",
                                           "ExtraTreesClassifier__max_depth",
                                           "ExtraTreesClassifier__min_samples_split",
                                           "ExtraTreesClassifier__min_samples_leaf",
                                           "ExtraTreesClassifier__min_weight_fraction_leaf",
                                           "ExtraTreesClassifier__max_features",
                                           "ExtraTreesClassifier__max_leaf_nodes",
                                           #"ExtraTreesClassifier__min_impurity_decrease",
                                           "ExtraTreesClassifier__class_weight",
                                           #"ExtraTreesClassifier__bootstrap",
                                           "ExtraTreesClassifier__n_jobs",
                                           #"ExtraTreesClassifier__oob_score",
                                           #"ExtraTreesClassifier__warm_start"
                                           ])
    sampler = {
             "ExtraTreesClassifier__n_estimators": Parameter("ExtraTreesClassifier__n_estimators", [100, 110], "uniform", "int"),
             "ExtraTreesClassifier__criterion": Parameter("ExtraTreesClassifier__criterion", ["gini", "entropy"], "choice", "string"),
             "ExtraTreesClassifier__max_depth": Parameter("ExtraTreesClassifier__max_depth", [1, 10], "uniform", "int"),
             "ExtraTreesClassifier__min_samples_split": Parameter("ExtraTreesClassifier__min_samples_split", [2, 20], "uniform", "float"),
             "ExtraTreesClassifier__min_samples_leaf": Parameter("ExtraTreesClassifier__min_samples_leaf", [1, 20], "uniform", "int"),
             "ExtraTreesClassifier__min_weight_fraction_leaf": Parameter("ExtraTreesClassifier__min_weight_fraction_leaf", 0.0, "constant", "float"),
             "ExtraTreesClassifier__max_features": Parameter("ExtraTreesClassifier__max_features", ["auto", "sqrt", "log2", None], "choice", "string"),
             "ExtraTreesClassifier__max_leaf_nodes": Parameter("ExtraTreesClassifier__max_leaf_nodes", None, "constant", "string"),
             #"ExtraTreesClassifier__min_impurity_decrease": Parameter("ExtraTreesClassifier__min_impurity_decrease", [0, 0.05], "uniform", "float"),
             "ExtraTreesClassifier__class_weight": Parameter("ExtraTreesClassifier__class_weight", "balanced", "constant", "string"),
             #"ExtraTreesClassifier__bootstrap": Parameter("ExtraTreesClassifier__bootstrap", [True, False], "choice", "bool"),
             "ExtraTreesClassifier__n_jobs": Parameter("ExtraTreesClassifier__n_jobs", -1, "constant", "int"),
             #"ExtraTreesClassifier__oob_score": Parameter("ExtraTreesClassifier__oob_score", [True, False], "choice", "bool"),
             #"ExtraTreesClassifier__warm_start": Parameter("ExtraTreesClassifier__warm_start", [True, False], "choice", "bool")
    }

    rules = [
        # ValueRule([("ExtraTreesClassifier__oob_score", True), ("ExtraTreesClassifier__bootstrap", True), ("ExtraTreesClassifier__warm_start", False)])
    ]
    return ExtraTreesClassifier, sampler, rules


def get_configuration_GradientBoostingClassifier():
    GradientBoostingClassifier = ListTask(is_ordered=False, name = "GradientBoostingClassifier",
                                  tasks = ["GradientBoostingClassifier__loss",
                                           "GradientBoostingClassifier__learning_rate",
                                           "GradientBoostingClassifier__n_estimators",
                                           "GradientBoostingClassifier__max_depth",
                                           "GradientBoostingClassifier__criterion",
                                           "GradientBoostingClassifier__min_samples_split",
                                           "GradientBoostingClassifier__min_samples_leaf",
                                           "GradientBoostingClassifier__min_weight_fraction_leaf",
                                           "GradientBoostingClassifier__subsample",
                                           "GradientBoostingClassifier__max_features",
                                           "GradientBoostingClassifier__max_leaf_nodes",
                                           #"GradientBoostingClassifier__min_impurity_decrease",
                                           #"GradientBoostingClassifier__warm_start"
                                           ])
    sampler = {
             "GradientBoostingClassifier__loss": Parameter("GradientBoostingClassifier__loss", "deviance", "constant", "string"),
             "GradientBoostingClassifier__learning_rate": Parameter("GradientBoostingClassifier__learning_rate", [0.001, 1], "uniform", "float"),
             "GradientBoostingClassifier__n_estimators": Parameter("GradientBoostingClassifier__n_estimators", [50, 500], "uniform", "int"),
             "GradientBoostingClassifier__max_depth": Parameter("GradientBoostingClassifier__max_depth", [1, 10], "uniform", "int"),
             "GradientBoostingClassifier__criterion": Parameter("GradientBoostingClassifier__criterion", ["friedman_mse", "mse", "mae"], "choice", "string"),
             "GradientBoostingClassifier__min_samples_split": Parameter("GradientBoostingClassifier__min_samples_split", [2, 20], "uniform", "float"),
             "GradientBoostingClassifier__min_samples_leaf": Parameter("GradientBoostingClassifier__min_samples_leaf", [1, 20], "uniform", "int"),
             "GradientBoostingClassifier__min_weight_fraction_leaf": Parameter("GradientBoostingClassifier__min_weight_fraction_leaf", 0.0, "constant", "float"),
             "GradientBoostingClassifier__subsample": Parameter("GradientBoostingClassifier__subsample", [0.01, 1], "uniform", "float"),
             "GradientBoostingClassifier__max_features": Parameter("GradientBoostingClassifier__max_features", ["auto", "sqrt", "log2", None], "choice", "string"),
             "GradientBoostingClassifier__max_leaf_nodes": Parameter("GradientBoostingClassifier__max_leaf_nodes", None, "constant", "string"),
             #"GradientBoostingClassifier__min_impurity_decrease": Parameter("GradientBoostingClassifier__min_impurity_decrease", [0, 0.05], "uniform", "float"),
             #"GradientBoostingClassifier__warm_start": Parameter("GradientBoostingClassifier__warm_start", [True, False], "choice", "bool")
    }

    rules = []
    return GradientBoostingClassifier, sampler, rules


def get_configuration_RandomForestClassifier():
    RandomForestClassifier = ListTask(is_ordered=False, name = "RandomForestClassifier",
                                  tasks = ["RandomForestClassifier__n_estimators",
                                           "RandomForestClassifier__criterion",
                                           "RandomForestClassifier__max_depth",
                                           "RandomForestClassifier__min_samples_split",
                                           "RandomForestClassifier__min_samples_leaf",
                                           "RandomForestClassifier__min_weight_fraction_leaf",
                                           "RandomForestClassifier__max_features",
                                           "RandomForestClassifier__max_leaf_nodes",
                                           #"RandomForestClassifier__min_impurity_decrease",
                                           "RandomForestClassifier__class_weight",
                                           "RandomForestClassifier__bootstrap",
                                           "RandomForestClassifier__n_jobs",
                                           #"RandomForestClassifier__oob_score",
                                           #"RandomForestClassifier__warm_start"
                                           ])
    sampler = {
             "RandomForestClassifier__n_estimators": Parameter("RandomForestClassifier__n_estimators", [100, 110], "uniform", "int"),
             "RandomForestClassifier__criterion": Parameter("RandomForestClassifier__criterion", ["gini", "entropy"], "choice", "string"),
             "RandomForestClassifier__max_depth": Parameter("RandomForestClassifier__max_depth", None, "constant", "string"),
             "RandomForestClassifier__min_samples_split": Parameter("RandomForestClassifier__min_samples_split", [2, 20], "uniform", "float"),
             "RandomForestClassifier__min_samples_leaf": Parameter("RandomForestClassifier__min_samples_leaf", [1, 20], "uniform", "int"),
             "RandomForestClassifier__min_weight_fraction_leaf": Parameter("RandomForestClassifier__min_weight_fraction_leaf", 0.0, "constant", "float"),
             "RandomForestClassifier__max_features": Parameter("RandomForestClassifier__max_features", ["auto", "sqrt", "log2", None], "choice", "string"),
             "RandomForestClassifier__max_leaf_nodes": Parameter("RandomForestClassifier__max_leaf_nodes", None, "constant", "string"),
             #"RandomForestClassifier__min_impurity_decrease": Parameter("RandomForestClassifier__min_impurity_decrease", [0, 0.05], "uniform", "float"),
             "RandomForestClassifier__class_weight": Parameter("RandomForestClassifier__class_weight", "balanced", "constant", "string"),
             "RandomForestClassifier__bootstrap": Parameter("RandomForestClassifier__bootstrap", [True, False], "choice", "bool"),
             "RandomForestClassifier__n_jobs": Parameter("RandomForestClassifier__n_jobs", -1, "constant", "int"),
             #"RandomForestClassifier__oob_score": Parameter("RandomForestClassifier__oob_score", [True, False], "choice", "bool"),
             #"RandomForestClassifier__warm_start": Parameter("RandomForestClassifier__warm_start", [True, False], "choice", "bool")
    }

    rules = [
        # ValueRule([("RandomForestClassifier__oob_score", True), ("RandomForestClassifier__bootstrap", True), ("RandomForestClassifier__warm_start", False)])
    ]
    return RandomForestClassifier, sampler, rules
