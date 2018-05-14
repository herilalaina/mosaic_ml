from mosaic.space import Parameter
from mosaic.scenario import ListTask
from mosaic.rules import ChildRule, ValueRule


def get_configuration_DecisionTreeClassifier():
    DecisionTreeClassifier = ListTask(is_ordered=False, name = "DecisionTreeClassifier",
                                  tasks = ["DecisionTreeClassifier__criterion",
                                           "DecisionTreeClassifier__splitter",
                                           "DecisionTreeClassifier__max_depth",
                                           "DecisionTreeClassifier__min_samples_split",
                                           "DecisionTreeClassifier__min_samples_leaf",
                                           "DecisionTreeClassifier__min_weight_fraction_leaf",
                                           "DecisionTreeClassifier__max_features",
                                           "DecisionTreeClassifier__max_leaf_nodes",
                                           "DecisionTreeClassifier__min_impurity_decrease",
                                           "DecisionTreeClassifier__class_weight"])
    sampler = {
             "DecisionTreeClassifier__criterion": Parameter("DecisionTreeClassifier__criterion", ["gini", "entropy"], "choice", "string"),
             "DecisionTreeClassifier__splitter": Parameter("DecisionTreeClassifier__splitter", ["best", "random"], "choice", "string"),
             "DecisionTreeClassifier__max_depth": Parameter("DecisionTreeClassifier__max_depth", [1, 10], "uniform", "int"),
             "DecisionTreeClassifier__min_samples_split": Parameter("DecisionTreeClassifier__min_samples_split", [0.01, 0.2], "uniform", "float"),
             "DecisionTreeClassifier__min_samples_leaf": Parameter("DecisionTreeClassifier__min_samples_leaf", [0.01, 0.2], "uniform", "float"),
             "DecisionTreeClassifier__min_weight_fraction_leaf": Parameter("DecisionTreeClassifier__min_weight_fraction_leaf", 0.0, "constant", "float"),
             "DecisionTreeClassifier__max_features": Parameter("DecisionTreeClassifier__max_features", ["auto", "sqrt", "log2", None], "choice", "string"),
             "DecisionTreeClassifier__max_leaf_nodes": Parameter("DecisionTreeClassifier__max_leaf_nodes", None, "constant", "string"),
             "DecisionTreeClassifier__min_impurity_decrease": Parameter("DecisionTreeClassifier__min_impurity_decrease", [0, 0.05], "uniform", "float"),
             "DecisionTreeClassifier__class_weight": Parameter("DecisionTreeClassifier__class_weight", "balanced", "constant", "string")
    }
    rules = []
    return DecisionTreeClassifier, sampler, rules


def get_configuration_ExtraTreeClassifier():
    ExtraTreeClassifier = ListTask(is_ordered=False, name = "ExtraTreeClassifier",
                                  tasks = ["ExtraTreeClassifier__criterion",
                                           "ExtraTreeClassifier__splitter",
                                           "ExtraTreeClassifier__max_depth",
                                           "ExtraTreeClassifier__min_samples_split",
                                           "ExtraTreeClassifier__min_samples_leaf",
                                           "ExtraTreeClassifier__min_weight_fraction_leaf",
                                           "ExtraTreeClassifier__max_features",
                                           "ExtraTreeClassifier__max_leaf_nodes",
                                           "ExtraTreeClassifier__min_impurity_decrease",
                                           "ExtraTreeClassifier__class_weight"])
    sampler = {
             "ExtraTreeClassifier__criterion": Parameter("ExtraTreeClassifier__criterion", ["gini", "entropy"], "choice", "string"),
             "ExtraTreeClassifier__splitter": Parameter("ExtraTreeClassifier__splitter", ["best", "random"], "choice", "string"),
             "ExtraTreeClassifier__max_depth": Parameter("ExtraTreeClassifier__max_depth", [1, 10], "uniform", "int"),
             "ExtraTreeClassifier__min_samples_split": Parameter("ExtraTreeClassifier__min_samples_split", [0.01, 0.2], "uniform", "float"),
             "ExtraTreeClassifier__min_samples_leaf": Parameter("ExtraTreeClassifier__min_samples_leaf", [0.01, 0.2], "uniform", "float"),
             "ExtraTreeClassifier__min_weight_fraction_leaf": Parameter("ExtraTreeClassifier__min_weight_fraction_leaf", 0.0, "constant", "float"),
             "ExtraTreeClassifier__max_features": Parameter("ExtraTreeClassifier__max_features", ["auto", "sqrt", "log2", None], "choice", "string"),
             "ExtraTreeClassifier__max_leaf_nodes": Parameter("ExtraTreeClassifier__max_leaf_nodes", None, "constant", "string"),
             "ExtraTreeClassifier__min_impurity_decrease": Parameter("ExtraTreeClassifier__min_impurity_decrease", [0, 0.05], "uniform", "float"),
             "ExtraTreeClassifier__class_weight": Parameter("ExtraTreeClassifier__class_weight", "balanced", "constant", "string")
    }
    rules = []
    return ExtraTreeClassifier, sampler, rules
