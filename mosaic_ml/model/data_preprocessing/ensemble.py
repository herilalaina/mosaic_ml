from mosaic.space import Parameter
from mosaic.scenario import ListTask
from mosaic.rules import ChildRule, ValueRule


def get_configuration_RandomTreesEmbedding():
    RandomTreesEmbedding = ListTask(is_ordered=False, name = "RandomTreesEmbedding",
                                  tasks = ["RandomTreesEmbedding__n_estimators",
                                           "RandomTreesEmbedding__max_depth",
                                           "RandomTreesEmbedding__min_samples_split",
                                           "RandomTreesEmbedding__min_samples_leaf",
                                           "RandomTreesEmbedding__min_weight_fraction_leaf",
                                           #"RandomTreesEmbedding__max_features",
                                           "RandomTreesEmbedding__max_leaf_nodes",
                                           "RandomTreesEmbedding__min_impurity_decrease",
                                           #"RandomTreesEmbedding__class_weight",
                                           #"RandomTreesEmbedding__bootstrap",
                                           "RandomTreesEmbedding__n_jobs",
                                           #"RandomTreesEmbedding__warm_start"
                                           ])
    sampler = {
             "RandomTreesEmbedding__n_estimators": Parameter("RandomTreesEmbedding__n_estimators", [10, 100], "uniform", "int"),
             "RandomTreesEmbedding__max_depth": Parameter("RandomTreesEmbedding__max_depth", [2, 10], "uniform", "int"),
             "RandomTreesEmbedding__min_samples_split": Parameter("RandomTreesEmbedding__min_samples_split", [2, 20], "uniform", "int"),
             "RandomTreesEmbedding__min_samples_leaf": Parameter("RandomTreesEmbedding__min_samples_leaf", [1, 20], "uniform", "int"),
             "RandomTreesEmbedding__min_weight_fraction_leaf": Parameter("RandomTreesEmbedding__min_weight_fraction_leaf", 1.0, "constant", "float"),
             #"RandomTreesEmbedding__max_features": Parameter("RandomTreesEmbedding__max_features", ["auto", "sqrt", "log2", None], "choice", "string"),
             "RandomTreesEmbedding__max_leaf_nodes": Parameter("RandomTreesEmbedding__max_leaf_nodes", None, "constant", "string"),
             "RandomTreesEmbedding__min_impurity_decrease": Parameter("RandomTreesEmbedding__min_impurity_decrease", [0, 0.05], "uniform", "float"),
             #"RandomTreesEmbedding__class_weight": Parameter("RandomTreesEmbedding__class_weight", "balanced", "constant", "string"),
             #"RandomTreesEmbedding__bootstrap": Parameter("RandomTreesEmbedding__bootstrap", [True, False], "choice", "bool"),
             "RandomTreesEmbedding__n_jobs": Parameter("RandomTreesEmbedding__n_jobs", -1, "constant", "int"),
             #"RandomTreesEmbedding__warm_start": Parameter("RandomTreesEmbedding__warm_start", [True, False], "choice", "bool")
    }

    rules = []
    return RandomTreesEmbedding, sampler, rules
