from mosaic.simulation.parameter import Parameter
from mosaic.simulation.scenario import WorkflowListTask
from mosaic.simulation.rules import ChildRule, ValueRule


def get_configuration_LabelPropagation():
    LabelPropagation = WorkflowListTask(is_ordered=False, name = "LabelPropagation",
                                  tasks = ["LabelPropagation__kernel",
                                           "LabelPropagation__gamma",
                                           "LabelPropagation__n_neighbors",
                                           "LabelPropagation__max_iter",
                                           "LabelPropagation__n_jobs"])
    sampler = {
          "LabelPropagation__kernel": Parameter("LabelPropagation__kernel", ["knn", "rbf"], "choice", "string"),
          "LabelPropagation__gamma": Parameter("LabelPropagation__gamma", [3e-8, 8], "log_uniform", "float"),
          "LabelPropagation__n_neighbors": Parameter("LabelPropagation__n_neighbors", [1, 30], "uniform", "int"),
          "LabelPropagation__max_iter": Parameter("LabelPropagation__max_iter", [1, 100], "uniform", "int"),
          "LabelPropagation__n_jobs": Parameter("LabelPropagation__n_jobs", -1, "constant", "int"),
    }
    rules = [
        ChildRule(applied_to = ["LabelPropagation__gamma"], parent = "LabelPropagation__kernel", value = ["rbf"]),
        ChildRule(applied_to = ["LabelPropagation__n_neighbors"], parent = "LabelPropagation__kernel", value = ["knn"])
    ]
    return LabelPropagation, sampler, rules


def get_configuration_LabelSpreading():
    LabelSpreading = WorkflowListTask(is_ordered=False, name = "LabelSpreading",
                                  tasks = ["LabelSpreading__kernel",
                                           "LabelSpreading__gamma",
                                           "LabelSpreading__n_neighbors",
                                           "LabelSpreading__max_iter",
                                           "LabelSpreading__n_jobs"])
    sampler = {
          "LabelSpreading__kernel": Parameter("LabelSpreading__kernel", ["knn", "rbf"], "choice", "string"),
          "LabelSpreading__gamma": Parameter("LabelSpreading__gamma", [3e-8, 8], "log_uniform", "float"),
          "LabelSpreading__n_neighbors": Parameter("LabelSpreading__n_neighbors", [1, 30], "uniform", "int"),
          "LabelSpreading__max_iter": Parameter("LabelSpreading__max_iter", [1, 100], "uniform", "int"),
          "LabelSpreading__n_jobs": Parameter("LabelSpreading__n_jobs", -1, "constant", "int"),
    }
    rules = [
        ChildRule(applied_to = ["LabelSpreading__gamma"], parent = "LabelSpreading__kernel", value = ["rbf"]),
        ChildRule(applied_to = ["LabelSpreading__n_neighbors"], parent = "LabelSpreading__kernel", value = ["knn"])
    ]
    return LabelSpreading, sampler, rules
