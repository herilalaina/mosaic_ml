from mosaic.space import Parameter
from mosaic.scenario import ListTask
from mosaic.rules import ChildRule, ValueRule


def get_configuration_LinearSVC():
    LinearSVC = ListTask(is_ordered=False, name = "LinearSVC",
                                  tasks = ["LinearSVC__penalty",
                                           "LinearSVC__loss",
                                           "LinearSVC__dual",
                                           "LinearSVC__tol",
                                           "LinearSVC__C",
                                           "LinearSVC__class_weight",
                                           "LinearSVC__max_iter"])
    sampler = {
           "LinearSVC__penalty": Parameter("LinearSVC__penalty", ["l1", "l2"], "choice", "string"),
           "LinearSVC__loss": Parameter("LinearSVC__loss", ["hinge", "squared_hinge"], "choice", "string"),
           "LinearSVC__dual": Parameter("LinearSVC__dual", [True, False], "choice", "bool"),
           "LinearSVC__tol": Parameter("LinearSVC__tol", [0, 0.5], "uniform", "bool"),
           "LinearSVC__C": Parameter("LinearSVC__C", [1e-5, 10], "log_uniform", "float"),
           "LinearSVC__class_weight": Parameter("LinearSVC__class_weight", "balanced", "constant", "string"),
           "LinearSVC__max_iter": Parameter("LinearSVC__max_iter", [1, 100], "uniform", "int")
    }
    rules = [
        #ChildRule(applied_to = ["LinearSVC__penalty"], parent = "LinearSVC__loss", value = ["squared_hinge"]),
        ValueRule([("LinearSVC__loss", "hinge"), ("LinearSVC__penalty", "l2"), ("LinearSVC__dual", True)]),
        ValueRule([("LinearSVC__loss", "hinge"), ("LinearSVC__penalty", "l2")])
    ]
    return LinearSVC, sampler, rules
