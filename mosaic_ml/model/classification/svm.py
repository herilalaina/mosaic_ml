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
           "LinearSVC__dual": Parameter("LinearSVC__dual", False, "constant", "bool"),
           "LinearSVC__tol": Parameter("LinearSVC__tol", [1e-5, 1e-1], "log_uniform", "bool"),
           "LinearSVC__C": Parameter("LinearSVC__C", [0.03125, 30], "log_uniform", "float"),
           "LinearSVC__class_weight": Parameter("LinearSVC__class_weight", "balanced", "constant", "string"),
           "LinearSVC__max_iter": Parameter("LinearSVC__max_iter", [1, 100], "uniform", "int")
    }
    rules = [
        ValueRule([("LinearSVC__loss", "hinge"), ("LinearSVC__penalty", "l2"), ("LinearSVC__dual", True)]),
        ValueRule([("LinearSVC__loss", "hinge"), ("LinearSVC__penalty", "l2")])
    ]
    return LinearSVC, sampler, rules


def get_configuration_NuSVC():
    NuSVC = ListTask(is_ordered=False, name = "NuSVC",
                                  tasks = ["NuSVC__nu",
                                           "NuSVC__kernel",
                                           "NuSVC__degree",
                                           "NuSVC__gamma",
                                           "NuSVC__coef0",
                                           "NuSVC__probability",
                                           "NuSVC__shrinking",
                                           "NuSVC__tol",
                                           "NuSVC__class_weight",
                                           "NuSVC__max_iter",
                                           "NuSVC__decision_function_shape"])
    sampler = {
           "NuSVC__nu": Parameter("NuSVC__nu", [0.25, 1], "uniform", "float"),
           "NuSVC__kernel": Parameter("NuSVC__kernel", ["linear", "poly", "rbf", "sigmoid"], "choice", "string"),
           "NuSVC__degree": Parameter("NuSVC__degree", [2, 3, 4, 5], "choice", "int"),
           "NuSVC__gamma": Parameter("NuSVC__gamma", [3e-8, 50], "log_uniform", "float"),
           "NuSVC__coef0": Parameter("NuSVC__coef0", [0, 1], "uniform", "float"),
           "NuSVC__probability": Parameter("NuSVC__probability", False, "constant", "bool"),
           "NuSVC__shrinking": Parameter("NuSVC__shrinking", [True, False], "choice", "bool"),
           "NuSVC__tol": Parameter("NuSVC__tol", [0, 0.1], "uniform", "float"),
           "NuSVC__class_weight": Parameter("NuSVC__class_weight", "balanced", "constant", "string"),
           "NuSVC__max_iter": Parameter("NuSVC__max_iter", [1, 100], "uniform", "int"),
           "NuSVC__decision_function_shape": Parameter("NuSVC__decision_function_shape", "ovr", "constant", "string")
    }

    rules = [
        ChildRule(applied_to = ["NuSVC__degree"], parent = "NuSVC__kernel", value = ["poly"]),
        ChildRule(applied_to = ["NuSVC__gamma"], parent = "NuSVC__kernel", value = ["poly", "rbf", "sigmoid"]),
        ChildRule(applied_to = ["NuSVC__coef0"], parent = "NuSVC__kernel", value = ["poly", "sigmoid"]),
    ]
    return NuSVC, sampler, rules



def get_configuration_SVC():
    rules = [
        ChildRule(applied_to = ["SVC__degree"], parent = "SVC__kernel", value = ["poly"]),
        ChildRule(applied_to = ["SVC__gamma"], parent = "SVC__kernel", value = ["poly", "rbf", "sigmoid"]),
        ChildRule(applied_to = ["SVC__coef0"], parent = "SVC__kernel", value = ["poly", "sigmoid"]),
    ]
    SVC = ListTask(is_ordered=False, name = "SVC",
                                  tasks = ["SVC__C",
                                           "SVC__kernel",
                                           "SVC__degree",
                                           "SVC__gamma",
                                           "SVC__coef0",
                                           #"SVC__probability",
                                           "SVC__shrinking",
                                           "SVC__tol",
                                           "SVC__class_weight",
                                           #"SVC__max_iter",
                                           "SVC__decision_function_shape"],
                                    rules = rules)
    sampler = {
           "SVC__C": Parameter("SVC__C", [0.03125, 30], "log_uniform", "float"),
           "SVC__kernel": Parameter("SVC__kernel", ["linear", "poly", "rbf", "sigmoid"], "choice", "string"),
           "SVC__degree": Parameter("SVC__degree", [2, 3, 4, 5], "choice", "int"),
           "SVC__gamma": Parameter("SVC__gamma", [3e-8, 50], "log_uniform", "float"),
           "SVC__coef0": Parameter("SVC__coef0", [-1, 1], "uniform", "float"),
           #"SVC__probability": Parameter("SVC__probability", False, "constant", "bool"),
           "SVC__shrinking": Parameter("SVC__shrinking", [True, False], "choice", "bool"),
           "SVC__tol": Parameter("SVC__tol", [1e-5, 1e-1], "log_uniform", "float"),
           "SVC__class_weight": Parameter("SVC__class_weight", "balanced", "constant", "string"),
           #"SVC__max_iter": Parameter("SVC__max_iter", [1, 100], "uniform", "int"),
           "SVC__decision_function_shape": Parameter("SVC__decision_function_shape", "ovr", "constant", "string")
    }
    return SVC, sampler, rules
