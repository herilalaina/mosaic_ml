from mosaic.space import Parameter
from mosaic.scenario import ListTask
from mosaic.rules import ChildRule, ValueRule


def get_configuration_LinearDiscriminantAnalysis():
    LinearDiscriminantAnalysis = ListTask(is_ordered=False, name = "LinearDiscriminantAnalysis",
                                  tasks = ["LinearDiscriminantAnalysis__solver",
                                           "LinearDiscriminantAnalysis__shrinkage",
                                           "LinearDiscriminantAnalysis__tol"])
    sampler = {
        "LinearDiscriminantAnalysis__solver": Parameter("LinearDiscriminantAnalysis__solver", ["svd", "lsqr", "eigen"], "choice", "string"),
         "LinearDiscriminantAnalysis__shrinkage": Parameter("LinearDiscriminantAnalysis__shrinkage", ["auto", None], "choice", "string"),
         "LinearDiscriminantAnalysis__tol": Parameter("LinearDiscriminantAnalysis__tol", [0, 0.1], "uniform", "float"),
    }

    rules = [
        ChildRule(applied_to = ["LinearDiscriminantAnalysis__shrinkage"], parent = "LinearDiscriminantAnalysis__solver", value = ["lsqr", "eigen"])
    ]
    return LinearDiscriminantAnalysis, sampler, rules


def get_configuration_QuadraticDiscriminantAnalysis():
    QuadraticDiscriminantAnalysis = ListTask(is_ordered=False, name = "QuadraticDiscriminantAnalysis",
                                  tasks = ["QuadraticDiscriminantAnalysis__reg_param",
                                           "QuadraticDiscriminantAnalysis__tol"])
    sampler = {
         "QuadraticDiscriminantAnalysis__reg_param": Parameter("QuadraticDiscriminantAnalysis__reg_param", [0, 1], "uniform", "float"),
         "QuadraticDiscriminantAnalysis__tol": Parameter("QuadraticDiscriminantAnalysis__tol", [0, 0.1], "uniform", "float"),
    }

    rules = []
    return QuadraticDiscriminantAnalysis, sampler, rules
