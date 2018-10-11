from mosaic.simulation.parameter import Parameter
from mosaic.simulation.scenario import WorkflowListTask


def get_configuration_PolynomialFeatures():
    PolynomialFeatures = WorkflowListTask(is_ordered=False, name = "PolynomialFeatures",
                                  tasks = ["PolynomialFeatures__degree"])
    sampler = {
         "PolynomialFeatures__degree": Parameter("PolynomialFeatures__degree", [2, 6], 'uniform', "int"),
    }

    rules = []
    return PolynomialFeatures, sampler, rules


def get_configuration_FunctionTransformer():
    FunctionTransformer = WorkflowListTask(is_ordered=False, name = "FunctionTransformer",
                                  tasks = ["FunctionTransformer__func"])
    sampler = {
         "FunctionTransformer__func": Parameter("FunctionTransformer__func", None, 'constant', "string"),
    }

    rules = []
    return FunctionTransformer, sampler, rules
