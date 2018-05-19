from mosaic.space import Parameter
from mosaic.scenario import ListTask
from mosaic.rules import ChildRule, ValueRule


def get_configuration_PolynomialFeatures():
    PolynomialFeatures = ListTask(is_ordered=False, name = "PolynomialFeatures",
                                  tasks = ["PolynomialFeatures__degree"])
    sampler = {
         "PolynomialFeatures__degree": Parameter("PolynomialFeatures__degree", [2, 6], 'uniform', "int"),
    }

    rules = []
    return PolynomialFeatures, sampler, rules


def get_configuration_FunctionTransformer():
    FunctionTransformer = ListTask(is_ordered=False, name = "FunctionTransformer",
                                  tasks = ["FunctionTransformer__func"])
    sampler = {
         "FunctionTransformer__func": Parameter("FunctionTransformer__func", None, 'constant', "string"),
    }

    rules = []
    return FunctionTransformer, sampler, rules
