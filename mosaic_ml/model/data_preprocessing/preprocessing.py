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
