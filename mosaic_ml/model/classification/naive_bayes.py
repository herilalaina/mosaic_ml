from mosaic.space import Parameter
from mosaic.scenario import ListTask
from mosaic.rules import ChildRule, ValueRule


def get_configuration_GaussianNB():
    gaussianNB = ListTask(is_ordered=False, name = "GaussianNB",
                                  tasks = ["GaussianNB__priors"])
    sampler = {
        "GaussianNB__priors": Parameter("GaussianNB__priors", None, "constant", "array")
    }
    return gaussianNB, sampler, []

def get_configuration_MultinomialNB():
    multinomialNB = ListTask(is_ordered=False, name = "MultinomialNB",
                                  tasks = ["MultinomialNB__alpha", "MultinomialNB__class_prior"])
    sampler = {
        "MultinomialNB__alpha": Parameter("MultinomialNB__alpha", [0, 1], "uniform", "float"),
        "MultinomialNB__class_prior": Parameter("MultinomialNB__class_prior", None, "constant", "array")
    }
    return multinomialNB, sampler, []
