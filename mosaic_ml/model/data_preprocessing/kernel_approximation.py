from mosaic.space import Parameter
from mosaic.scenario import ListTask
from mosaic.rules import ChildRule, ValueRule


def get_configuration_RBFSampler():
    RBFSampler = ListTask(is_ordered=False, name = "RBFSampler",
                                  tasks = ["RBFSampler__gamma",
                                           "RBFSampler__n_components"])
    sampler = {
         "RBFSampler__gamma": Parameter("RBFSampler__gamma", [3e-8, 50], "uniform", "float"),
         "RBFSampler__n_components": Parameter("RBFSampler__n_components", [2, 30], "uniform", "int")
    }

    rules = []
    return RBFSampler, sampler, rules
