from mosaic.simulation.parameter import Parameter
from mosaic.simulation.scenario import WorkflowListTask


def get_configuration_DummyClassifier():
    DummyClassifier = WorkflowListTask(is_ordered=False, name = "DummyClassifier",
                                  tasks = ["DummyClassifier__strategy"])
    sampler = {
         "DummyClassifier__strategy": Parameter("DummyClassifier__strategy", ["stratified", "most_frequent", "prior", "uniform"], "choice", "string"),
    }

    rules = []
    return DummyClassifier, sampler, rules
