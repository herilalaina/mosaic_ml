from mosaic.space import Parameter
from mosaic.scenario import ListTask
from mosaic.rules import ChildRule, ValueRule


def get_configuration_DummyClassifier():
    DummyClassifier = ListTask(is_ordered=False, name = "DummyClassifier",
                                  tasks = ["DummyClassifier__strategy"])
    sampler = {
         "DummyClassifier__strategy": Parameter("DummyClassifier__strategy", ["stratified", "most_frequent", "prior", "uniform"], "choice", "string"),
    }

    rules = []
    return DummyClassifier, sampler, rules
