import unittest

from mosaic.space import Parameter
from mosaic_ml.model_config.classification.linear_model import get_configuration_LogisticRegression, get_configuration_SGDClassifier

class TypoTest(unittest.TestCase):

    def test_length_sampler(self):
        for get_configuration in [get_configuration_LogisticRegression, get_configuration_SGDClassifier]:
            scenario, sampler, rules = get_configuration()
            for task in scenario.queue:
                assert(task in sampler)
            for name, val in sampler.items():
                assert(val.name == name)
                assert(val.type_sampling in ["uniform", "choice", "constant", "log_uniform"])
                assert(val.type in ["float", "int", "string", "func", "bool"])
                if val.type_sampling == "uniform":
                    assert(len(val.value_list) == 2)
                if val.type_sampling == "choice":
                    assert(len(val.value_list) > 0)
