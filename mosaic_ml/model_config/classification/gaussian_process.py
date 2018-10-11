from mosaic.simulation.parameter import Parameter
from mosaic.simulation.scenario import WorkflowListTask


def get_configuration_GaussianProcessClassifier():
    GaussianProcessClassifier = WorkflowListTask(is_ordered=False, name = "GaussianProcessClassifier",
                                  tasks = ["GaussianProcessClassifier__kernel",
                                           "GaussianProcessClassifier__optimizer",
                                           "GaussianProcessClassifier__n_restarts_optimizer",
                                           "GaussianProcessClassifier__max_iter_predict",
                                           "GaussianProcessClassifier__warm_start",
                                           "GaussianProcessClassifier__n_jobs"])
    sampler = {
              "GaussianProcessClassifier__kernel": Parameter("GaussianProcessClassifier__kernel", None, "constant", "string"),
              "GaussianProcessClassifier__optimizer": Parameter("GaussianProcessClassifier__optimizer", "fmin_l_bfgs_b", "constant", "string"),
              "GaussianProcessClassifier__n_restarts_optimizer": Parameter("GaussianProcessClassifier__n_restarts_optimizer", [0, 6], "uniform", "int"),
              "GaussianProcessClassifier__max_iter_predict": Parameter("GaussianProcessClassifier__max_iter_predict", [1, 100], "uniform", "int"),
              "GaussianProcessClassifier__warm_start": Parameter("GaussianProcessClassifier__warm_start", True, "constant", "bool"),
              "GaussianProcessClassifier__n_jobs": Parameter("GaussianProcessClassifier__n_jobs", -1, "constant", "int")
    }
    rules = []
    return GaussianProcessClassifier, sampler, rules
