from mosaic.simulation.parameter import Parameter
from mosaic.simulation.scenario import WorkflowListTask
from mosaic.simulation.rules import ChildRule


def get_configuration_KNeighborsClassifier():
    kNeighborsClassifier = WorkflowListTask(is_ordered=False, name = "KNeighborsClassifier",
                                  tasks = ["KNeighborsClassifier__n_neighbors",
                                           "KNeighborsClassifier__weights",
                                           "KNeighborsClassifier__algorithm",
                                           "KNeighborsClassifier__leaf_size",
                                           "KNeighborsClassifier__p",
                                           "KNeighborsClassifier__metric",
                                           "KNeighborsClassifier__n_jobs"])
    sampler = {
        "KNeighborsClassifier__n_neighbors": Parameter("KNeighborsClassifier__n_neighbors", [2, 100], "uniform", "int"),
         "KNeighborsClassifier__weights": Parameter("KNeighborsClassifier__weights", ["uniform", "distance"], "choice", "string"),
         "KNeighborsClassifier__algorithm": Parameter("KNeighborsClassifier__algorithm", ["ball_tree", "kd_tree", "brute", "auto"], "choice", "string"),
         "KNeighborsClassifier__leaf_size": Parameter("KNeighborsClassifier__leaf_size", [10, 50], "uniform", "int"),
         "KNeighborsClassifier__p": Parameter("KNeighborsClassifier__p", [1, 2], "choice", "int"),
         "KNeighborsClassifier__metric": Parameter("KNeighborsClassifier__metric", ["euclidean", "manhattan", "chebyshev", "minkowski"], "choice", "string"),
         "KNeighborsClassifier__n_jobs": Parameter("KNeighborsClassifier__n_jobs", -1, "constant", "int")
    }

    rules = [ChildRule(applied_to = ["KNeighborsClassifier__leaf_size"], parent = "KNeighborsClassifier__algorithm", value = ["ball_tree", "kd_tree"]),
             ChildRule(applied_to = ["KNeighborsClassifier__p"], parent = "KNeighborsClassifier__metric", value = ["minkowski"])
    ]
    return kNeighborsClassifier, sampler, rules


def get_configuration_RadiusNeighborsClassifier():
    RadiusNeighborsClassifier = WorkflowListTask(is_ordered=False, name = "RadiusNeighborsClassifier",
                                  tasks = ["RadiusNeighborsClassifier__radius",
                                           "RadiusNeighborsClassifier__weights",
                                           #"RadiusNeighborsClassifier__algorithm",
                                           #"RadiusNeighborsClassifier__leaf_size",
                                           "RadiusNeighborsClassifier__p",
                                           "RadiusNeighborsClassifier__metric"
                                           ])
    sampler = {
         "RadiusNeighborsClassifier__radius": Parameter("RadiusNeighborsClassifier__radius", [0, 1000], "uniform", "int"),
         "RadiusNeighborsClassifier__weights": Parameter("RadiusNeighborsClassifier__weights", ["uniform", "distance"], "choice", "string"),
         #"RadiusNeighborsClassifier__algorithm": Parameter("RadiusNeighborsClassifier__algorithm", ["ball_tree", "kd_tree", "brute", "auto"], "choice", "string"),
         #"RadiusNeighborsClassifier__leaf_size": Parameter("RadiusNeighborsClassifier__leaf_size", [10, 50], "uniform", "int"),
         "RadiusNeighborsClassifier__p": Parameter("RadiusNeighborsClassifier__p", [1, 2, 3, 4, 5], "choice", "int"),
         #"RadiusNeighborsClassifier__metric": Parameter("RadiusNeighborsClassifier__metric", ["euclidean", "manhattan", "chebyshev", "minkowski"], "choice", "string"),
    }

    rules = [
             # ChildRule(applied_to = ["RadiusNeighborsClassifier__leaf_size"], parent = "RadiusNeighborsClassifier__algorithm", value = ["ball_tree", "kd_tree"]),
             ChildRule(applied_to = ["RadiusNeighborsClassifier__p"], parent = "RadiusNeighborsClassifier__metric", value = ["minkowski"])
    ]
    return RadiusNeighborsClassifier, sampler, rules
