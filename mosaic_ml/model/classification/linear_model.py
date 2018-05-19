from mosaic.space import Parameter
from mosaic.scenario import ListTask
from mosaic.rules import ChildRule, ValueRule

def get_configuration_LogisticRegression():
    logisticRegression = ListTask(is_ordered=False, name = "LogisticRegression",
                                  tasks = ["LogisticRegression__penalty",
                                           "LogisticRegression__C",
                                           "LogisticRegression__class_weight"])

    sampler = {"LogisticRegression__penalty": Parameter("LogisticRegression__penalty", ["l1", "l2"], "choice", "string"),
             "LogisticRegression__C": Parameter("LogisticRegression__C", [1e-8, 200], "log_uniform", "float"),
             "LogisticRegression__class_weight": Parameter("LogisticRegression__class_weight", "balanced", "constant", "string")}

    return logisticRegression, sampler, []


def get_configuration_SGDClassifier():
    sgdClassifier = ListTask(is_ordered=False, name = "SGDClassifier",
                             tasks = ["SGDClassifier__loss", "SGDClassifier__penalty",
                                     "SGDClassifier__alpha", "SGDClassifier__l1_ratio",
                                     "SGDClassifier__epsilon", "SGDClassifier__learning_rate",
                                     "SGDClassifier__eta0", "SGDClassifier__power_t",
                                     #"SGDClassifier__class_weight", "SGDClassifier__warm_start",
                                     "SGDClassifier__max_iter", "SGDClassifier__tol"])

    sampler = {"SGDClassifier__loss": Parameter("SGDClassifier__loss", ["hinge", "log", "modified_huber", "squared_hinge", "perceptron", "squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"], "choice", "string"),
               "SGDClassifier__penalty": Parameter("SGDClassifier__penalty", ["l1", "l2", "elasticnet"], "choice", "string"),
               "SGDClassifier__alpha": Parameter("SGDClassifier__alpha", [1e-7, 1e-1], "log_uniform", "float"),
               "SGDClassifier__l1_ratio": Parameter("SGDClassifier__l1_ratio", [1e-9, 1], "uniform", "float"),
               "SGDClassifier__epsilon": Parameter("SGDClassifier__epsilon", [1e-5, 1e-1], "uniform", "float"),
               "SGDClassifier__learning_rate": Parameter("SGDClassifier__learning_rate", ["constant", "optimal", "invscaling"], "choice", "string"),
               "SGDClassifier__eta0": Parameter("SGDClassifier__eta0", [1e-7, 1e-1], "uniform", "float"),
               "SGDClassifier__power_t": Parameter("SGDClassifier__power_t", [1e-5, 1], "uniform", "float"),
               "SGDClassifier__class_weight": Parameter("SGDClassifier__class_weight", "balanced", "constant", "string"),
               #"SGDClassifier__warm_start": Parameter("SGDClassifier__warm_start", [True, False], "choice", "bool"),
               "SGDClassifier__max_iter": Parameter("SGDClassifier__max_iter", [1, 100], "uniform", "int"),
               "SGDClassifier__tol": Parameter("SGDClassifier__tol", [1e-5, 1e-1], "log_uniform", "float")}

    rules = [
        # Elastic net constraint
        ChildRule(applied_to = ["SGDClassifier__l1_ratio"], parent = "SGDClassifier__penalty", value = ["elasticnet"]),
        ChildRule(applied_to = ["SGDClassifier__epsilon"], parent = "SGDClassifier__loss", value = ["huber", "epsilon_insensitive", "squared_epsilon_insensitive"]),
        ChildRule(applied_to = ["SGDClassifier__eta0"], parent = "SGDClassifier__learning_rate", value = ["constant", "invscaling"]),
        ChildRule(applied_to = ["SGDClassifier__power_t"], parent = "SGDClassifier__learning_rate", value = ["invscaling"])
    ]

    return sgdClassifier, sampler, rules

def get_configuration_RidgeClassifier():
    ridgeClassifier = ListTask(is_ordered=False, name = "RidgeClassifier",
                             tasks = ["RidgeClassifier__alpha", "RidgeClassifier__max_iter", "RidgeClassifier__class_weight",
                                      "RidgeClassifier__solver"])
    sampler = {
        "RidgeClassifier__alpha": Parameter("RidgeClassifier__alpha", [0.0001, 1], "uniform", "float"),
        "RidgeClassifier__max_iter": Parameter("RidgeClassifier__max_iter", [1, 100], "uniform", "int"),
        "RidgeClassifier__class_weight": Parameter("RidgeClassifier__class_weight", "balanced", "constant", "string"),
        "RidgeClassifier__solver": Parameter("RidgeClassifier__solver", ["auto", "svd", "cholesky", "sparse_cg", "lsqr", "sag"], "choice", "string"),
    }
    rules = []
    return ridgeClassifier, sampler, rules

def get_configuration_Perceptron():
    perceptron = ListTask(is_ordered=False, name = "Perceptron",
                             tasks = ["Perceptron__penalty", "Perceptron__alpha", "Perceptron__max_iter",
                                      "Perceptron__tol", "Perceptron__shuffle", "Perceptron__eta0", "Perceptron__n_jobs",
                                      "Perceptron__class_weight", "Perceptron__warm_start"])
    sampler = {
        "Perceptron__penalty": Parameter("Perceptron__penalty", ["l1", "l2", "elasticnet"], "choice", "string"),
        "Perceptron__alpha": Parameter("Perceptron__alpha", [0.0001, 1], "uniform", "float"),
        "Perceptron__max_iter": Parameter("Perceptron__max_iter", [1, 100], "uniform", "int"),
        "Perceptron__tol": Parameter("Perceptron__tol", None, "constant", "string"),
        "Perceptron__shuffle": Parameter("Perceptron__shuffle", [True, False], "choice", "bool"),
        "Perceptron__eta0": Parameter("Perceptron__eta0", [0.001, 0.3], "uniform", "float"),
        "Perceptron__n_jobs": Parameter("Perceptron__n_jobs", -1, "constant", "int"),
        "Perceptron__class_weight": Parameter("Perceptron__class_weight", "balanced", "constant", "string"),
        "Perceptron__warm_start": Parameter("Perceptron__warm_start", [True, False], "choice", "bool"),
    }

    rules = []
    return perceptron, sampler, rules

def get_configuration_PassiveAggressiveClassifier():
    passiveAggressiveClassifier = ListTask(is_ordered=False, name = "PassiveAggressiveClassifier",
                             tasks = ["PassiveAggressiveClassifier__C",
                                      "PassiveAggressiveClassifier__max_iter",
                                      "PassiveAggressiveClassifier__tol",
                                      #"PassiveAggressiveClassifier__shuffle",
                                      "PassiveAggressiveClassifier__loss",
                                      "PassiveAggressiveClassifier__n_jobs",
                                      #"PassiveAggressiveClassifier__warm_start",
                                      "PassiveAggressiveClassifier__class_weight"])
    sampler = {
        "PassiveAggressiveClassifier__C": Parameter("PassiveAggressiveClassifier__C", [1e-5, 10], "log_uniform", "float"),
        "PassiveAggressiveClassifier__max_iter": Parameter("PassiveAggressiveClassifier__max_iter", [90, 100], "uniform", "int"),
        "PassiveAggressiveClassifier__tol": Parameter("PassiveAggressiveClassifier__tol", [1e-5, 1e-1], "log_uniform", "float"),
        # "PassiveAggressiveClassifier__shuffle": Parameter("PassiveAggressiveClassifier__shuffle", [True, False], "choice", "bool"),
        "PassiveAggressiveClassifier__loss": Parameter("PassiveAggressiveClassifier__loss", ["hinge", "squared_hinge"], "choice", "string"),
        "PassiveAggressiveClassifier__n_jobs": Parameter("PassiveAggressiveClassifier__n_jobs", -1, "constant", "int"),
        #"PassiveAggressiveClassifier__warm_start": Parameter("PassiveAggressiveClassifier__warm_start", [True, False], "choice", "bool"),
        "PassiveAggressiveClassifier__class_weight": Parameter("PassiveAggressiveClassifier__class_weight", "balanced", "constant", "string"),
    }

    rules = []
    return passiveAggressiveClassifier, sampler, rules
