from mosaic.space import Parameter
from mosaic.scenario import ListTask
from mosaic.rules import ChildRule, ValueRule


def get_configuration_MLPClassifier():
    mLPClassifier = ListTask(is_ordered=False, name = "MLPClassifier",
                                  tasks = ["MLPClassifier__hidden_layer_sizes",
                                           "MLPClassifier__activation",
                                           "MLPClassifier__solver",
                                           "MLPClassifier__alpha",
                                           "MLPClassifier__batch_size",
                                           "MLPClassifier__learning_rate",
                                           "MLPClassifier__learning_rate_init",
                                           "MLPClassifier__power_t",
                                           "MLPClassifier__max_iter",
                                           "MLPClassifier__shuffle",
                                           "MLPClassifier__warm_start",
                                           "MLPClassifier__momentum",
                                           "MLPClassifier__nesterovs_momentum",
                                           "MLPClassifier__early_stopping",
                                           "MLPClassifier__validation_fraction",
                                           "MLPClassifier__beta_1",
                                           "MLPClassifier__beta_2",
                                           "MLPClassifier__epsilon"])
    sampler = {
          "MLPClassifier__hidden_layer_sizes": Parameter("MLPClassifier__hidden_layer_sizes", (50, 25, 10), "constant", "tuple"),
          "MLPClassifier__activation": Parameter("MLPClassifier__activation", ["identity", "logistic", "tanh", "relu"], "choice", "string"),
          "MLPClassifier__solver": Parameter("MLPClassifier__solver", ["lbfgs", "sgd", "adam"], "choice", "string"),
          "MLPClassifier__alpha": Parameter("MLPClassifier__alpha", [0.001, 1], "uniform", "float"),
          "MLPClassifier__batch_size": Parameter("MLPClassifier__batch_size", "auto", "constant", "string"),
          "MLPClassifier__learning_rate": Parameter("MLPClassifier__learning_rate", ["constant", "adaptive", "invscaling"], "choice", "string"),
          "MLPClassifier__learning_rate_init": Parameter("MLPClassifier__learning_rate_init", [0.001, 0.3], "uniform","float"),
          "MLPClassifier__power_t": Parameter("MLPClassifier__power_t", [0.1, 0.9], "uniform", "float"),
          "MLPClassifier__max_iter": Parameter("MLPClassifier__max_iter", [1, 100], "uniform", "int"),
          "MLPClassifier__shuffle": Parameter("MLPClassifier__shuffle", [True, False], "choice", "bool"),
          "MLPClassifier__warm_start": Parameter("MLPClassifier__warm_start", [True, False], "choice", "bool"),
          "MLPClassifier__momentum": Parameter("MLPClassifier__momentum", [0, 1], "uniform", "float"),
          "MLPClassifier__nesterovs_momentum": Parameter("MLPClassifier__nesterovs_momentum", [True, False], "choice", "bool"),
          "MLPClassifier__early_stopping": Parameter("MLPClassifier__early_stopping", [True, False], "choice", "bool"),
          "MLPClassifier__validation_fraction": Parameter("MLPClassifier__validation_fraction", [0.1, 0.4], "uniform", "float"),
          "MLPClassifier__beta_1": Parameter("MLPClassifier__beta_1", [0.5, 0.999], "uniform", "float"),
          "MLPClassifier__beta_2": Parameter("MLPClassifier__beta_2", [0.5, 0.999], "uniform", "float"),
          "MLPClassifier__epsilon": Parameter("MLPClassifier__epsilon", 1e-8, "constant", "float"),
    }

    rules = [
        ChildRule(applied_to = ["MLPClassifier__learning_rate"], parent = "MLPClassifier__solver", value = ["sgd"]),
        ChildRule(applied_to = ["MLPClassifier__learning_rate_init"], parent = "MLPClassifier__solver", value = ["sgd", "adam"]),
        ChildRule(applied_to = ["MLPClassifier__power_t"], parent = "MLPClassifier__solver", value = ["sgd"]),
        ChildRule(applied_to = ["MLPClassifier__shuffle"], parent = "MLPClassifier__solver", value = ["sgd", "adam"]),
        ChildRule(applied_to = ["MLPClassifier__momentum"], parent = "MLPClassifier__solver", value = ["sgd"]),
        ChildRule(applied_to = ["MLPClassifier__nesterovs_momentum"], parent = "MLPClassifier__solver", value = ["sgd"]),
        ChildRule(applied_to = ["MLPClassifier__early_stopping"], parent = "MLPClassifier__solver", value = ["sgd", "adam"]),
        ChildRule(applied_to = ["MLPClassifier__validation_fraction"], parent = "MLPClassifier__early_stopping", value = [True]),
        ChildRule(applied_to = ["MLPClassifier__beta_1"], parent = "MLPClassifier__solver", value = ["adam"]),
        ChildRule(applied_to = ["MLPClassifier__beta_2"], parent = "MLPClassifier__solver", value = ["adam"]),
        ChildRule(applied_to = ["MLPClassifier__epsilon"], parent = "MLPClassifier__solver", value = ["adam"])
    ]

    return mLPClassifier, sampler, rules
