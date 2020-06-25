import numpy as np

from autosklearn.pipeline.components.classification.decision_tree import DecisionTree



def get_model(name, config, random_state):
    list_param = {"random_state": random_state,
                  "class_weight": "weighting" if config["class_weight"] == "weighting" else None}
    for k in config:
        if k.startswith("classifier:decision_tree:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = DecisionTree(**list_param)
    return (name, model)
