import numpy as np

from mosaic_ml.model_config.util import check_for_bool, check_none, convert_multioutput_multiclass_to_multilabel

from autosklearn.pipeline.components.classification.random_forest import RandomForest

def get_model(name, config, random_state):
    list_param = {"random_state": random_state,
                  "class_weight": "weighting" if config["class_weight"] == "weighting" else None}
    for k in config:
        if k.startswith("classifier:random_forest:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = RandomForest(**list_param)
    return (name, model)
