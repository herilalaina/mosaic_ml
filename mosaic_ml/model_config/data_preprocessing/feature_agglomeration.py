from autosklearn.pipeline.components.feature_preprocessing.feature_agglomeration import FeatureAgglomeration
import numpy as np


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("preprocessor:feature_agglomeration:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = FeatureAgglomeration(**list_param)
    return (name, model)
