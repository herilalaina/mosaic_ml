
import numpy as np
import sklearn.ensemble
from mosaic_ml.model_config.util import check_none

from autosklearn.pipeline.components.classification.gradient_boosting import GradientBoostingClassifier


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:gradient_boosting:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = GradientBoostingClassifier(**list_param)
    return (name, model)
