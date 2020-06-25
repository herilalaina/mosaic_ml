
from mosaic_ml.model_config.util import check_for_bool, check_none
from autosklearn.pipeline.components.feature_preprocessing.liblinear_svc_preprocessor import LibLinear_Preprocessor


def get_model(name, config, random_state):
    list_param = {"random_state": random_state,
                  "class_weight": "weighting" if config["class_weight"] == "weighting" else None}
    for k in config:
        if k.startswith("feature_preprocessor:liblinear_svc_preprocessor:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = LibLinear_Preprocessor(**list_param)
    return (name, model)
