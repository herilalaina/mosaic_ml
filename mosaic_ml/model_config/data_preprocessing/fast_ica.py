from autosklearn.pipeline.components.feature_preprocessing.fast_ica import FastICA


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("preprocessor:fast_ica:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = FastICA(**list_param)
    return (name, model)
