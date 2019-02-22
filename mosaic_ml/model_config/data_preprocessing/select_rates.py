from autosklearn.pipeline.components.feature_preprocessing.select_rates import SelectRates


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("preprocessor:select_rates:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = SelectRates(**list_param)

    return (name, model)
