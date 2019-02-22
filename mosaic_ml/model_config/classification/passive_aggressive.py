from autosklearn.pipeline.components.classification.passive_aggressive import PassiveAggressive


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:passive_aggressive:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = PassiveAggressive(**list_param)
    return (name, model)
