from autosklearn.pipeline.components.classification.random_forest import RandomForest


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:random_forest:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = RandomForest(**list_param)
    return (name, model)
